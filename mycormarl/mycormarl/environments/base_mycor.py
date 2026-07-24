
from __future__ import annotations
import math
from typing import Dict, Tuple, Optional
from functools import partial
from enum import IntEnum

import chex
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box
import jax
import jax.numpy as jnp

from mycormarl import fungus, plant
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.growth import grow
from mycormarl.soil import (
    axisymmetric_cylindrical_cell_volumes,
    axisymmetric_diffusion_conductances,
    axisymmetric_edges_from_intervals,
    axisymmetric_radial_face_areas,
    axisymmetric_vertical_face_areas,
    evolve_soil_p,
    explicit_diffusion_cfl_seconds,
    initial_labile_p_from_micromolar,
    required_diffusion_substeps,
    validate_axisymmetric_grid_parameters,
    validate_diffusion_parameters,
    validate_linear_buffer_parameters,
)
from mycormarl.plant import photosynthesise
from mycormarl.observations import actor_observation
from mycormarl.state import State


# TODO: consider more complex allocation strategies based on past allocations.
# TODO: decouple constraints from the environment step function.


PLANT = "plant"
FUNGUS = "fungus"
AGENTS = (PLANT, FUNGUS)

class Actions(IntEnum):
    # Physical action: independent trade plus a biological allocation simplex.
    trade = 0
    growth = 1
    reproduction = 2
    reserve = 3


class BaseMycorMarl(MultiAgentEnv):
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        species: Optional[SpeciesParams] = None,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        super().__init__(num_agents=2)
        self.config = config or EnvConfig()
        self.agents = list(AGENTS)
        self.plant_active = self.config.consumer_mode != "fungus-only"
        self.fungus_active = self.config.consumer_mode != "plant-only"

        if species is not None:
            self.species = species
        else:
            raise ValueError("Species parameters must be provided.")

        plant.validate_plant_growth_geometry_traits(self.species.plant)
        fungus.validate_fungus_growth_geometry_traits(self.species.fungus)
        self._validate_soil_config(self.config)
        self.r_edges, self.z_edges = axisymmetric_edges_from_intervals(
            self.config.soil_radius_cm,
            self.config.soil_depth_cm,
            self.config.radial_interval_cm,
            self.config.depth_interval_cm,
        )
        self.cell_volumes = axisymmetric_cylindrical_cell_volumes(
            self.r_edges, self.z_edges
        )
        self.radial_face_areas = axisymmetric_radial_face_areas(
            self.r_edges, self.z_edges
        )
        self.vertical_face_areas = axisymmetric_vertical_face_areas(
            self.r_edges, self.z_edges
        )
        (
            self.radial_diffusion_conductance,
            self.vertical_diffusion_conductance,
        ) = axisymmetric_diffusion_conductances(
            self.r_edges,
            self.z_edges,
            self.radial_face_areas,
            self.vertical_face_areas,
            self.config.phosphate_diffusion_coefficient_cm2_s,
            self.config.theta_water,
            self.config.phosphate_impedance_factor,
        )
        self.diffusion_cfl_seconds = float(
            explicit_diffusion_cfl_seconds(
                self.cell_volumes,
                self.config.theta_water,
                self.config.b_p,
                self.radial_diffusion_conductance,
                self.vertical_diffusion_conductance,
            )
        )
        self.soil_substeps = required_diffusion_substeps(
            self.config.dt,
            self.diffusion_cfl_seconds,
            self.config.diffusion_cfl_safety,
        )
        self.soil_substep_days = self.config.dt / self.soil_substeps
        self.grid_shape = (
            self.r_edges.shape[0] - 1,
            self.z_edges.shape[0] - 1,
        )

        obs_dim = 5

        self.action_set = jnp.array(
            [Actions.trade, Actions.growth, Actions.reproduction, Actions.reserve]
        )

        self.observation_spaces = {
            PLANT: Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=jnp.float32),
            FUNGUS: Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=jnp.float32),
        }
        # Physical action: trade, growth, reproduction, reserve.
        self.action_spaces = {
            PLANT: Box(low=0.0, high=1.0, shape=self.action_set.shape, dtype=jnp.float32),
            FUNGUS: Box(low=0.0, high=1.0, shape=self.action_set.shape, dtype=jnp.float32),
        }

        # Environment parameters
        self.max_episode_steps = (
            self.config.max_steps
            if max_episode_steps is None
            else max_episode_steps
        )

    @property
    def agent_classes(self) -> dict:
        return {PLANT: [PLANT], FUNGUS: [FUNGUS]}

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Reconstruct bounded actor observations entirely from environment state."""
        # Calculate reference biomasses for normalization.
        # Use 0.5 * maximum so that normed obs is also half-maximum.
        plant_biomass_reference = 0.5 * self.species.plant.biomass_cap

        # Calculate ref biomass based on maximum radius possible constrained by
        # grid boundaries.
        fungus_biomass_reference = (
            0.5
            * fungus.fungal_biomass_for_colony_radius(
                self.config.soil_radius_cm,
                self.species.fungus,
            )
        )

        association = (
            jnp.asarray(self.plant_active and self.fungus_active)
            & ~state.plant_dead
            & ~state.fungus_dead
        )

        return {
            PLANT: actor_observation(
                biomass=state.plant_biomass,
                biomass_reference=plant_biomass_reference,
                c_pool=state.plant_c_pool,
                gamma_c=self.species.plant.gamma_c,
                p_pool=state.plant_p_pool,
                gamma_p=self.species.plant.gamma_p,
                last_received=state.plant_last_p_received,
                maintenance_need=(
                    self.species.plant.kappa_p
                    * state.plant_biomass
                    * self.config.dt
                ),
                association=association,
                operational=~state.plant_dead,
            ),
            FUNGUS: actor_observation(
                biomass=state.fungus_biomass,
                biomass_reference=fungus_biomass_reference,
                c_pool=state.fungus_c_pool,
                gamma_c=self.species.fungus.gamma_c,
                p_pool=state.fungus_p_pool,
                gamma_p=self.species.fungus.gamma_p,
                last_received=state.fungus_last_c_received,
                maintenance_need=(
                    self.species.fungus.kappa_c
                    * state.fungus_biomass
                    * self.config.dt
                ),
                association=association,
                operational=~state.fungus_dead,
            ),
        }

    def _initial_state(self) -> State:
        """Build biological pools, P state, and biomass-derived geometry.

        Grid geometry created during environment construction is combined with
        configured solution µM and buffering to store ``soil_labile_p``.
        Concentration remains derived; initial biomass is converted into 2D
        root/hyphal density; cumulative P-loss/export diagnostics start at
        zero.
        """
        soil_labile_p = initial_labile_p_from_micromolar(
            self.r_edges,
            self.z_edges,
            self.config.initial_solution_p_um,
            self.config.topsoil_depth_cm,
            self.config.theta_water,
            self.config.b_p,
        )

        # Positive initial biomass only if the organism is active.
        plant_biomass = jnp.array([
            self.species.plant.initial_biomass if self.plant_active else 0.0
        ])
        fungus_biomass = jnp.array([
            self.species.fungus.initial_biomass if self.fungus_active else 0.0
        ])

        # Calculate root density fields from initial biomass.
        root_length_density = plant.density_field_from_biomass(
            plant_biomass,
            self.species.plant,
            self.r_edges,
            self.z_edges,
        )
        hyphae_length_density = fungus.density_field_from_biomass(
            fungus_biomass,
            self.species.fungus,
            self.r_edges,
            self.z_edges,
        )

        return State(
            terminal=False,
            step=0,
            plant_biomass=plant_biomass,
            fungus_biomass=fungus_biomass,
            plant_history_max_biomass=plant_biomass,
            fungus_history_max_biomass=fungus_biomass,
            plant_c_pool=jnp.array([
                self.species.plant.initial_c_pool if self.plant_active else 0.0
            ]),
            plant_p_pool=jnp.array([
                self.species.plant.initial_p_pool if self.plant_active else 0.0
            ]),
            fungus_c_pool=jnp.array([
                self.species.fungus.initial_c_pool if self.fungus_active else 0.0
            ]),
            fungus_p_pool=jnp.array([
                self.species.fungus.initial_p_pool if self.fungus_active else 0.0
            ]),
            plant_last_p_received=jnp.array([0.0]),
            fungus_last_c_received=jnp.array([0.0]),
            soil_labile_p=soil_labile_p,
            root_length_density=root_length_density,
            hyphae_length_density=hyphae_length_density,
            cumulative_plant_p_mortality_loss_mg=jnp.array([0.0]),
            cumulative_fungus_p_mortality_loss_mg=jnp.array([0.0]),
            cumulative_plant_p_maintenance_loss_mg=jnp.array([0.0]),
            cumulative_fungus_p_maintenance_loss_mg=jnp.array([0.0]),
            cumulative_plant_p_reproduction_export_mg=jnp.array([0.0]),
            cumulative_fungus_p_reproduction_export_mg=jnp.array([0.0]),
            plant_dead=jnp.array([not self.plant_active]),
            fungus_dead=jnp.array([not self.fungus_active])
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        state = self._initial_state()

        obs = self.get_obs(state)
        return obs, state

    def step_env(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: Dict[str, chex.Array]
        ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Apply a valid Physical action to the environment state.

        Checks plant/fungus operational status, pays maintenance, applies growth and
        reproduction, and updates the soil P field. Returns the next observation, state,
        rewards, dones, and infos. The labile P field is a conserved quantity.

        # TODO: decrease root/hyphal density with biomass loss.
        """
        plant_operational_at_start = jnp.logical_and(
            self.plant_active, jnp.logical_not(state.plant_dead)
        )
        fungus_operational_at_start = jnp.logical_and(
            self.fungus_active, jnp.logical_not(state.fungus_dead)
        )

        plant_maintenance = self._pay_maintenance(
            state.plant_biomass,
            state.plant_c_pool,
            state.plant_p_pool,
            self.species.plant,
            plant_operational_at_start,
        )
        fungus_maintenance = self._pay_maintenance(
            state.fungus_biomass,
            state.fungus_c_pool,
            state.fungus_p_pool,
            self.species.fungus,
            fungus_operational_at_start,
        )

        plant_biomass_after_maintenance = plant_maintenance["biomass"]
        fungus_biomass_after_maintenance = fungus_maintenance["biomass"]

        # Check biomass has not fallen below death threshold. If so, mark dead and zero pools.
        plant_dead = jnp.logical_or(
            state.plant_dead,
            jnp.logical_or(
                not self.plant_active,
                plant_biomass_after_maintenance
                < self.species.plant.death_fraction
                * jnp.maximum(state.plant_history_max_biomass, 1e-8),
            ),
        )
        fungus_dead = jnp.logical_or(
            state.fungus_dead,
            jnp.logical_or(
                not self.fungus_active,
                fungus_biomass_after_maintenance
                < self.species.fungus.death_fraction
                * jnp.maximum(state.fungus_history_max_biomass, 1e-8),
            ),
        )

        # Re-check operational status after maintenance and death.
        plant_operational_at_end = jnp.logical_and(
            self.plant_active, jnp.logical_not(plant_dead)
        )
        fungus_operational_at_end = jnp.logical_and(
            self.fungus_active, jnp.logical_not(fungus_dead)
        )

        # Bilateral trade is resolved after maintenance. If either organism died
        # while paying maintenance, neither proposed transfer leaves its owner.
        trading_enabled = jnp.logical_and(
            plant_operational_at_end, fungus_operational_at_end
        )
        plant_c_trade_proposed = jnp.where(
            plant_operational_at_start,
            actions[PLANT][Actions.trade] * plant_maintenance["c_pool"],
            0.0,
        )
        fungus_p_trade_proposed = jnp.where(
            fungus_operational_at_start,
            actions[FUNGUS][Actions.trade] * fungus_maintenance["p_pool"],
            0.0,
        )
        plant_c_trade_out = jnp.where(
            trading_enabled,
            plant_c_trade_proposed,
            0.0,
        )
        fungus_p_trade_out = jnp.where(
            trading_enabled,
            fungus_p_trade_proposed,
            0.0,
        )

        # Add a flag to indicate whether a trade was proposed but cancelled due
        # to one party dying or being non-operational.
        trade_cancelled = jnp.logical_and(
            jnp.logical_and(
                plant_operational_at_start, fungus_operational_at_start
            ),
            jnp.logical_and(
                jnp.logical_not(trading_enabled),
                jnp.logical_or(
                    plant_c_trade_proposed > 0.0,
                    fungus_p_trade_proposed > 0.0,
                ),
            ),
        )

        # Update state with post-maintenance biomass and pools, then apply growth
        # and reproduction.
        allocation_state = state.replace(
            plant_biomass=plant_biomass_after_maintenance,
            fungus_biomass=fungus_biomass_after_maintenance,
            plant_c_pool=plant_maintenance["c_pool"] - plant_c_trade_out,
            plant_p_pool=plant_maintenance["p_pool"],
            fungus_c_pool=fungus_maintenance["c_pool"],
            fungus_p_pool=fungus_maintenance["p_pool"] - fungus_p_trade_out,
            plant_dead=plant_dead,
            fungus_dead=fungus_dead,
        )
        # Perform allocations.
        plant_step = self.step_plant(key, allocation_state, actions[PLANT])
        fungus_step = self.step_fungus(key, allocation_state, actions[FUNGUS])

        plant_biomass = jnp.minimum(
            plant_biomass_after_maintenance + plant_step["growth"],
            self.species.plant.biomass_cap,
        )
        fungus_biomass = fungus_biomass_after_maintenance + fungus_step["growth"]
        plant_history_max = jnp.maximum(state.plant_history_max_biomass, plant_biomass)
        fungus_history_max = jnp.maximum(state.fungus_history_max_biomass, fungus_biomass)

        state = allocation_state.replace(
            plant_biomass=plant_biomass,
            fungus_biomass=fungus_biomass,
            plant_history_max_biomass=plant_history_max,
            fungus_history_max_biomass=fungus_history_max,
            plant_c_pool=plant_step["c_pool"],
            plant_p_pool=plant_step["p_pool"] + fungus_p_trade_out,
            fungus_c_pool=fungus_step["c_pool"] + plant_c_trade_out,
            fungus_p_pool=fungus_step["p_pool"],
            plant_last_p_received=jnp.where(
                plant_operational_at_end, fungus_p_trade_out, 0.0
            ),
            fungus_last_c_received=jnp.where(
                fungus_operational_at_end, plant_c_trade_out, 0.0
            ),
            cumulative_plant_p_mortality_loss_mg=(
                state.cumulative_plant_p_mortality_loss_mg
                + plant_maintenance["mortality_biomass"]
                * self.species.plant.gamma_p
            ),
            cumulative_fungus_p_mortality_loss_mg=(
                state.cumulative_fungus_p_mortality_loss_mg
                + fungus_maintenance["mortality_biomass"]
                * self.species.fungus.gamma_p
            ),
            cumulative_plant_p_maintenance_loss_mg=(
                state.cumulative_plant_p_maintenance_loss_mg
                + plant_maintenance["p_used"]
            ),
            cumulative_fungus_p_maintenance_loss_mg=(
                state.cumulative_fungus_p_maintenance_loss_mg
                + fungus_maintenance["p_used"]
            ),
            cumulative_plant_p_reproduction_export_mg=(
                state.cumulative_plant_p_reproduction_export_mg
                + plant_step["reproduction_p"]
            ),
            cumulative_fungus_p_reproduction_export_mg=(
                state.cumulative_fungus_p_reproduction_export_mg
                + fungus_step["reproduction_p"]
            ),
        )

        # New resources acquired during the transition are added after
        # allocation, so they are first visible to the policy at t + 1.
        plant_carbon_fixed = jnp.where(
            state.plant_dead,
            0.0,
            photosynthesise(state, self.species.plant, self.config),
        )
        state = state.replace(plant_c_pool=state.plant_c_pool + plant_carbon_fixed)

        # Density fields should reflect the post-allocation biomass before soil
        # uptake is calculated.
        state = state.replace(
            root_length_density=plant.density_field_from_biomass(
                jnp.where(state.plant_dead, 0.0, state.plant_biomass),
                self.species.plant,
                self.r_edges,
                self.z_edges,
            ),
            hyphae_length_density=fungus.density_field_from_biomass(
                jnp.where(state.fungus_dead, 0.0, state.fungus_biomass),
                self.species.fungus,
                self.r_edges,
                self.z_edges,
            ),
        )
        state = self.step_phosphorus_field(state)

        state = state.replace(step=state.step + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)

        rewards = {
            PLANT: plant_step["reward"],
            FUNGUS: fungus_step["reward"],
        }
        dones = {
            PLANT: plant_dead.squeeze(),
            FUNGUS: fungus_dead.squeeze(),
            "__all__": done,
        }
        infos = {
            PLANT: {
                **plant_step["info"],
                **plant_maintenance["info"],
                "biomass": state.plant_biomass,
                "c_pool": state.plant_c_pool,
                "p_pool": state.plant_p_pool,
                "proposed_trade_out": plant_c_trade_proposed,
                "trade_out": plant_c_trade_out,
                "trade_in": fungus_p_trade_out,
                "trade_cancelled": trade_cancelled,
            },
            FUNGUS: {
                **fungus_step["info"],
                **fungus_maintenance["info"],
                "biomass": state.fungus_biomass,
                "c_pool": state.fungus_c_pool,
                "p_pool": state.fungus_p_pool,
                "proposed_trade_out": fungus_p_trade_proposed,
                "trade_out": fungus_p_trade_out,
                "trade_in": plant_c_trade_out,
                "trade_cancelled": trade_cancelled,
            },
        }

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            rewards,
            dones,
            infos,
        )

    def step_phosphorus_field(self, state: State) -> State:
        """Apply cached diffusion and blended P uptake to the soil state."""
        return evolve_soil_p(
            state,
            self.config.dt,
            self.species,
            self.cell_volumes,
            self.config.theta_water,
            self.config.b_p,
            self.radial_diffusion_conductance,
            self.vertical_diffusion_conductance,
            self.soil_substeps,
            self.config.phosphate_diffusion_coefficient_cm2_s,
            self.config.phosphate_impedance_factor,
            self.config.uptake_reference_time_days,
            self.config.uptake_transition_exponent,
        )

    def _pay_maintenance(
        self,
        biomass: chex.Array,
        c_pool: chex.Array,
        p_pool: chex.Array,
        traits,
        operational: chex.Array,
    ) -> dict:
        """Pay unavoidable maintenance from start-of-step free pools."""
        active_biomass = jnp.where(operational, biomass, 0.0)
        required_c = traits.kappa_c * active_biomass * self.config.dt
        required_p = traits.kappa_p * active_biomass * self.config.dt
        c_used = jnp.minimum(c_pool, required_c)
        p_used = jnp.minimum(p_pool, required_p)
        c_deficit = required_c - c_used
        p_deficit = required_p - p_used
        mortality_biomass = jnp.maximum(
            c_deficit / traits.gamma_c,
            p_deficit / traits.gamma_p,
        )
        mortality_biomass = jnp.minimum(mortality_biomass, active_biomass)
        return {
            "biomass": jnp.where(
                operational, biomass - mortality_biomass, biomass
            ),
            "c_pool": jnp.where(operational, c_pool - c_used, c_pool),
            "p_pool": jnp.where(operational, p_pool - p_used, p_pool),
            "p_used": p_used,
            "mortality_biomass": mortality_biomass,
            "info": {
                "maint_c": required_c,
                "maint_p": required_p,
                "maint_c_used": c_used,
                "maint_p_used": p_used,
                "c_deficit": c_deficit,
                "p_deficit": p_deficit,
            },
        }

    def _apply_allocation(
        self,
        biomass: chex.Array,
        c_pool: chex.Array,
        p_pool: chex.Array,
        action: chex.Array,
        traits,
        operational: chex.Array,
        biomass_cap: float | None = None,
    ) -> dict:
        """Apply growth/reproduction/reserve fractions to disposable C and P."""
        action = jnp.where(operational, action, jnp.zeros_like(action))
        growth_alloc = action[Actions.growth]
        reproduction_alloc = action[Actions.reproduction]

        growth_c_alloc = growth_alloc * c_pool
        growth_p_alloc = growth_alloc * p_pool
        growth = grow(
            allocated_c=growth_c_alloc,
            allocated_p=growth_p_alloc,
            grow_c_cost=traits.gamma_c,
            grow_p_cost=traits.gamma_p,
            grow_type="essential",
        )
        if biomass_cap is not None:
            growth = jnp.minimum(
                growth,
                jnp.maximum(biomass_cap - biomass, 0.0),
            )
        growth_c_used = growth * traits.gamma_c
        growth_p_used = growth * traits.gamma_p
        reproduction_c = reproduction_alloc * c_pool
        reproduction_p = reproduction_alloc * p_pool
        remaining_c_pool = c_pool - reproduction_c - growth_c_used
        remaining_p_pool = p_pool - reproduction_p - growth_p_used
        reproduction_c_scaled = reproduction_c / traits.gamma_c
        reproduction_p_scaled = reproduction_p / traits.gamma_p
        reward = self._cobb_douglas(reproduction_c_scaled, reproduction_p_scaled, self.config.alpha)
        info = {
            "growth": growth,
            "growth_c_allocated": growth_c_alloc,
            "growth_p_allocated": growth_p_alloc,
            "growth_c_used": growth_c_used,
            "growth_p_used": growth_p_used,
            "reproduction_c": reproduction_c,
            "reproduction_p": reproduction_p,
        }
        return {
            "growth": growth,
            "reproduction_p": reproduction_p,
            "c_pool": remaining_c_pool,
            "p_pool": remaining_p_pool,
            "reward": reward.squeeze(),
            "info": info,
        }

    def step_plant(self, key: jax.Array, state: State, action: jax.Array) -> dict:
        """Apply a valid Physical action to the plant's disposable pools."""
        operational = jnp.logical_and(
            self.plant_active, jnp.logical_not(state.plant_dead)
        )
        return self._apply_allocation(
            state.plant_biomass,
            state.plant_c_pool,
            state.plant_p_pool,
            action,
            self.species.plant,
            operational,
            biomass_cap=self.species.plant.biomass_cap,
        )

    def step_fungus(self, key: jax.Array, state: State, action: jax.Array) -> dict:
        """Apply a valid Physical action to the fungus's disposable pools."""
        operational = jnp.logical_and(
            self.fungus_active, jnp.logical_not(state.fungus_dead)
        )
        return self._apply_allocation(
            state.fungus_biomass,
            state.fungus_c_pool,
            state.fungus_p_pool,
            action,
            self.species.fungus,
            operational,
        )

    @staticmethod
    def _validate_soil_config(config: EnvConfig) -> None:
        """Validate all reset-critical soil scalars before allocating arrays.

        This is the configuration boundary for grid geometry, topsoil extent,
        non-negative initial solution P, linear buffering, diffusion, and CFL
        safety. Numerical helpers can therefore stay branch-free and
        JAX-compatible.
        """
        validate_axisymmetric_grid_parameters(
            config.soil_radius_cm,
            config.soil_depth_cm,
            config.radial_interval_cm,
            config.depth_interval_cm,
        )
        validate_linear_buffer_parameters(config.theta_water, config.b_p)
        validate_diffusion_parameters(
            config.phosphate_diffusion_coefficient_cm2_s,
            config.theta_water,
            config.phosphate_impedance_factor,
            config.diffusion_cfl_safety,
        )
        if not math.isfinite(config.dt) or config.dt <= 0.0:
            raise ValueError("dt must be finite and greater than zero")
        if config.consumer_mode not in {"mixed", "plant-only", "fungus-only"}:
            raise ValueError(
                "consumer_mode must be 'mixed', 'plant-only', or 'fungus-only'"
            )
        if not math.isfinite(config.alpha) or not 0.0 <= config.alpha <= 1.0:
            raise ValueError("alpha must be finite and within [0, 1]")
        if (
            isinstance(config.max_steps, bool)
            or not isinstance(config.max_steps, int)
            or config.max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")
        if (
            not math.isfinite(config.uptake_reference_time_days)
            or config.uptake_reference_time_days <= 0.0
        ):
            raise ValueError(
                "uptake_reference_time_days must be finite and greater than zero"
            )
        if (
            not math.isfinite(config.uptake_transition_exponent)
            or config.uptake_transition_exponent <= 0.0
        ):
            raise ValueError(
                "uptake_transition_exponent must be finite and greater than zero"
            )
        if not math.isfinite(config.topsoil_depth_cm) or not (
            0.0 <= config.topsoil_depth_cm <= config.soil_depth_cm
        ):
            raise ValueError(
                "topsoil_depth_cm must be finite and within the soil domain"
            )
        if not math.isfinite(config.initial_solution_p_um) or (
            config.initial_solution_p_um < 0.0
        ):
            raise ValueError(
                "initial_solution_p_um must be finite and non-negative"
            )

    def _cobb_douglas(self, c: chex.Array, p: chex.Array, alpha: float) -> chex.Array:
        """Compute Cobb-Douglas rewarding for reproduction based on phosphorus and carbon."""
        return (c ** alpha) * (p ** (1 - alpha))

    def is_terminal(self, state: State) -> chex.Array:
        return state.terminal | \
               (state.step >= self.max_episode_steps) | \
               jnp.all(jnp.concatenate([state.plant_dead, state.fungus_dead]))
