
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
from mycormarl import state
from mycormarl.fungus.physiology import fungal_maintenance_demand
from mycormarl.params import EnvConfig, SpeciesParams
from mycormarl.growth import grow
from mycormarl.plant.physiology import plant_maintenance_demand
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
from mycormarl.actions import constrain_allocation
from mycormarl.observations import _normalize_if
from mycormarl.state import State


# TODO: consider more complex allocation strategies based on past allocations.
# TODO: decouple constraints from the environment step function.


PLANT = "plant"
FUNGUS = "fungus"
AGENTS = (PLANT, FUNGUS)

class Actions(IntEnum):
    # Trade, growth, maintenance, reproduction fractions.
    trade = 0
    growth = 1
    maintenance = 2
    reproduction = 3


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
                self.config.buffer_power,
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

        obs_dim = 4 # [biomass, carbon_pool, phosphorus_pool, received_trades] normalised

        self.action_set = jnp.array(
            [Actions.trade, Actions.growth, Actions.maintenance, Actions.reproduction]
        )

        self.observation_spaces = {
            PLANT: Box(low=-jnp.inf, high=jnp.inf, shape=(obs_dim,), dtype=jnp.float32),
            FUNGUS: Box(low=-jnp.inf, high=jnp.inf, shape=(obs_dim,), dtype=jnp.float32),
        }
        # Four continuous allocations: trade, growth, maintenance, reproduction.
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

    def _get_obs(
        self,
        state: State,
        last_trade: Optional[Dict[str, chex.Array]] = None,
    ) -> Dict[str, chex.Array]:

        if last_trade is None:
            plant_trade_obs = jnp.zeros_like(state.plant_p_pool)
            fungus_trade_obs = jnp.zeros_like(state.fungus_c_pool)
        else:
            plant_trade_obs = last_trade[PLANT]
            fungus_trade_obs = last_trade[FUNGUS]

        plant_obs = jnp.concatenate([
            _normalize_if(self.config.norm_obs, state.plant_biomass, self.species.plant.biomass_cap),
            _normalize_if(self.config.norm_obs, state.plant_c_pool, state.plant_biomass),
            _normalize_if(self.config.norm_obs, state.plant_p_pool, state.plant_biomass),
            _normalize_if(self.config.norm_obs, plant_trade_obs, state.plant_p_pool),
        ])

        fungus_obs = jnp.concatenate([
            _normalize_if(self.config.norm_obs, state.fungus_biomass, self.species.plant.biomass_cap),
            _normalize_if(self.config.norm_obs, state.fungus_c_pool, state.fungus_biomass),
            _normalize_if(self.config.norm_obs, state.fungus_p_pool, state.fungus_biomass),
            _normalize_if(self.config.norm_obs, fungus_trade_obs, state.fungus_c_pool),
        ])

        return {
            PLANT: plant_obs.astype(jnp.float32),
            FUNGUS: fungus_obs.astype(jnp.float32),
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
            self.config.buffer_power,
        )

        plant_biomass = jnp.array([
            self.species.plant.initial_biomass if self.plant_active else 0.0
        ])
        fungus_biomass = jnp.array([
            self.species.fungus.initial_biomass if self.fungus_active else 0.0
        ])
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
            soil_labile_p=soil_labile_p,
            root_length_density=root_length_density,
            hyphae_length_density=hyphae_length_density,
            cumulative_plant_p_mortality_loss_mg=jnp.array([0.0]),
            cumulative_fungus_p_mortality_loss_mg=jnp.array([0.0]),
            cumulative_plant_p_reproduction_export_mg=jnp.array([0.0]),
            cumulative_fungus_p_reproduction_export_mg=jnp.array([0.0]),
            plant_dead=jnp.array([not self.plant_active]),
            fungus_dead=jnp.array([not self.fungus_active])
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        state = self._initial_state()

        obs = self._get_obs(state)
        return obs, state

    def step_env(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: Dict[str, chex.Array]
        ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:

        plant_operational = jnp.logical_and(
            self.plant_active, jnp.logical_not(state.plant_dead)
        )
        fungus_operational = jnp.logical_and(
            self.fungus_active, jnp.logical_not(state.fungus_dead)
        )
        plant_actions = jnp.where(
            plant_operational,
            constrain_allocation(actions[PLANT]),
            jnp.zeros_like(actions[PLANT]),
        )
        fungus_actions = jnp.where(
            fungus_operational,
            constrain_allocation(actions[FUNGUS]),
            jnp.zeros_like(actions[FUNGUS]),
        )

        # Agents spend only the pools that were observable at the start of the
        # timestep. Newly acquired resources are credited after allocation.
        plant_c_start = state.plant_c_pool
        plant_p_start = state.plant_p_pool
        fungus_c_start = state.fungus_c_pool
        fungus_p_start = state.fungus_p_pool

        # Trade is an allocation choice on the same simplex as the other
        # actions, but it only affects the traded currency for each partner.
        trading_enabled = jnp.logical_and(plant_operational, fungus_operational)
        plant_c_trade_out = jnp.where(
            trading_enabled, plant_actions[Actions.trade] * plant_c_start, 0.0
        )
        fungus_p_trade_out = jnp.where(
            trading_enabled, fungus_actions[Actions.trade] * fungus_p_start, 0.0
        )

        plant_step = self.step_plant(key, state, plant_actions)
        fungus_step = self.step_fungus(key, state, fungus_actions)

        plant_biomass_before_mortality = jnp.clip(
            state.plant_biomass + plant_step["growth"],
            0.0,
            self.species.plant.biomass_cap,
        )
        fungus_biomass_before_mortality = jnp.clip(
            state.fungus_biomass + fungus_step["growth"],
            0.0,
            jnp.inf,
        )
        plant_biomass = jnp.clip(
            state.plant_biomass + plant_step["growth"] - plant_step["maint_deficit_biomass"],
            0.0,
            self.species.plant.biomass_cap,
        )
        fungus_biomass = jnp.clip(
            state.fungus_biomass + fungus_step["growth"] - fungus_step["maint_deficit_biomass"],
            0.0,
            jnp.inf,
        )
        plant_mortality_biomass = jnp.maximum(
            plant_biomass_before_mortality - plant_biomass, 0.0
        )
        fungus_mortality_biomass = jnp.maximum(
            fungus_biomass_before_mortality - fungus_biomass, 0.0
        )

        plant_history_max = jnp.maximum(state.plant_history_max_biomass, plant_biomass)
        fungus_history_max = jnp.maximum(state.fungus_history_max_biomass, fungus_biomass)

        plant_dead = jnp.logical_or(
            state.plant_dead,
            jnp.logical_or(
                not self.plant_active,
                plant_biomass < (
                self.species.plant.death_fraction * jnp.maximum(plant_history_max, 1e-8)
                ),
            ),
        )
        fungus_dead = jnp.logical_or(
            state.fungus_dead,
            jnp.logical_or(
                not self.fungus_active,
                fungus_biomass < (
                self.species.fungus.death_fraction * jnp.maximum(fungus_history_max, 1e-8)
                ),
            ),
        )

        state = state.replace(
            plant_biomass=plant_biomass,
            fungus_biomass=fungus_biomass,
            plant_history_max_biomass=plant_history_max,
            fungus_history_max_biomass=fungus_history_max,
            plant_c_pool=jnp.clip(plant_step["c_pool"] - plant_c_trade_out, 0.0, jnp.inf),
            plant_p_pool=plant_step["p_pool"] + fungus_p_trade_out,
            fungus_c_pool=fungus_step["c_pool"] + plant_c_trade_out,
            fungus_p_pool=jnp.clip(fungus_step["p_pool"] - fungus_p_trade_out, 0.0, jnp.inf),
            cumulative_plant_p_mortality_loss_mg=(
                state.cumulative_plant_p_mortality_loss_mg
                + plant_mortality_biomass * self.species.plant.gamma_p
            ),
            cumulative_fungus_p_mortality_loss_mg=(
                state.cumulative_fungus_p_mortality_loss_mg
                + fungus_mortality_biomass * self.species.fungus.gamma_p
            ),
            cumulative_plant_p_reproduction_export_mg=(
                state.cumulative_plant_p_reproduction_export_mg
                + plant_step["reproduction_p"]
            ),
            cumulative_fungus_p_reproduction_export_mg=(
                state.cumulative_fungus_p_reproduction_export_mg
                + fungus_step["reproduction_p"]
            ),
            plant_dead=plant_dead,
            fungus_dead=fungus_dead,
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

        obs = self._get_obs(
            state,
            last_trade={
                PLANT: fungus_p_trade_out,
                FUNGUS: plant_c_trade_out,
            },
        )

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
                "biomass": state.plant_biomass,
                "c_pool": state.plant_c_pool,
                "p_pool": state.plant_p_pool,
                "trade_out": plant_c_trade_out,
                "trade_in": fungus_p_trade_out,
            },
            FUNGUS: {
                **fungus_step["info"],
                "biomass": state.fungus_biomass,
                "c_pool": state.fungus_c_pool,
                "p_pool": state.fungus_p_pool,
                "trade_out": fungus_p_trade_out,
                "trade_in": plant_c_trade_out,
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
            self.config.buffer_power,
            self.radial_diffusion_conductance,
            self.vertical_diffusion_conductance,
            self.soil_substeps,
            self.config.phosphate_diffusion_coefficient_cm2_s,
            self.config.phosphate_impedance_factor,
            self.config.uptake_reference_time_days,
            self.config.uptake_transition_exponent,
        )

    def step_plant(self, key: jax.Array, state: State, action: jax.Array) -> dict:
        """Step the plant dynamics based on the action allocation.

        Currently, assumes action allocations are valid and sum to 1.0.
        """

        operational = jnp.logical_and(
            self.plant_active, jnp.logical_not(state.plant_dead)
        )
        action = jnp.where(operational, action, jnp.zeros_like(action))
        active_biomass = jnp.where(operational, state.plant_biomass, 0.0)
        growth_alloc = action[Actions.growth]
        maintenance_alloc = action[Actions.maintenance]
        reproduction_alloc = action[Actions.reproduction]

        # Determine the growth based on essential resources.
        growth_c_alloc = growth_alloc * state.plant_c_pool
        growth_p_alloc = growth_alloc * state.plant_p_pool
        growth = grow(
            allocated_c=growth_c_alloc,
            allocated_p=growth_p_alloc,
            grow_c_cost=self.species.plant.gamma_c,
            grow_p_cost=self.species.plant.gamma_p,
            grow_type="essential",
        )
        growth = jnp.minimum(
            growth,
            jnp.maximum(self.species.plant.biomass_cap - state.plant_biomass, 0.0),
        )

        # Determine real resources used for growth, e.g.,
        # if using essential resources, the minimum of the two will determine actual growth.
        growth_c_used = growth * self.species.plant.gamma_c
        growth_p_used = growth * self.species.plant.gamma_p

        # Determine required maintenance resources based on biomass and species traits.
        required_maint_c, required_maint_p = plant_maintenance_demand(
            active_biomass,
            self.species.plant.kappa_c,
            self.species.plant.kappa_p,
            self.config.dt,
        )

        # Calculate allocated maintenance resources.
        allocated_maint_c = maintenance_alloc * state.plant_c_pool
        allocated_maint_p = maintenance_alloc * state.plant_p_pool

        # Calculate actual maintenance resources used, which cannot exceed required amount.
        # Over-allocated resources are returned to their pools.
        maint_c_used = jnp.minimum(allocated_maint_c, required_maint_c)
        maint_p_used = jnp.minimum(allocated_maint_p, required_maint_p)

        # Calculate maintenance resource deficit, if any.
        c_deficit = jnp.maximum(required_maint_c - allocated_maint_c, 0.0)
        p_deficit = jnp.maximum(required_maint_p - allocated_maint_p, 0.0)

        # Calculate the biomass deficit due to maintenance shortfall.
        # Taking the maximum of carbon and phosphorus deficits converted to biomass.
        maint_deficit_biomass = jnp.maximum(
            c_deficit / self.species.plant.gamma_c,
            p_deficit / self.species.plant.gamma_p,
        )

        # Calculate the resources allocated to reproduction.
        reproduction_c = reproduction_alloc * state.plant_c_pool
        reproduction_p = reproduction_alloc * state.plant_p_pool

        # Calculate total resources used for maintenance, reproduction, and growth.
        c_used = maint_c_used + reproduction_c + growth_c_used
        p_used = maint_p_used + reproduction_p + growth_p_used

        # Update the carbon and phosphorus pools after accounting for used resources.
        c_pool = jnp.clip(state.plant_c_pool - c_used, 0.0, jnp.inf)
        p_pool = jnp.clip(state.plant_p_pool - p_used, 0.0, jnp.inf)

        # Get reward for reproduction based on Cobb-Douglas function of carbon and phosphorus.
        # Scale reproduction C and P to be in terms of biomass for reward calculation.
        reproduction_c_scaled = reproduction_c / self.species.plant.gamma_c
        reproduction_p_scaled = reproduction_p / self.species.plant.gamma_p
        reward = self._cobb_douglas(reproduction_c_scaled, reproduction_p_scaled, self.config.alpha)

        info = {
            "growth": growth,
            "maint_c": required_maint_c,
            "maint_p": required_maint_p,
            "maint_c_used": maint_c_used,
            "maint_p_used": maint_p_used,
            "reproduction_c": reproduction_c,
            "reproduction_p": reproduction_p,
            "c_deficit": c_deficit,
            "p_deficit": p_deficit,
        }

        return {
            "growth": growth,
            "maint_deficit_biomass": maint_deficit_biomass,
            "reproduction_p": reproduction_p,
            "c_pool": c_pool,
            "p_pool": p_pool,
            "reward": reward.squeeze(),
            "info": info,
        }

    def step_fungus(self, key: jax.Array, state: State, action: jax.Array) -> dict:
        """Step the fungus dynamics based on the action allocation.

        Currently, assumes action allocations are valid and sum to 1.0.
        """
        operational = jnp.logical_and(
            self.fungus_active, jnp.logical_not(state.fungus_dead)
        )
        action = jnp.where(operational, action, jnp.zeros_like(action))
        active_biomass = jnp.where(operational, state.fungus_biomass, 0.0)
        growth_alloc = action[Actions.growth]
        maint_alloc = action[Actions.maintenance]
        reproduction_alloc = action[Actions.reproduction]

        # Determine the growth based on essential resources.
        growth_c_alloc = growth_alloc * state.fungus_c_pool
        growth_p_alloc = growth_alloc * state.fungus_p_pool
        growth = grow(
            allocated_c=growth_c_alloc,
            allocated_p=growth_p_alloc,
            grow_c_cost=self.species.fungus.gamma_c,
            grow_p_cost=self.species.fungus.gamma_p,
            grow_type="essential",
        )

        # Determine real resources used for growth, e.g.,
        # if using essential resources, the minimum of the two will determine actual growth.
        growth_c_used = growth * self.species.fungus.gamma_c
        growth_p_used = growth * self.species.fungus.gamma_p

        # Determine required maintenance resources based on biomass and species traits.
        required_maint_c, required_maint_p = fungal_maintenance_demand(
            active_biomass,
            self.species.fungus.kappa_c,
            self.species.fungus.kappa_p,
            self.config.dt,
        )

        # Calculate allocated maintenance resources.
        allocated_maint_c = maint_alloc * state.fungus_c_pool
        allocated_maint_p = maint_alloc * state.fungus_p_pool

        # Calculate actual maintenance resources used, which cannot exceed required amount.
        # Over-allocated resources are returned to their pools.
        maint_c_used = jnp.minimum(allocated_maint_c, required_maint_c)
        maint_p_used = jnp.minimum(allocated_maint_p, required_maint_p)

        # Calculate maintenance resource deficit, if any.
        c_deficit = jnp.maximum(required_maint_c - allocated_maint_c, 0.0)
        p_deficit = jnp.maximum(required_maint_p - allocated_maint_p, 0.0)

        # Calculate the biomass deficit due to maintenance shortfall.
        # Taking the maximum of carbon and phosphorus deficits converted to biomass.
        maint_deficit_biomass = jnp.maximum(
            c_deficit / self.species.fungus.gamma_c,
            p_deficit / self.species.fungus.gamma_p,
        )

        # Calculate the resources allocated to reproduction.
        reproduction_c = reproduction_alloc * state.fungus_c_pool
        reproduction_p = reproduction_alloc * state.fungus_p_pool

        # Calculate total resources used for maintenance, reproduction, and growth.
        c_used = maint_c_used + reproduction_c + growth_c_used
        p_used = maint_p_used + reproduction_p + growth_p_used

        # Update the carbon and phosphorus pools after accounting for used resources.
        c_pool = jnp.clip(state.fungus_c_pool - c_used, 0.0, jnp.inf)
        p_pool = jnp.clip(state.fungus_p_pool - p_used, 0.0, jnp.inf)

        # Get reward for reproduction based on Cobb-Douglas function of carbon and phosphorus.
        # Scale reproduction C and P to be in terms of biomass for reward calculation.
        reproductive_c = reproduction_c / self.species.fungus.gamma_c
        reproductive_p = reproduction_p / self.species.fungus.gamma_p
        reward = self._cobb_douglas(reproductive_c, reproductive_p, self.config.alpha)

        info = {
            "growth": growth,
            "maint_c": required_maint_c,
            "maint_p": required_maint_p,
            "maint_c_used": maint_c_used,
            "maint_p_used": maint_p_used,
            "reproduction_c": reproduction_c,
            "reproduction_p": reproduction_p,
            "c_deficit": c_deficit,
            "p_deficit": p_deficit,
        }

        return {
            "growth": growth,
            "maint_deficit_biomass": maint_deficit_biomass,
            "reproduction_p": reproduction_p,
            "c_pool": c_pool,
            "p_pool": p_pool,
            "reward": reward.squeeze(),
            "info": info,
        }

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
        validate_linear_buffer_parameters(config.theta_water, config.buffer_power)
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
        if not isinstance(config.norm_obs, bool):
            raise ValueError("norm_obs must be a boolean")
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
