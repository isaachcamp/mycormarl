

import chex
from flax import struct


@struct.dataclass
class State:
    """Full environment state.

    Arrays are stored explicitly so the model stays JAX-friendly and easy to
    extend.

    Cumulative P loss fields are stored to ensure P conservation.
    """

    plant_biomass: chex.Array # shape (n_plant_agents,)
    fungus_biomass: chex.Array # shape (n_fungus_agents,)
    plant_history_max_biomass: chex.Array # shape (n_plant_agents,)
    fungus_history_max_biomass: chex.Array # shape (n_fungus_agents,)
    plant_c_pool: chex.Array # shape (n_plant_agents,)
    plant_p_pool: chex.Array # shape (n_plant_agents,)
    fungus_c_pool: chex.Array # shape (n_fungus_agents,)
    fungus_p_pool: chex.Array # shape (n_fungus_agents,)
    plant_last_p_received: chex.Array  # shape (n_plant_agents,)
    fungus_last_c_received: chex.Array  # shape (n_fungus_agents,)
    soil_labile_p: chex.Array  # µmol P per cell; shape (n_r, n_z)
    root_length_density: chex.Array  # cm cm^-3; shape (n_r, n_z)
    hyphae_length_density: chex.Array  # cm cm^-3; shape (n_r, n_z)
    cumulative_plant_p_mortality_loss_mg: chex.Array  # shape (n_plant_agents,)
    cumulative_fungus_p_mortality_loss_mg: chex.Array  # shape (n_fungus_agents,)
    cumulative_plant_p_maintenance_loss_mg: chex.Array  # shape (n_plant_agents,)
    cumulative_fungus_p_maintenance_loss_mg: chex.Array  # shape (n_fungus_agents,)
    cumulative_plant_p_reproduction_export_mg: chex.Array  # shape (n_plant_agents,)
    cumulative_fungus_p_reproduction_export_mg: chex.Array  # shape (n_fungus_agents,)
    plant_dead: chex.Array # shape (n_plant_agents,)
    fungus_dead: chex.Array # shape (n_fungus_agents,)
    step: int
    terminal: bool


MycorMarlState = State
