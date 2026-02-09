
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

# TODO: get rid of default factory if using YMAL config file.

@jdc.pytree_dataclass
class AgentState:
    """
    Represents the state of a single agent.
    
    species_id: int
        0 for tree, 1 for fungus.
    health: float
        health of the agent.
    biomass: float
        biomass of the agent.
    phosphorus: int 
        number of phosphorus the agent has.
    sugars: int 
        number of sugars the agent has.
    """
    species_id: jax.Array # 0 for tree, 1 for fungus
    health: jax.Array = jdc.field(default_factory=lambda: jnp.array(100.0))
    biomass: jax.Array = jdc.field(default_factory=lambda: jnp.array(0.1))
    phosphorus: jax.Array = jdc.field(default_factory=lambda: jnp.array(100.))
    sugars: jax.Array = jdc.field(default_factory=lambda: jnp.array(50.))
    p_uptake_efficiency: jax.Array = jdc.field(default_factory=lambda: jnp.array(0.0))
