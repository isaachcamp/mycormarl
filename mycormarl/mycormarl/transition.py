import chex
from flax import struct


@struct.dataclass
class Transition:
    """Algorithm-independent facts for one agent's completed environment step."""

    requested_action: chex.Array
    realised_action: chex.Array
    operational_at_start: chex.Array
    operational_at_end: chex.Array
    allocation_executed: chex.Array
    trade_executed: chex.Array
    truncated: chex.Array
    final_observation: chex.Array
