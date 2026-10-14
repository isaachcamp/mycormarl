
from dataclasses import dataclass
import math
from typing import Dict, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from flax import struct
import optax

from mycormarl.environments.base_mycor import FUNGUS, PLANT
from mycormarl.random_streams import RandomStreamContract
from mycormarl.trade_only import fixed_allocation_rate_action
from mycormarl.transition import Transition


class PolicyParameters(NamedTuple):
    """Parameters of the factorised Gaussian latent policy."""

    trade_loc: jax.Array
    trade_log_std: jax.Array
    biological_rate_loc: jax.Array
    biological_rate_log_std: jax.Array


class PPOStepFields(NamedTuple):
    """Learning controls derived from one environment transition."""

    critic_valid: jax.Array
    biological_rate_actor_valid: jax.Array
    trade_actor_valid: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    bootstrap_valid: jax.Array
    gae_trace_continues: jax.Array
    bootstrap_observation: jax.Array


def transition_to_ppo_fields(
    transition: Transition,
    *,
    finite_horizon_returns: bool = False,
) -> PPOStepFields:
    """Convert algorithm-independent lifecycle facts into PPO controls.

    Continuing tasks bootstrap through administrative truncations. A declared
    finite-horizon return instead treats that same boundary as terminal.
    """
    critic_valid = transition.operational_at_start
    terminated = transition.operational_at_start & ~transition.operational_at_end
    return PPOStepFields(
        critic_valid=critic_valid,
        biological_rate_actor_valid=transition.allocation_executed,
        trade_actor_valid=transition.trade_executed,
        terminated=terminated,
        truncated=transition.truncated,
        bootstrap_valid=(
            transition.operational_at_end
            & ~(finite_horizon_returns & transition.truncated)
        ),
        gae_trace_continues=(
            critic_valid & ~terminated & ~transition.truncated
        ),
        bootstrap_observation=transition.final_observation,
    )


def discount_from_half_life(
    dt_days: float,
    half_life_days: float | None,
) -> float:
    """Convert a physical reward half-life to one environment-step discount."""
    dt_days = float(dt_days)
    if not math.isfinite(dt_days) or dt_days <= 0.0:
        raise ValueError("dt_days must be positive")
    if half_life_days is None:
        return 1.0
    half_life_days = float(half_life_days)
    if half_life_days == math.inf:
        return 1.0
    if not math.isfinite(half_life_days) or half_life_days <= 0.0:
        raise ValueError("half_life_days must be positive, infinite, or None")
    return math.exp(-math.log(2.0) * dt_days / half_life_days)


def calculate_gae(
    *,
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_values: jax.Array,
    critic_valid: jax.Array,
    bootstrap_valid: jax.Array,
    gae_trace_continues: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Calculate masked GAE from explicit bootstrap and trace controls."""

    def _step(next_advantage, step):
        reward, value, bootstrap_value, valid, bootstrap, trace = step
        delta = reward + gamma * bootstrap * bootstrap_value - value
        advantage = valid * (
            delta + gamma * gae_lambda * trace * next_advantage
        )
        return advantage, advantage

    initial_advantage = jnp.zeros_like(values[-1])
    _, advantages = jax.lax.scan(
        _step,
        initial_advantage,
        (
            rewards,
            values,
            bootstrap_values,
            critic_valid,
            bootstrap_valid,
            gae_trace_continues,
        ),
        reverse=True,
    )
    return advantages, advantages + values


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Return the mean over valid samples, or zero when none are valid."""
    mask = jnp.asarray(mask, dtype=bool)
    weights = mask.astype(values.dtype)
    count = jnp.sum(weights)
    return jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(count, 1.0)


def masked_normalize(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Normalise valid values and return zero for every invalid sample."""
    mask = jnp.asarray(mask, dtype=bool)
    mean = masked_mean(values, mask)
    variance = masked_mean(jnp.square(values - mean), mask)
    normalized = (values - mean) / (jnp.sqrt(variance) + 1e-8)
    return jnp.where(mask, normalized, 0.0)


def normal_log_probability(
    sample: jax.Array,
    location: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    """Return elementwise log density under a diagonal Gaussian latent policy."""
    standardised = (sample - location) * jnp.exp(-log_std)
    return -0.5 * jnp.square(standardised) - log_std - 0.5 * jnp.log(2.0 * jnp.pi)


def latent_to_rate_action(
    trade_latent: jax.Array,
    biological_rate_latent: jax.Array,
) -> jax.Array:
    """Map four policy latents to non-negative ``d^-1`` Rate actions."""
    trade_latent = jnp.asarray(trade_latent, dtype=jnp.float32)
    biological_rate_latent = jnp.asarray(biological_rate_latent, dtype=jnp.float32)
    trade_rate = jax.nn.softplus(trade_latent)[..., None]
    biological_rates = jax.nn.softplus(biological_rate_latent)
    return jnp.concatenate((trade_rate, biological_rates), axis=-1)


@dataclass(frozen=True)
class PPOConfig:
    """Typed configuration for the two-policy PPO training loop."""

    TOTAL_TIMESTEPS: int = 5_000_000
    RUN_TIMESTEPS: int | None = None
    NUM_STEPS: int = 128
    NUM_ENVS: int = 16
    NUM_ACTORS: int = 2
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 4
    DISCOUNT_HALF_LIFE_DAYS: float | None = None
    GAE_LAMBDA: float = 0.95
    VF_COEF: float = 0.5
    CLIP_EPS: float = 0.2
    ACTIVATION: str = "tanh"
    LR: float = 2.5e-4
    PLANT_INITIAL_TRADE: float = 0.05
    FUNGUS_INITIAL_TRADE: float = 0.75
    NORMALIZE_CRITIC_TARGETS: bool = True
    FINITE_HORIZON_RETURNS: bool = False
    TRADE_ONLY: bool = False


@struct.dataclass
class CriticNormalizer:
    """Per-policy running moments for valid raw return targets."""

    count: jax.Array
    mean: jax.Array
    mean_square: jax.Array


class IPPOTrainState(TrainState):
    """Independent PPO optimizer state plus its critic return scale."""

    critic_normalizer: CriticNormalizer


def initial_critic_normalizer() -> CriticNormalizer:
    """Create an empty running return summary with a finite fallback scale."""
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return CriticNormalizer(count=zero, mean=zero, mean_square=zero)


def update_critic_normalizer(
    normalizer: CriticNormalizer,
    targets: jax.Array,
    valid: jax.Array,
) -> CriticNormalizer:
    """Merge valid raw targets into one agent's running moments."""
    valid = jnp.asarray(valid, dtype=bool)
    weights = valid.astype(targets.dtype)
    batch_count = jnp.sum(weights)
    total_count = normalizer.count + batch_count
    batch_sum = jnp.sum(jnp.where(valid, targets, 0.0))
    batch_square_sum = jnp.sum(jnp.where(valid, jnp.square(targets), 0.0))
    mean = (normalizer.count * normalizer.mean + batch_sum) / jnp.maximum(
        total_count, 1.0
    )
    mean_square = (
        normalizer.count * normalizer.mean_square + batch_square_sum
    ) / jnp.maximum(total_count, 1.0)
    return CriticNormalizer(
        count=total_count,
        mean=jnp.where(batch_count > 0.0, mean, normalizer.mean),
        mean_square=jnp.where(
            batch_count > 0.0, mean_square, normalizer.mean_square
        ),
    )


def critic_normalizer_scale(normalizer: CriticNormalizer) -> jax.Array:
    """Return a finite scale without inventing variance for one sample."""
    variance = jnp.maximum(
        normalizer.mean_square - jnp.square(normalizer.mean), 0.0
    )
    return jnp.where(normalizer.count > 1.0, jnp.sqrt(variance + 1e-8), 1.0)


def normalize_critic_values(
    values: jax.Array, normalizer: CriticNormalizer
) -> jax.Array:
    return (values - normalizer.mean) / critic_normalizer_scale(normalizer)


def denormalize_critic_values(
    values: jax.Array, normalizer: CriticNormalizer
) -> jax.Array:
    return values * critic_normalizer_scale(normalizer) + normalizer.mean


class ActorCritic(nn.Module):
    """Shared network architecture used by each independent actor--critic."""

    activation: str = "relu"
    initial_trade: float = 0.1
    trade_only: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[PolicyParameters, jnp.ndarray]:
        """Forward pass of the actor-critic model."""
        activation = getattr(jax.nn, self.activation, jax.nn.relu)

        policy_features = nn.Dense(64, name="policy_encoder_0")(obs)
        policy_features = activation(policy_features)
        policy_features = nn.Dense(64, name="policy_encoder_1")(policy_features)
        policy_features = activation(policy_features)
        trade_loc = nn.Dense(
            1,
            kernel_init=constant(0.0),
            bias_init=constant(jnp.log(jnp.expm1(self.initial_trade))),
            name="trade_head",
        )(policy_features)[..., 0]
        if self.trade_only:
            biological_rate_loc = jnp.zeros(obs.shape[:-1] + (3,))
        else:
            biological_rate_loc = nn.Dense(
                3,
                kernel_init=constant(0.0),
                bias_init=constant(0.0),
                name="biological_rate_head",
            )(policy_features)
        trade_log_std = self.param("trade_log_std", constant(0.0), (1,))
        biological_rate_log_std = (
            jnp.zeros(3) if self.trade_only else self.param(
                "biological_rate_log_std", constant(0.0), (3,)
            )
        )
        policy = PolicyParameters(
            trade_loc=trade_loc,
            trade_log_std=jnp.broadcast_to(trade_log_std[0], trade_loc.shape),
            biological_rate_loc=biological_rate_loc,
            biological_rate_log_std=jnp.broadcast_to(
                biological_rate_log_std, biological_rate_loc.shape
            ),
        )

        critic = nn.Dense(64, name="critic_0")(obs)
        critic = activation(critic)
        critic = nn.Dense(64, name="critic_1")(critic)
        critic = activation(critic)
        critic = nn.Dense(1, name="critic_value")(critic)

        return policy, jnp.squeeze(critic, axis=-1)


class Trajectory(NamedTuple):
    """PPO rollout fields for one policy, distinct from environment transitions."""

    latent_trade_action: jnp.ndarray
    latent_biological_rate_action: jnp.ndarray
    rate_action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    trade_log_probability: jnp.ndarray
    biological_rate_log_probability: jnp.ndarray
    obs: jnp.ndarray
    info: dict
    critic_valid: jnp.ndarray
    biological_rate_actor_valid: jnp.ndarray
    trade_actor_valid: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    bootstrap_valid: jnp.ndarray
    gae_trace_continues: jnp.ndarray
    bootstrap_observation: jnp.ndarray


class PPOUpdateMetrics(NamedTuple):
    """Per-species PPO losses and rollout-validity diagnostics."""

    total_loss: jnp.ndarray
    value_loss: jnp.ndarray
    actor_loss: jnp.ndarray
    learning_rate: jnp.ndarray
    approx_kl: jnp.ndarray
    latent_entropy: jnp.ndarray
    critic_valid_count: jnp.ndarray
    biological_rate_actor_valid_count: jnp.ndarray
    trade_actor_valid_count: jnp.ndarray
    critic_valid_fraction: jnp.ndarray
    biological_rate_actor_valid_fraction: jnp.ndarray
    trade_actor_valid_fraction: jnp.ndarray
    raw_return_mean: jnp.ndarray
    normalized_return_mean: jnp.ndarray
    raw_critic_mean: jnp.ndarray
    normalized_critic_mean: jnp.ndarray
    critic_target_scale: jnp.ndarray


def batchify(
    x: Dict[str, jax.Array],
    agent_list: List[str],
    num_envs: int,
    num_actors: int,
) -> jax.Array:
    """Stack per-agent observations as ``(actors, environments, features)``."""
    # I've adapted this as it was collapsing the envs dimension – I don't know how it
    # worked for their code...
    x_inter = jnp.stack([x[a] for a in agent_list])
    return x_inter.reshape((num_actors, num_envs, -1))

def unbatchify(
    x: jax.Array,
    agent_list: List[str],
    num_envs: int,
    num_actors: int,
) -> Dict[str, jax.Array]:
    """Convert an actor-major batch back to the environment's agent mapping."""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def initialize_runner_state(
    env,
    config: PPOConfig,
    random_streams: RandomStreamContract | None = None,
    rng: jax.Array | None = None,
):
    """Create a fresh, reproducible PPO runner state without an update.

    This keeps network/optimizer initialisation and the first environment reset
    outside the expensive PPO-update executable. Study runners can initialise
    each seed with named streams while sharing one updater across replicates.
    """
    if random_streams is None:
        if rng is None:
            raise ValueError("initialization without named streams requires rng")
        rng, plant_rng, fungus_rng = jax.random.split(rng, 3)
        action_rng = environment_rng = minibatch_rng = rng
    else:
        plant_rng = random_streams.key("plant_initialization")
        fungus_rng = random_streams.key("fungal_initialization")
        action_rng = random_streams.key("policy_action_sampling")
        environment_rng = random_streams.key("environment_variation")
        minibatch_rng = random_streams.key("minibatch_ordering")

    rollout_timesteps = config.NUM_STEPS * config.NUM_ENVS
    total_optimizer_steps = (
        config.TOTAL_TIMESTEPS // rollout_timesteps
        * config.NUM_MINIBATCHES * config.UPDATE_EPOCHS
    )

    def linear_schedule(count):
        return config.LR * jnp.maximum(0.0, 1.0 - count / total_optimizer_steps)

    init_x = jnp.zeros((1, env.observation_spaces[PLANT].shape[0]))
    policies = {
        PLANT: ActorCritic(
            activation=config.ACTIVATION,
            initial_trade=getattr(config, "PLANT_INITIAL_TRADE", 0.05),
            trade_only=config.TRADE_ONLY,
        ),
        FUNGUS: ActorCritic(
            activation=config.ACTIVATION,
            initial_trade=getattr(config, "FUNGUS_INITIAL_TRADE", 0.75),
            trade_only=config.TRADE_ONLY,
        ),
    }
    train_state = {
        PLANT: IPPOTrainState.create(
            apply_fn=policies[PLANT].apply,
            params=policies[PLANT].init(plant_rng, init_x),
            tx=optax.adam(learning_rate=linear_schedule),
            critic_normalizer=initial_critic_normalizer(),
        ),
        FUNGUS: IPPOTrainState.create(
            apply_fn=policies[FUNGUS].apply,
            params=policies[FUNGUS].init(fungus_rng, init_x),
            tx=optax.adam(learning_rate=linear_schedule),
            critic_normalizer=initial_critic_normalizer(),
        ),
    }
    environment_rng, reset_key = jax.random.split(environment_rng)
    reset_rng = jax.random.split(reset_key, config.NUM_ENVS)
    obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    return train_state, env_state, obs, (action_rng, environment_rng, minibatch_rng)

def make_train(
    env,
    config,
    random_streams: RandomStreamContract | None = None,
    initial_runner_state=None,
):
    """Build a JIT-compatible independent-PPO trainer for the fixed agent API.

    Each policy consumes its own bounded observation and learns only from the
    validity, bootstrap, and trace controls derived from typed Transitions.
    The returned training output contains both policy states, trajectories,
    losses, advantages, targets, and returns for smoke validation.
    """

    gamma = discount_from_half_life(
        getattr(env, "decision_interval_days", env.config.dt),
        config.DISCOUNT_HALF_LIFE_DAYS,
    )
    if gamma == 1.0 and not config.FINITE_HORIZON_RETURNS:
        configured_consumers = []
        if env.config.consumer_mode in ("mixed", "plant-only"):
            configured_consumers.append((PLANT, env.species.plant))
        if env.config.consumer_mode in ("mixed", "fungus-only"):
            configured_consumers.append((FUNGUS, env.species.fungus))
        for agent, traits in configured_consumers:
            if (
                traits.initial_biomass <= 0.0
                or traits.kappa_p <= 0.0
                or traits.death_fraction <= 0.0
            ):
                raise ValueError(
                    "undiscounted PPO requires a guaranteed finite lifetime "
                    f"for each configured consumer; {agent} does not satisfy it"
                )

    if config.NUM_ACTORS != len(env.agents):
        raise ValueError("NUM_ACTORS must match the environment agent count")
    if config.NUM_STEPS % config.NUM_MINIBATCHES != 0:
        raise ValueError("NUM_STEPS must be divisible by NUM_MINIBATCHES")
    rollout_timesteps = config.NUM_STEPS * config.NUM_ENVS
    if config.TOTAL_TIMESTEPS < rollout_timesteps:
        raise ValueError("TOTAL_TIMESTEPS must contain at least one PPO update")
    configured_run_timesteps = getattr(config, "RUN_TIMESTEPS", None)
    run_timesteps = (
        config.TOTAL_TIMESTEPS
        if configured_run_timesteps is None
        else configured_run_timesteps
    )
    if run_timesteps < rollout_timesteps or run_timesteps % rollout_timesteps:
        raise ValueError("RUN_TIMESTEPS must contain whole PPO updates")
    if any(env.observation_spaces[agent].shape != (5,) for agent in env.agents):
        raise ValueError("each independent actor-critic requires five observations")

    NUM_UPDATES = (
        run_timesteps // config.NUM_STEPS // config.NUM_ENVS
    )
    MINIBATCH_SIZE = rollout_timesteps // config.NUM_MINIBATCHES

    total_optimizer_steps = (
        config.TOTAL_TIMESTEPS // rollout_timesteps
        * config.NUM_MINIBATCHES * config.UPDATE_EPOCHS
    )

    def linear_schedule(count):
        return config.LR * jnp.maximum(0.0, 1.0 - count / total_optimizer_steps)

    def critic_output_to_raw(
        values: jax.Array, normalizer: CriticNormalizer
    ) -> jax.Array:
        """Keep the raw-target ablation on the pre-normalization critic scale."""
        if config.NORMALIZE_CRITIC_TARGETS:
            return denormalize_critic_values(values, normalizer)
        return values

    # Keep resumed state as a dynamic JAX argument so callers can reuse one
    # compiled executable across updates instead of recompiling per checkpoint.
    resume_mode = initial_runner_state is not None

    def train(rng, resumed_runner_state=initial_runner_state):
        """
        Main training function for PPO. 
        --- Steps ---
        1. Initialize the actor-critic network and training state.
        2. Initialize the environment.
        3. Scan over the number of updates:
            a. For each update, perform a full step to update the environment and network.
            b. Calculate advantages using Generalized Advantage Estimation (GAE).
        4. Return the final training state and metrics.
        
        Plant and fungal policies consume their own vectorised observations.
        """
        # Initialize independent plant and fungus networks.  The optional
        # runner state is used by the study checkpoint seam to continue the
        # same optimizer, environment, and named-RNG state.
        plant_policy = ActorCritic(
            activation=config.ACTIVATION,
            initial_trade=getattr(config, "PLANT_INITIAL_TRADE", 0.05),
            trade_only=config.TRADE_ONLY,
        )
        fungus_policy = ActorCritic(
            activation=config.ACTIVATION,
            initial_trade=getattr(config, "FUNGUS_INITIAL_TRADE", 0.75),
            trade_only=config.TRADE_ONLY,
        )

        if not resume_mode:
            train_state, env_state, obs, runner_rngs = initialize_runner_state(
                env, config, random_streams, rng
            )
            action_rng, environment_rng, minibatch_rng = runner_rngs
        else:
            if resumed_runner_state is None:
                raise ValueError("resumed trainer requires runner state")
            train_state, env_state, obs, runner_rngs = resumed_runner_state
            action_rng, environment_rng, minibatch_rng = runner_rngs

        def _update_step(runner_state, x):
            """
            Full update step for environment and network.
            Collects trajectories using lax.scan, calculates GAE advantages.
            """
            def _env_step(runner_state, x):
                """
                Execute a single step in the environment.
                
                1. Sample actions from the actor network.
                2. Step the environment with the sampled actions.
                3. Collect Transition for the trajectory.
                """
                train_state, env_state, last_obs, rngs = runner_state
                action_rng, environment_rng, minibatch_rng = rngs
                action_rng, plant_act_rng, fungus_act_rng = jax.random.split(
                    action_rng, 3
                )
                plant_trade_rng, plant_allocation_rng = jax.random.split(plant_act_rng)
                fungus_trade_rng, fungus_allocation_rng = jax.random.split(fungus_act_rng)

                # Batch observations for the independent plant and fungal policies.
                obs_batch = batchify(
                    last_obs, env.agents, config.NUM_ENVS, config.NUM_ACTORS
                )
                plant_obs_batch, fungus_obs_batch = obs_batch[0], obs_batch[1]

                # Keep each factor's sampling and likelihood explicit in the rollout.
                plant_policy_parameters, plant_value = plant_policy.apply(
                    train_state[PLANT].params, plant_obs_batch
                )
                plant_latent_trade = plant_policy_parameters.trade_loc + jnp.exp(
                    plant_policy_parameters.trade_log_std
                ) * jax.random.normal(
                    plant_trade_rng, plant_policy_parameters.trade_loc.shape
                )
                plant_latent_biological_rate = jax.lax.cond(
                    config.TRADE_ONLY,
                    lambda _: jnp.zeros_like(plant_policy_parameters.biological_rate_loc),
                    lambda _: plant_policy_parameters.biological_rate_loc
                    + jnp.exp(plant_policy_parameters.biological_rate_log_std)
                    * jax.random.normal(plant_allocation_rng, plant_policy_parameters.biological_rate_loc.shape),
                    operand=None,
                )
                plant_trade_log_probability = normal_log_probability(
                    plant_latent_trade,
                    plant_policy_parameters.trade_loc,
                    plant_policy_parameters.trade_log_std,
                )
                plant_biological_rate_log_probability = jax.lax.cond(
                    config.TRADE_ONLY,
                    lambda _: jnp.zeros_like(plant_trade_log_probability),
                    lambda _: jnp.sum(normal_log_probability(
                        plant_latent_biological_rate, plant_policy_parameters.biological_rate_loc,
                        plant_policy_parameters.biological_rate_log_std), axis=-1),
                    operand=None,
                )
                plant_rate_action = (
                    fixed_allocation_rate_action(jax.nn.softplus(plant_latent_trade))
                    if config.TRADE_ONLY else latent_to_rate_action(
                        plant_latent_trade, plant_latent_biological_rate
                    )
                )

                fungus_policy_parameters, fungus_value = fungus_policy.apply(
                    train_state[FUNGUS].params, fungus_obs_batch
                )
                fungus_latent_trade = fungus_policy_parameters.trade_loc + jnp.exp(
                    fungus_policy_parameters.trade_log_std
                ) * jax.random.normal(
                    fungus_trade_rng, fungus_policy_parameters.trade_loc.shape
                )
                fungus_latent_biological_rate = jax.lax.cond(
                    config.TRADE_ONLY,
                    lambda _: jnp.zeros_like(fungus_policy_parameters.biological_rate_loc),
                    lambda _: fungus_policy_parameters.biological_rate_loc
                    + jnp.exp(fungus_policy_parameters.biological_rate_log_std)
                    * jax.random.normal(fungus_allocation_rng, fungus_policy_parameters.biological_rate_loc.shape),
                    operand=None,
                )
                fungus_trade_log_probability = normal_log_probability(
                    fungus_latent_trade,
                    fungus_policy_parameters.trade_loc,
                    fungus_policy_parameters.trade_log_std,
                )
                fungus_biological_rate_log_probability = jax.lax.cond(
                    config.TRADE_ONLY,
                    lambda _: jnp.zeros_like(fungus_trade_log_probability),
                    lambda _: jnp.sum(normal_log_probability(
                        fungus_latent_biological_rate, fungus_policy_parameters.biological_rate_loc,
                        fungus_policy_parameters.biological_rate_log_std), axis=-1),
                    operand=None,
                )
                fungus_rate_action = (
                    fixed_allocation_rate_action(jax.nn.softplus(fungus_latent_trade))
                    if config.TRADE_ONLY else latent_to_rate_action(
                        fungus_latent_trade, fungus_latent_biological_rate
                    )
                )

                # Unbatchify the actions to match the environment's expected input format
                env_act = unbatchify(
                    jnp.stack([plant_rate_action, fungus_rate_action]),
                    env.agents, config.NUM_ENVS, config.NUM_ACTORS
                )

                environment_rng, _rng = jax.random.split(environment_rng)
                rng_step = jax.random.split(_rng, config.NUM_ENVS)
                obs, env_state, reward, _, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )
                plant_fields = transition_to_ppo_fields(
                    info["transitions"][PLANT],
                    finite_horizon_returns=config.FINITE_HORIZON_RETURNS,
                )
                fungus_fields = transition_to_ppo_fields(
                    info["transitions"][FUNGUS],
                    finite_horizon_returns=config.FINITE_HORIZON_RETURNS,
                )

                # Collect Trajectory object
                plant_trajectory = Trajectory(
                    latent_trade_action=plant_latent_trade,
                    latent_biological_rate_action=plant_latent_biological_rate,
                    rate_action=plant_rate_action,
                    value=critic_output_to_raw(
                        jnp.array(plant_value),
                        train_state[PLANT].critic_normalizer,
                    ),
                    reward=reward[PLANT].reshape((config.NUM_ENVS,)),
                    trade_log_probability=plant_trade_log_probability,
                    biological_rate_log_probability=plant_biological_rate_log_probability,
                    obs=plant_obs_batch,
                    info=info[PLANT],
                    critic_valid=plant_fields.critic_valid,
                    biological_rate_actor_valid=(
                        jnp.zeros_like(plant_fields.biological_rate_actor_valid)
                        if config.TRADE_ONLY else plant_fields.biological_rate_actor_valid
                    ),
                    trade_actor_valid=plant_fields.trade_actor_valid,
                    terminated=plant_fields.terminated,
                    truncated=plant_fields.truncated,
                    bootstrap_valid=plant_fields.bootstrap_valid,
                    gae_trace_continues=plant_fields.gae_trace_continues,
                    bootstrap_observation=plant_fields.bootstrap_observation,
                )
                fungus_trajectory = Trajectory(
                    latent_trade_action=fungus_latent_trade,
                    latent_biological_rate_action=fungus_latent_biological_rate,
                    rate_action=fungus_rate_action,
                    value=critic_output_to_raw(
                        jnp.array(fungus_value),
                        train_state[FUNGUS].critic_normalizer,
                    ),
                    reward=reward[FUNGUS].reshape((config.NUM_ENVS,)),
                    trade_log_probability=fungus_trade_log_probability,
                    biological_rate_log_probability=fungus_biological_rate_log_probability,
                    obs=fungus_obs_batch,
                    info=info[FUNGUS],
                    critic_valid=fungus_fields.critic_valid,
                    biological_rate_actor_valid=(
                        jnp.zeros_like(fungus_fields.biological_rate_actor_valid)
                        if config.TRADE_ONLY else fungus_fields.biological_rate_actor_valid
                    ),
                    trade_actor_valid=fungus_fields.trade_actor_valid,
                    terminated=fungus_fields.terminated,
                    truncated=fungus_fields.truncated,
                    bootstrap_valid=fungus_fields.bootstrap_valid,
                    gae_trace_continues=fungus_fields.gae_trace_continues,
                    bootstrap_observation=fungus_fields.bootstrap_observation,
                )

                runner_state = (
                    train_state,
                    env_state,
                    obs,
                    (action_rng, environment_rng, minibatch_rng),
                )

                return runner_state, (plant_trajectory, fungus_trajectory)

            # Scan over the number of steps to collect trajectories for parallel envs, per update.
            runner_state, (plant_traj, fungus_traj) = jax.lax.scan(
                _env_step, runner_state, None, config.NUM_STEPS
            )

            train_state, env_state, last_obs, rngs = runner_state
            _, plant_bootstrap_values = plant_policy.apply(
                train_state[PLANT].params, plant_traj.bootstrap_observation
            )
            _, fungus_bootstrap_values = fungus_policy.apply(
                train_state[FUNGUS].params, fungus_traj.bootstrap_observation
            )
            plant_bootstrap_values = critic_output_to_raw(
                plant_bootstrap_values, train_state[PLANT].critic_normalizer
            )
            fungus_bootstrap_values = critic_output_to_raw(
                fungus_bootstrap_values, train_state[FUNGUS].critic_normalizer
            )
            plant_advantages, plant_targets = calculate_gae(
                rewards=plant_traj.reward,
                values=plant_traj.value,
                bootstrap_values=plant_bootstrap_values,
                critic_valid=plant_traj.critic_valid,
                bootstrap_valid=plant_traj.bootstrap_valid,
                gae_trace_continues=plant_traj.gae_trace_continues,
                gamma=gamma,
                gae_lambda=config.GAE_LAMBDA,
            )
            fungus_advantages, fungus_targets = calculate_gae(
                rewards=fungus_traj.reward,
                values=fungus_traj.value,
                bootstrap_values=fungus_bootstrap_values,
                critic_valid=fungus_traj.critic_valid,
                bootstrap_valid=fungus_traj.bootstrap_valid,
                gae_trace_continues=fungus_traj.gae_trace_continues,
                gamma=gamma,
                gae_lambda=config.GAE_LAMBDA,
            )

            train_state = {
                PLANT: train_state[PLANT].replace(
                    critic_normalizer=update_critic_normalizer(
                        train_state[PLANT].critic_normalizer,
                        plant_targets,
                        plant_traj.critic_valid,
                    )
                ),
                FUNGUS: train_state[FUNGUS].replace(
                    critic_normalizer=update_critic_normalizer(
                        train_state[FUNGUS].critic_normalizer,
                        fungus_targets,
                        fungus_traj.critic_valid,
                    )
                ),
            }

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(agent_train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        """
                        Calculate the loss for the PPO update. Same implementation as in original
                        Schulman et al. (2017) PPO paper (section 5, eq.(9)).
                        
                        Loss = -L_actor + L_value
                        """
                        # RERUN NETWORK
                        policy_parameters, value = agent_train_state.apply_fn(
                            params, traj_batch.obs
                        )
                        value = critic_output_to_raw(
                            value, agent_train_state.critic_normalizer
                        )
                        trade_log_probability = normal_log_probability(
                            traj_batch.latent_trade_action,
                            policy_parameters.trade_loc,
                            policy_parameters.trade_log_std,
                        )
                        biological_rate_log_probability = (
                            jnp.zeros_like(traj_batch.trade_log_probability)
                            if config.TRADE_ONLY else jnp.sum(
                                normal_log_probability(
                                    traj_batch.latent_biological_rate_action,
                                    policy_parameters.biological_rate_loc,
                                    policy_parameters.biological_rate_log_std,
                                ), axis=-1,
                            )
                        )

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.CLIP_EPS, config.CLIP_EPS)
                        if config.NORMALIZE_CRITIC_TARGETS:
                            value = normalize_critic_values(
                                value, agent_train_state.critic_normalizer
                            )
                            value_pred_clipped = normalize_critic_values(
                                value_pred_clipped,
                                agent_train_state.critic_normalizer,
                            )
                            targets = normalize_critic_values(
                                targets, agent_train_state.critic_normalizer
                            )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * masked_mean(
                            jnp.maximum(value_losses, value_losses_clipped),
                            traj_batch.critic_valid,
                        )

                        # CALCULATE ACTOR LOSS
                        actor_valid = (
                            traj_batch.trade_actor_valid if config.TRADE_ONLY
                            else traj_batch.biological_rate_actor_valid
                        )
                        log_probability = (
                            trade_log_probability if config.TRADE_ONLY else
                            biological_rate_log_probability + jnp.where(
                                traj_batch.trade_actor_valid, trade_log_probability, 0.0
                            )
                        )
                        old_log_probability = (
                            traj_batch.trade_log_probability if config.TRADE_ONLY else
                            traj_batch.biological_rate_log_probability + jnp.where(
                                traj_batch.trade_actor_valid,
                                traj_batch.trade_log_probability, 0.0
                            )
                        )
                        ratio = jnp.exp(log_probability - old_log_probability)
                        gae = masked_normalize(gae, actor_valid)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.CLIP_EPS,
                                1.0 + config.CLIP_EPS,
                            )
                            * gae
                        )
                        loss_actor = masked_mean(
                            -jnp.minimum(loss_actor1, loss_actor2),
                            actor_valid,
                        )
                        approx_kl = masked_mean(
                            old_log_probability - log_probability,
                            actor_valid,
                        )
                        normal_entropy = lambda log_std: jnp.sum(
                            log_std + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)), axis=-1
                        )
                        latent_entropy = masked_mean(
                            normal_entropy(policy_parameters.trade_log_std)
                            if config.TRADE_ONLY else
                            normal_entropy(policy_parameters.biological_rate_log_std)
                            + jnp.where(
                                traj_batch.trade_actor_valid,
                                normal_entropy(policy_parameters.trade_log_std), 0.0
                            ),
                            actor_valid,
                        )
                        total_loss = loss_actor + config.VF_COEF * value_loss
                        return total_loss, (value_loss, loss_actor, approx_kl, latent_entropy)

                    def _apply_update(state):
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        (total_loss, losses), grads = grad_fn(
                            state.params, traj_batch, advantages, targets
                        )
                        return state.apply_gradients(grads=grads), (
                            total_loss,
                            *losses,
                        )

                    def _skip_update(state):
                        zero = jnp.asarray(0.0)
                        return state, (zero, zero, zero, zero, zero)

                    return jax.lax.cond(
                        jnp.any(
                            traj_batch.trade_actor_valid if config.TRADE_ONLY
                            else traj_batch.biological_rate_actor_valid
                        ),
                        _apply_update,
                        _skip_update,
                        agent_train_state,
                    )

                agent_train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Flatten time and vector-environment axes into one independent
                # sample axis before shuffling. Every trajectory field (masks,
                # actions, observations, returns) must follow this convention.
                batch_size = MINIBATCH_SIZE * config.NUM_MINIBATCHES
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)

                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )

                # Shuffle the batch.
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                agent_train_state, total_loss = jax.lax.scan(
                    _update_minibatch, agent_train_state, minibatches
                )
                update_state = (agent_train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Update the plant policy network.
            update_plant_state = (
                train_state[PLANT],
                plant_traj,
                plant_advantages,
                plant_targets,
                rngs[2],
            )
            update_plant_state, plant_loss_info = jax.lax.scan(
                _update_epoch, update_plant_state, None, config.UPDATE_EPOCHS
            )

            # Update the fungus policy network.
            update_fungus_state = (
                train_state['fungus'], fungus_traj,
                fungus_advantages, fungus_targets,
                update_plant_state[-1]
            )
            update_fungus_state, fungus_loss_info = jax.lax.scan(
                _update_epoch, update_fungus_state, None, config.UPDATE_EPOCHS
            )

            train_state = {
                PLANT: update_plant_state[0],
                FUNGUS: update_fungus_state[0],
            }
            rngs = (rngs[0], rngs[1], update_fungus_state[-1])

            def _metrics(trajectory, targets, loss_info, state):
                total_loss, value_loss, actor_loss, approx_kl, latent_entropy = loss_info
                sample_count = trajectory.critic_valid.size
                critic_count = jnp.sum(trajectory.critic_valid)
                allocation_count = jnp.sum(
                    trajectory.biological_rate_actor_valid
                )
                trade_count = jnp.sum(trajectory.trade_actor_valid)
                normalizer = state.critic_normalizer
                return PPOUpdateMetrics(
                    total_loss=total_loss,
                    value_loss=value_loss,
                    actor_loss=actor_loss,
                    learning_rate=linear_schedule(state.step),
                    approx_kl=approx_kl,
                    latent_entropy=latent_entropy,
                    critic_valid_count=critic_count,
                    biological_rate_actor_valid_count=allocation_count,
                    trade_actor_valid_count=trade_count,
                    critic_valid_fraction=critic_count / sample_count,
                    biological_rate_actor_valid_fraction=(
                        allocation_count / sample_count
                    ),
                    trade_actor_valid_fraction=trade_count / sample_count,
                    raw_return_mean=masked_mean(targets, trajectory.critic_valid),
                    normalized_return_mean=masked_mean(
                        normalize_critic_values(targets, normalizer),
                        trajectory.critic_valid,
                    ),
                    raw_critic_mean=masked_mean(
                        trajectory.value, trajectory.critic_valid
                    ),
                    normalized_critic_mean=masked_mean(
                        normalize_critic_values(trajectory.value, normalizer),
                        trajectory.critic_valid,
                    ),
                    critic_target_scale=critic_normalizer_scale(normalizer),
                )

            plant_metrics = _metrics(
                plant_traj, plant_targets, plant_loss_info, train_state[PLANT]
            )
            fungus_metrics = _metrics(
                fungus_traj, fungus_targets, fungus_loss_info, train_state[FUNGUS]
            )

            runner_state = (train_state, env_state, last_obs, rngs)
            return runner_state, (
                (plant_traj, fungus_traj),
                (plant_metrics, fungus_metrics),
                (plant_advantages, fungus_advantages),
                (plant_targets, fungus_targets),
            )

        # Scan over update steps. Each stochastic subsystem retains its own key.
        if not resume_mode:
            runner_rngs = (action_rng, environment_rng, minibatch_rng)
        runner_state = (train_state, env_state, obs, runner_rngs)
        runner_state, (
            (plant_traj, fungus_traj),
            (plant_metrics, fungus_metrics),
            (plant_advantages, fungus_advantages),
            (plant_targets, fungus_targets),
        ) = jax.lax.scan(
            _update_step, runner_state, None, NUM_UPDATES
        )

        return {
            "runner_state": runner_state,
            "trajectories": (plant_traj, fungus_traj),
            "metrics": {PLANT: plant_metrics, FUNGUS: fungus_metrics},
            "advantages": {
                PLANT: plant_advantages,
                FUNGUS: fungus_advantages,
            },
            "targets": {PLANT: plant_targets, FUNGUS: fungus_targets},
            # GAE lambda-returns are the critic regression targets.
            "returns": {PLANT: plant_targets, FUNGUS: fungus_targets},
        }

    return train
