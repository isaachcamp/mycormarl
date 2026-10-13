# PPO used by MycoMARL: algorithm and primary references

This note documents the implementation currently used by the study runners. It
is independent PPO (IPPO): plant and fungus each have their own actor--critic
and update from their own local observation and reward stream. It is not MAPPO,
because the critics are not centralised and no parameter sharing is used.

## Core PPO objective

At each iteration, the current policy collects an on-policy rollout. For a
stored transition $t$, let $\rho_t$ be its old-policy probability ratio:

$$
  \rho_t(\theta) =
  \frac{\pi_\theta(a_t\mid o_t)}{\pi_{\theta_{\rm old}}(a_t\mid o_t)}.
$$

The clipped surrogate maximised by PPO is

$$
 L^{\rm CLIP}(\theta)=\mathbb{E}_t\left[
 \min\left(\rho_t(\theta)\hat A_t,
 \operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)
 \right].
$$

The clip prevents a single batch from benefiting from an excessively large
policy change. The implementation minimises the negative actor objective and
adds a value loss:

$$
 L = L_{\rm actor} + c_V L_{\rm value},
\qquad
 L_{\rm value}=\frac12\mathbb{E}_t\left[
 \max\{(V_\theta-y_t)^2,
 (\operatorname{clip}(V_\theta,V_{\rm old}-\epsilon,V_{\rm old}+\epsilon)-y_t)^2\}
 \right].
$$

Here $\epsilon=0.2$, $c_V=0.5$, and the value-function clipping is an
implementation choice consistent with common PPO implementations. The code is
in [`mycormarl/algos/ppo.py`](../../mycormarl/mycormarl/algos/ppo.py), in
`_loss_fn`.

## Why parallel environments and JaxMARL are used

The trainer can step several independent copies of the environment at the
same time. `NUM_ENVS` is therefore a throughput and batching parameter, not a
change to the IPPO objective: each copy has its own state, reset key, and
trajectory, while the plant and fungus policies are shared across copies. A
parallel rollout collects `NUM_STEPS × NUM_ENVS` samples per update, keeping
the accelerator occupied and reducing the wall-clock cost of the many
environment transitions required by on-policy learning. It also gives each
update samples from multiple independently reset trajectories, which reduces
reliance on the particular sequence of states in one episode. The copies do
not remove the multi-agent non-stationarity described below, and increasing
`NUM_ENVS` does not create additional biological agents within an environment.

JaxMARL is used as the environment framework because it provides the typed,
JAX-compatible multi-agent boundary around which this trainer is organised.
`BaseMycorMarl` exposes the standard reset/step interaction with agent-keyed
observations, actions, rewards, and termination information, while JAX
transformations such as `jit` and `vmap` can compile and batch the same
environment logic. This is important here because the environment includes
soil and biological state updates that must be evaluated repeatedly during
training. The framework keeps those dynamics in a reusable MARL environment
rather than embedding them in PPO, so the same environment transition
contract can support IPPO and other algorithms. The repository-specific
`Transition` and PPO transition adapter preserve that separation: JaxMARL
reports what happened in the biological model, and the PPO layer derives the
learning masks and return bookkeeping.

Parallelism is a capability rather than a requirement of every study
protocol. The general PPO defaults use multiple environments, whereas the
reduced trade-only study deliberately sets `NUM_ENVS=1` to match its stated
finite-horizon protocol; that setting changes collection efficiency and batch
composition, not the definition of PPO.

## What the advantage means

The advantage asks whether an action turned out better or worse than the
policy's usual expectation at the same observation. Formally, it is the
difference between the action value and the state value:

$$
A^\pi(o_t,a_t)=Q^\pi(o_t,a_t)-V^\pi(o_t).
$$

A positive advantage means that the sampled action produced more return than
the critic expected, so PPO increases its probability (subject to clipping).
A negative advantage means that it underperformed the expectation, so PPO
reduces its probability. Using this relative quantity also removes much of
the scale shared by all actions in a state and gives a lower-variance policy
gradient than using the raw return alone.

The implementation does not estimate $Q^\pi$ with a separate action-value
network. Instead, it uses the critic's state-value estimates and generalized
advantage estimation (GAE). First it forms the one-step temporal-difference
residual:

$$
\delta_t=r_t+\gamma V(o_{t+1})-V(o_t).
$$

It then combines residuals backward through the rollout:

$$
\hat A_t=\delta_t+\gamma\lambda m_t\hat A_{t+1},
$$

or equivalently, a discounted sum of future residuals weighted by
$(\gamma\lambda)^l$. The parameter $\lambda$ controls the bias--variance
trade-off: values near zero rely mostly on the one-step residual, while values
near one use a longer return estimate. Here $\lambda=0.95$.

The mask $m_t$ stops the trace at an episode boundary or another point where
bootstrapping should not continue. Separate validity masks exclude death and
post-death samples from the actor and critic means. After GAE is computed,
advantages are standardised over valid actor samples separately for plant and
fungus; the standardised values are the weights used by the clipped actor
objective. The calculation is implemented by
[`calculate_gae`](../../mycormarl/mycormarl/algos/ppo.py).

The implementation keeps two boundary decisions separate. `bootstrap_valid`
controls whether the one-step residual includes the next-state value,
$\delta_t=r_t+\gamma V(o_{t+1})-V(o_t)$. `gae_trace_continues` controls whether
later residuals are propagated through the recursive GAE calculation,
$\hat A_t=\delta_t+\gamma\lambda m_t\hat A_{t+1}$. At biological death both
are stopped because there is no valid future value. At an administrative
truncation in continuing-return mode, the next-state value may still be used
for bootstrapping, but the GAE trace stops so it does not cross into a reset or
new rollout segment. In finite-horizon mode, truncation is treated as a true
return boundary: both bootstrapping and trace continuation stop.

## What the actor and critic networks do

PPO uses two cooperating function approximators for each species. They are
separate plant and fungus networks, so each organism has its own policy and
value estimate.

The **actor** is the policy network. It takes the organism's current local
observation $o_t$ and outputs the parameters of a factorised diagonal Gaussian
over latent actions:

$$
z_t \sim \pi_\theta(\cdot\mid o_t).
$$

The latent sample is transformed with `softplus` before being passed to the
environment as a non-negative physical rate. During data collection, the
actor therefore chooses the trade (and, in the full model, biological
allocation) action. During learning, PPO compares the new actor probability
of the recorded latent action with its old probability and adjusts the actor
so that actions with positive advantages become more likely and actions with
negative advantages become less likely. The clipped objective limits the size
of that change.

The **critic** is the value network. It takes the same local observation but
outputs a scalar estimate of the expected future discounted reward:

$$
V_\phi(o_t)\approx\mathbb{E}_\pi\left[\sum_{l\ge 0}\gamma^l r_{t+l}\mid o_t\right].
$$

It does not select actions. Instead, its estimate supplies the baseline used
to calculate advantages and the bootstrap term in GAE. The critic is trained
to predict the GAE return target
$y_t=\hat A_t+V_\phi(o_t)$ using the clipped value loss. A good critic makes
the advantage estimates less noisy; a poor or rapidly changing critic can
make actor updates noisy or misleading.

In this repository both networks are two-layer 64-unit MLPs (with `tanh` in
the study configurations), and the actor and critic parameters are updated
from the same on-policy rollout but with their respective PPO losses. The
actor's likelihood, KL, and entropy diagnostics are computed in latent
Gaussian space, while only the transformed physical action is sent to the
environment.

## Rollouts, GAE, and targets

The trainer collects `NUM_STEPS × NUM_ENVS` transitions, then performs several
epochs of shuffled minibatch updates on that batch. It estimates advantages
with generalized advantage estimation (GAE):

$$
 \delta_t=r_t+\gamma V(o_{t+1})-V(o_t),
 \qquad
 \hat A_t=\delta_t+\gamma\lambda m_t\hat A_{t+1},
$$

where $m_t$ is the explicit trace-continuation mask. The critic target is
$y_t=\hat A_t+V(o_t)$. The implementation uses $\lambda=0.95$, and its
discount is derived from a configured physical reward half-life. The trade-only
study uses finite-horizon returns, so the administrative episode boundary is
treated as terminal for return estimation. Death and invalid post-death samples
are masked separately; they do not contribute to the critic or actor means.

Advantages are normalised over valid actor samples separately for each species.
Critic targets are optionally normalised with a running per-species mean and
variance, while the reported raw return diagnostics are converted back to the
original reward scale.

## Policy and action parameterisation

Each species has an independent actor--critic. The actor is a two-layer MLP
with 64 units per layer and `tanh` activation in study configurations; the
critic is a separate two-layer 64-unit MLP. The continuous latent policy is a
factorised diagonal Gaussian. Its trade latent has one dimension and its
biological-allocation head has three dimensions in the full model. In the
trade-only study only the trade latent is sampled and scored.

The sampled latent is transformed with `softplus`, producing non-negative
physical rates. PPO likelihoods, ratios, KL, and entropy are computed in the
latent Gaussian space, matching the distribution that is actually sampled;
the transformed physical action is passed to the environment. This is an
important distinction from treating the bounded/nonlinear physical fraction as
an untransformed Gaussian variable.

The plant and fungus initial trade locations are configured to correspond to
approximately 0.05 and 0.75, respectively. Their log standard deviations are
learned state-independent parameters. There is currently no entropy bonus in
the objective; latent entropy is logged as a diagnostic. Approximate KL is also
logged to monitor policy movement.

## Multi-agent interpretation

IPPO applies the PPO update independently to each agent while both policies
interact through the same environment. Thus each policy sees a non-stationary
learning environment because the other policy is changing. This is deliberate
for the study protocol and means results are learned IPPO outcomes, not a claim
of a globally optimal policy or game-theoretic equilibrium.

MAPPO is a related but different algorithm: it retains separate policy actors
but uses a centralised critic during training (centralised training with
decentralised execution), usually with global state and agent information. The
MAPPO paper reports that PPO variants can be strong cooperative-MARL baselines
and analyses practical implementation choices, but its centralised critic is
not part of this repository's current trainer.

## Configuration used in the study

The general defaults are declared in `PPOConfig`:

* learning rate $2.5\times10^{-4}$, linearly decayed;
* four optimisation epochs and four minibatches by default;
* clip coefficient 0.2, GAE $\lambda=0.95$, value coefficient 0.5;
* vectorised rollout over `NUM_ENVS` environments;
* explicit termination, truncation, bootstrap, and actor-validity masks.

The reduced trade-only study overrides this to one environment, 120 steps per
rollout, one minibatch, four epochs, finite-horizon returns, and 500 episodes
of 120 policy decisions per seed. The authoritative values are in
[`trade_only_study.py`](../../mycormarl/mycormarl/trade_only_study.py).

## Primary references

1. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017),
   “Proximal Policy Optimization Algorithms,” arXiv:1707.06347.
   [Paper and abstract](https://arxiv.org/abs/1707.06347). This is the original
   PPO paper introducing the clipped surrogate and multi-epoch minibatch
   optimisation.
2. C. Yu et al. (2022), “The Surprising Effectiveness of PPO in Cooperative
   Multi-Agent Games,” NeurIPS 2022 / arXiv:2103.01955.
   [Paper](https://arxiv.org/abs/2103.01955) and
   [official implementation](https://github.com/zoeyuchao/mappo). This is the
   primary MAPPO/IPPO multi-agent reference; it studies centralised-critic
   MAPPO and local-critic IPPO-style PPO variants.
3. CleanRL, `ppo_continuous_action.py`, an executable first-party research
   implementation and implementation-detail reference for continuous-action
   PPO. [Source](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
   and [algorithm notes](https://docs.cleanrl.dev/rl-algorithms/ppo/).
   It is a comparison/reference implementation, not a dependency of this
   repository.
4. C. Bettini et al. (2024), “JaxMARL: Multi-Agent Reinforcement Learning
   Environments and Algorithms in JAX,” NeurIPS 2024 Datasets and Benchmarks.
   [Paper](https://arxiv.org/abs/2311.10090) and
   [documentation](https://jaxmarl.foersterlab.com/). This is the framework
   reference for the JAX-compatible multi-agent environment boundary and
   parallel environment execution used by this repository.

The equations and algorithmic claims above are based on the original PPO paper;
the JaxMARL framework claims are based on its paper and documentation, and the
repository-specific details are traced to the local implementation and
configuration files linked above.
