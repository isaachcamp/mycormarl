# Paired random seeds for `mixed` versus `plant-only` policy comparisons

**Research date:** 13 August 2026  
**Scope:** Whether MycorMARL should match random-seed IDs when independently
training `mixed` and `plant-only` IPPO policies at each initial-P level. The
phase-one pilot uses five training runs per mode and deterministic
latent-location evaluation. Plant reproductive fitness is compared by an
absolute difference of treatment means; biomass response is reported as a
ratio of treatment means, not as a mean of seedwise ratios.

## Executive answer

Matched seed IDs are potentially useful as a **common-random-numbers (CRN)**
variance-reduction device, but they are not intrinsically more representative
or more comparable to greenhouse data. They do not change either scientific
estimand:

\[
\widehat{\Delta}_{\mathrm{fitness}}
=\bar Y_{\mathrm{mixed}}-\bar Y_{\mathrm{plant-only}}
\]

or

\[
\widehat{\mathrm{MGR}}
=100\left(
\frac{\bar B_{\mathrm{mixed}}}{\bar B_{\mathrm{plant-only}}}-1
\right).
\]

Pairing changes the **joint sampling design and uncertainty calculation**, not
the treatment means. With equal replication, the mean of paired fitness
differences is algebraically identical to the difference of treatment means.
For biomass, the reported quantity remains the ratio of the two treatment
means; it must not be replaced with the average of seedwise ratios.

CRN improves precision only when the matched outcomes have positive covariance:

\[
\operatorname{Var}(Y_M-Y_P)
=\operatorname{Var}(Y_M)+\operatorname{Var}(Y_P)
-2\operatorname{Cov}(Y_M,Y_P).
\]

This is the basic rationale for CRN in simulation. Glasserman and Yao stress
that the benefit depends on the compared systems responding similarly to the
shared inputs; the same numbers do not guarantee useful positive dependence
([Glasserman & Yao 1992](https://business.columbia.edu/sites/default/files-efs/pubfiles/4261/glasserman_yao_guidelines.pdf)).
Modern RL experiment guidance likewise recommends controlled, paired
comparisons where possible, but specifically advises separating the agent and
environment random streams so that the intended nuisance factors—not accidental
call order—are aligned
([Patterson et al. 2024](https://www.jmlr.org/papers/v25/23-0183.html)).

For MycorMARL, the correlation induced by one shared master seed is presently
unknown and may be weak because `mixed` and `plant-only` have different initial
observations and their learning trajectories diverge immediately. Therefore:

1. Use the same five seed IDs in both modes for the pilot because it costs
   nothing and preserves the option of a paired analysis.
2. Treat the pairing as a diagnostic, not as established variance reduction.
   Report all runs, both marginal treatment distributions, paired scatter, and
   paired differences.
3. Do not make confirmatory inference from five runs. Use the pilot to estimate
   outcome variability and the sign and stability of cross-mode covariance,
   then choose final replication by precision or power requirements.
4. Before the final experiment, replace the single master stream with named,
   separately controlled streams for plant initialization, fungus
   initialization, policy action sampling, minibatch permutation, and future
   environmental fields.
5. Pair held-out heterogeneous evaluation fields across modes in later phases.
   That comparison is conceptually stronger than pairing training seeds because
   it exposes both policy sets to the exact same environmental challenge.

## What a “training seed” represents

A training run is one draw from the performance distribution of a fully
specified learning procedure. Its seed controls algorithmic events such as
network initialization, exploratory action draws, and minibatch order. It is
not a biological replicate, plant identity, or hidden environmental treatment.
Deep-RL outcomes can vary substantially between seeds even under identical
hyperparameters; Henderson et al. demonstrated materially different learning
curves from two groups of five seeds and recommend multiple trials plus
uncertainty reporting
([Henderson et al. 2018](https://doi.org/10.1609/aaai.v32i1.11694)).

The phase-one scientific target can be stated as follows. Let
\(Y_{m,s}(P)\) be deterministic evaluation fitness after training mode \(m\) at
P level \(P\) under algorithm seed \(s\). The target is the difference between
the expected outcomes of two separately optimized learning procedures:

\[
\Delta_{\mathrm{AM}}(P)
=\mathbb E_s[Y_{\mathrm{mixed},s}(P)]
-\mathbb E_s[Y_{\mathrm{plant-only},s}(P)].
\]

Choosing a coupling—matched seeds or independent seeds—does not alter either
marginal expectation when seed IDs are sampled from the same predeclared seed
generation process. It only determines the covariance between the two sample
means. This is why the observationally familiar biomass ratio of treatment
means remains valid under a paired seed design.

In a conventional paired experiment, the simple estimator based on the average
within-pair difference equals the ordinary difference in treatment means; what
changes is its variance analysis
([Wu & Gagnon-Bartsch 2021](https://doi.org/10.3102/1076998620941469)).
The analogy is useful for the algebra, although MycorMARL seed pairs are
algorithmic couplings rather than randomized biological subjects.

## Why matching can help

Pairing aims to make a “fortunate” source of randomness favorable to both
treatments and an “unfortunate” source unfavorable to both. Subtracting within
the pair then cancels some nuisance variation. Examples include:

- identical initial plant network parameters;
- the same environmental realization or start state;
- the same sequence of exploration disturbances;
- the same minibatch ordering.

Patterson et al. give an RL repeated-measures example in which independently
controlled agent and environment seeds expose two agents to the same state
sequence, allowing shared environmental variation to cancel and requiring
fewer agents for a comparison
([Patterson et al. 2024, section 4.4](https://jmlr.org/papers/volume25/23-0183/23-0183.pdf)).
Earlier policy-search work similarly found that fixed starts and random-number
sequences can improve paired policy comparisons, while also documenting the
risk of overfitting to fixed random sequences
([Strens & Moore 2002](https://www.jmlr.org/papers/v3/strens02a.html)).

The gain is largest when the treatment contrast is small relative to
seed-to-seed variability and both modes respond in the same direction to the
shared randomness. It disappears when covariance is near zero and reverses
when covariance is negative. The simulation literature therefore treats
synchronization and structural similarity as substantive requirements, not as
properties conferred merely by using the same integer seed
([Heikes, Montgomery & Rardin 1976](https://doi.org/10.1177/003754977602700301);
[Glasserman & Yao 1992](https://business.columbia.edu/sites/default/files-efs/pubfiles/4261/glasserman_yao_guidelines.pdf)).

## What the current MycorMARL master seed actually aligns

The training script passes one JAX key made from `--seed` into `make_train`
([`scripts/train_ppo.py`](../../scripts/train_ppo.py)). The trainer then follows
the same fixed splitting schedule in both consumer modes
([`mycormarl/algos/ppo.py`](../../mycormarl/mycormarl/algos/ppo.py)):

1. The master key is split into plant-parameter, fungus-parameter, and
   continuing keys. With the same master seed and unchanged architecture, the
   plant policy therefore receives the same initial parameters in `mixed` and
   `plant-only`.
2. On every rollout step, the continuing key is split into plant and fungus
   action keys, then into trade and allocation keys. The two modes therefore
   receive the same underlying standard-normal arrays for the plant's action
   sampling as long as program shapes and split order remain unchanged.
3. PPO minibatch permutations follow the same key schedule. Because the fixed
   two-agent API still constructs and samples the inactive fungal actor in
   `plant-only`, the present call structure remains aligned even though invalid
   fungal updates are skipped.
4. Reset and step keys are also generated identically, but they currently add
   no environmental matching: `BaseMycorMarl.reset` and `step_env` accept keys
   but do not use them
   ([`mycormarl/environments/base_mycor.py`](../../mycormarl/mycormarl/environments/base_mycor.py)).

The modes are nevertheless different from the start. In `plant-only`, fungal
biomass and pools are zero, the fungus is marked dead, hyphal density is zero,
and the plant's association observation is false. In `mixed`, the fungus is
active and co-learns. Thus the same plant parameters are evaluated on different
initial observations, gradients diverge after the first rollout, and matching
the later standard-normal disturbances does not imply matching physical
actions or data. Small implementation changes that alter array shape or PRNG
split order could also silently change what a global seed aligns.

Consequently, current matched master seeds are a plausible but fragile coupling.
They should not be described as “the same training experience.”

## Training-seed pairing versus evaluation-field pairing

These are separate designs and should use separate identifiers.

### Uniform phase one

The environment is deterministic and the primary evaluation uses the learned
latent locations without sampling. There is only one evaluation trajectory per
checkpoint, so all remaining replicate variation is inherited from training.
Matching master training seeds may or may not reduce this variance; the pilot
must measure that rather than assume it.

### Heterogeneous phases

When random initial P fields are introduced, each saved policy should be tested
on a fixed, held-out collection of field IDs. Every `mixed` and `plant-only`
policy should encounter every selected field, yielding a crossed design:

\[
\text{mode}\times\text{training seed}\times\text{held-out field}.
\]

Contrasting modes within the same field directly controls the environmental
realization. It also answers the relevant question: how do the two learned
strategies perform under the same hotspot/dead-zone layout? Field IDs should be
paired in analysis even if training seed IDs ultimately are not.

Training fields and evaluation fields must be disjoint. Reusing a small fixed
set of stochastic sequences while optimizing a policy can create sequence-
specific overfitting, a risk shown in paired policy-search studies
([Strens & Moore 2002](https://www.jmlr.org/papers/v3/strens02a.html)).
Named agent and environment streams also prevent a fungus-only code path from
changing which P field a nominal seed denotes, following the RNG-separation
recommendation of Patterson et al.

## Pilot analysis

For each P level and each of the five predeclared seed IDs:

1. Train one `mixed` and one `plant-only` run with identical training settings
   and the same master seed ID.
2. Evaluate each converged checkpoint deterministically using its latent-location
   policy.
3. Report the five outcomes in each mode, not only their mean and standard
   error. Five seeds are explicitly a range-finding pilot.
4. For fitness, report
   \(\bar Y_M-\bar Y_P\) and display the five matched differences
   \(Y_{M,s}-Y_{P,s}\).
5. For biomass, report the ratio of treatment means
   \(\bar B_M/\bar B_P\), never the mean of \(B_{M,s}/B_{P,s}\).
6. Plot \(Y_{M,s}\) against \(Y_{P,s}\) and \(B_{M,s}\) against \(B_{P,s}\),
   joined or labeled by seed. Record covariance/correlation and compare the
   paired-difference variance with the variance expected from treating the two
   samples independently. With only five pairs, these are descriptive
   diagnostics, not reliable estimates of the population correlation.
7. Also display the two marginal distributions. This ensures the substantive
   result does not depend on an arbitrary seed correspondence.

For a paired uncertainty interval for the biomass ratio of means, resample
whole pairs \((B_{M,s},B_{P,s})\) and recompute the ratio of treatment means in
each resample. Do not bootstrap the two columns independently if the final
analysis declares the seed block to be paired. However, a five-pair bootstrap
should not be used for confirmatory claims: Colas et al. demonstrate that small
seed samples can badly estimate variability and that bootstrap tests can
misstate error rates in deep-RL settings
([Colas, Sigaud & Oudeyer 2018](https://arxiv.org/abs/1806.08295)).

## Decision rule for the final study

After the pilot, choose the final design without reference to whether pairing
makes the AM effect look more favorable:

- **Retain paired training seeds** if cross-mode covariance is consistently
  positive across P levels and endpoints, the alignment mechanism remains
  meaningful, and a named-stream implementation can preserve it.
- **Do not claim a pairing benefit** if covariance is weak, unstable, or
  negative. Continue to use reproducible seed IDs, but analyze independent
  training-run distributions or use a model that does not rely on seed-pair
  covariance.
- **Always retain paired held-out fields** in heterogeneous evaluation, because
  environmental field realization is an explicit nuisance variable and the
  scientific contrast is conditional on the same realized soil field.
- **Set final replication from precision/power needs**, informed by pilot
  variance but allowing for its uncertainty. Five seeds can produce apparently
  different conclusions for identical RL settings; primary RL methodology
  papers consistently recommend multiple runs and interval estimates rather
  than unqualified point estimates
  ([Henderson et al. 2018](https://doi.org/10.1609/aaai.v32i1.11694);
  [Agarwal et al. 2021](https://arxiv.org/abs/2108.13264)).

## Recommended RNG architecture before confirmatory work

A mode-independent run ID should derive named keys rather than pass one mutable
conceptual stream through every component. At minimum:

```text
run_id
├── plant_parameter_initialization
├── fungus_parameter_initialization
├── plant_action_sampling
├── fungus_action_sampling
├── plant_minibatch_permutation
├── fungus_minibatch_permutation
├── training_environment_or_field
└── evaluation_field
```

This makes the intended coupling explicit. For example, plant initialization
and a held-out field can be shared across modes, while fungus initialization is
defined only where biologically relevant. It also permits sensitivity checks
that vary agent randomness while holding a field fixed, or vary fields while
holding trained policies fixed. Independent streams and substreams are a
standard mechanism for maintaining synchronization and independent
replications in simulation
([L'Ecuyer et al. 2002](https://doi.org/10.1287/opre.50.6.1073.358)).

## Bottom line for interpretation

Paired seeds do not make the biomass statistic an average of ratios and do not
create an inconsistency with observational reporting. They are an optional
precision device. The phase-one conclusions should be based on treatment means:
an absolute mean difference for reproductive fitness and a ratio of mean
biomass for greenhouse comparison. Seed pairing is retained for the pilot only
to test whether controlled algorithmic randomness cancels useful nuisance
variation. Paired held-out P fields, by contrast, should be designed explicitly
into the later heterogeneous evaluation.

