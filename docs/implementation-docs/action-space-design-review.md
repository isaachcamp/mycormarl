# Action-space and action-application design review

> **Implementation update, 22 October 2026:** This is the historical design review and decision record; its selected three-rate subset is now implemented.
> The current action/reproduction baseline is [ADR-0020](../adr/0020-use-three-rate-actions-and-stoichiometric-reproductive-output.md),
> verified in the [allocation report](../qualification/three-rate-allocation.md).
> Original analysis is retained below with its original provenance.

Prepared 2026-10-15; reproductive-model decision updated 2026-10-21. No environment implementation has been changed. Code and existing design documents are the evidence for the findings below. Mathematical examples are derived directly from their equations.

## Accepted reproductive-model decision — 21 October 2026

The user selected stoichiometrically constrained reproduction with the same essential-resource form as growth. This supersedes the earlier request to support both reproductive models. The selected action is three nonnegative per-day rates: `[k_trade, k_growth, k_reproduction]`. Retention is implicit, with no storage hazard.

Use each organism's growth resource costs as the reproductive baseline. Introduce a finite, strictly positive dimensionless trait `reproduction_cost_scale`, default `1.0`, multiplying both C and P costs. The same trait is available for plant and fungus, each using its own growth stoichiometry. With scale `s`:

```text
rho_C = s * gamma_C
rho_P = s * gamma_P
G = k_growth * min(C/gamma_C, P/gamma_P)
R = k_reproduction * min(C/rho_C, P/rho_P)
growth consumption = (gamma_C*G, gamma_P*G)
reproductive export = (rho_C*R, rho_P*R)
interval reward = integral(R dt)
```

These are instantaneous flux definitions for the proposed coupled model. Realised growth and reproduction must compete through the same resource-conserving integrator, rather than independently spending the full pools in one finite step. Reproduction is exported production and does not increase the parent's living biomass or uptake geometry. Reward is realised reproductive output in reproductive dry-mass-equivalent units, not a literal seed/spore count.

At `s=1`, reproduction and growth have identical costs per output unit. At `s=2`, each reproductive unit costs twice as much C and P, preserving the C:P ratio. At fixed pools and reproductive rate, output is halved; reproductive resource-export flux is unchanged because the increased cost and reduced output cancel. This parameter controls cost per unit output/fitness, not processing speed.

Existing four-dimensional policies are incompatible with this schema. Update action construction, actor dimensions, static controls, transition provenance, qualification fixtures, and checkpoint metadata together. Explicitly supersede ADR-0012's four-component/storage contract when implementing; `CONTEXT.md` continues to describe current implemented behavior until migration.

### Accepted sampling and competition design — 21 October 2026

The user accepted sampling three Gaussian latents and mapping them to trade hazard `kt=softplus(zt)`, total biological hazard `kb=softplus(zb)` (both per day), and reproductive expenditure share `a=sigmoid(za)`. The physical three-rate action is `[kt, (1-a)*kb, a*kb]`. Growth and reproduction draw on the same essential-resource availability; their combined resource expenditure is `kb*min(C/gamma_C,P/gamma_P)` in growth-equivalent units per day. The share divides resource expenditure, not output when reproductive costs differ.

Trade and biological demand compete simultaneously through the coupled resource balances. Trade has no unconditional first claim. There is no single allocation simplex spanning both elemental pools; donor-specific trade must not reserve a matching fraction of the non-traded element. Ending free stocks explicitly account for finite-rate retention and genuine stoichiometric surplus. Positive rate transforms do not themselves ensure feasible finite-step execution: the integrator must prevent overdrawing while preserving paired stoichiometric costs and trade credits/debits.

Illustrative, reproducible case studies are in [action-allocation-case-studies/README.md](action-allocation-case-studies/README.md). These isolate the accepted flux definitions using a high-accuracy ODE calculation; they are not production-environment rollouts or selection of the eventual JAX integrator.

### Required configuration for new static-policy experiments — 21 October 2026

The user confirmed retaining rate actions. For future static allocation experiments, including parameter screens, specify the total commanded intensity as a function of the declared experiment interval `delta_t`. Policy and outer environment intervals are equal for these experiments. With target residual fraction `epsilon=0.001` (99.9% nominal mobilisation per interval):

```text
K(delta_t) = -log(epsilon) / delta_t       # delta_t in days; K in day^-1
kt = p * K
kb = (1-p) * K
kg = (1-a) * kb
kr = a * kb
physical action = [kt, kg, kr]
```

The static allocation strategy fixes trade share `p` and reproductive expenditure share `a` for each organism/run. Thus `kt+kg+kr=K`; growth-only static strategies use `a=0`, while other static strategies may specify nonzero reproduction. These runs do not use a learned allocation policy. At `delta_t=0.05 days`, the required total is approximately `138.1551056 day^-1`.

**Run requirement:** derive and check `K` against the declared experiment interval before executing a new static-policy experiment. Record `delta_t`, policy interval, `epsilon`, derived `K`, each organism's allocation shares, reproductive cost scale, and resulting physical rate vector with the run configuration/results. A manually supplied `K` must agree with the declared formula and target within an explicitly recorded numerical tolerance; do not silently reuse a daily-total constant at a different interval. This is a requirement for future experiment setup, not an already implemented runtime check.

`delta_t` here is the declared outer allocation/environment interval, not an internal biological or soil solver substep. Hold the derived rates unchanged during internal integration. Changing the declared interval and recomputing `K` deliberately changes the physical processing speed to preserve nominal per-interval mobilisation; it is not a pure numerical-convergence comparison. For convergence studies, hold physical commands fixed while refining internal numerical steps.

The 99.9% value is an isolated-pool competing-hazard reference. It does not guarantee 99.9% depletion of both elemental pools under stoichiometric limitation, acquisition, maintenance, or other constraints. The share `p` is a commanded intensity share, not a guaranteed realised fraction of initial stock traded.

**Scope:** this interval-dependent `K` requirement applies to new static-control allocation experiments only. The user explicitly deferred trade-only experiments: neither the fixed-throughput proposal nor the separate trade-first/all-remainder-to-growth proposal is selected for those runs. General learned-policy sampling remains the accepted rate design with separately sampled total biological intensity; do not impose this static-run `K` constraint globally.

The production integration scheme and maintenance/death boundary semantics remain unresolved proposals. Historical four-control and both-model recommendations below are superseded by this section.

## Historical recommendation — superseded above

Keep rate commands and the separate policy interval. Define actions as controls of biological processes in a coupled resource-balance model, rather than fractions of a repeatedly partitioned pool. Remove the current storage hazard: resources retained in the existing free pool need no additional outflow process.

Preserving the current Cobb–Douglas reproductive objective, the recommended action has **four dimensions per agent**:

| Control | Meaning | Plant resource effect | Fungal resource effect |
| --- | --- | --- | --- |
| `k_trade` | Donor-resource export hazard, per day | Exports free C to fungus | Exports free P to plant |
| `k_growth` | Essential-resource growth intensity, per day | Consumes C and P at plant tissue stoichiometry | Consumes C and P at fungal tissue stoichiometry |
| `k_reproduction_c` | Reproductive C-export hazard, per day | Exports free C | Exports free C |
| `k_reproduction_p` | Reproductive P-export hazard, per day | Exports free P | Exports free P |

This uses the present action budget more meaningfully: replace storage with independent reproductive investment in the two resources. Do not add separate C/P growth controls for fixed-composition biomass; realised growth has one process extent and cannot consume the elements independently.

The user wants to test **both reproductive models**, with a selectable model configuration and corresponding action/reward contracts. Current reproduction exports independently valued amounts of C and P and computes Cobb–Douglas fitness; it is not fixed-stoichiometry propagule production. The flexible-investment option uses the four controls above. The fixed-composition propagule option uses **three** controls (`trade`, `growth`, `reproduction`) and separately specified reproductive stoichiometry. These are alternative biological models, not merely alternative scalar reward functions. Actor construction, static controls, action metadata, and checkpoint compatibility must follow the selected model. Neither option is being discarded; details such as reproductive stoichiometry remain to be agreed.

## What the code actually does

The rate contract is documented in `CONTEXT.md` and ADR-0012. Actual actions are four nonnegative rates; PPO applies softplus to four Gaussian latents (`algos/ppo.py:154`). The base environment consumes these commands without clipping. The public non-policy constructor does sanitize inputs (`actions.py:6`), including replacing infinities and NaNs with zero; that distinction matters when diagnosing invalid callers.

An environment step performs the following sequence (`environments/base_mycor.py:323`):

1. Pay maintenance from initial free C/P pools; unmet maintenance removes biomass.
2. Decide death before trade, growth, incoming resources, or photosynthesis.
3. Export plant C and fungal P with fraction `1 - exp(-k_trade * dt)`.
4. Apply growth/reproduction/storage hazards to the remaining pools.
5. Credit received trade.
6. Add photosynthate from post-growth biomass, rebuild uptake geometry, and acquire soil P.

The held-policy wrapper repeats this entire sequence with the same rate action and sums rewards (`environments/policy_interval.py:64`). New resources can participate on the next numerical substep even though the policy does not act again. That is a useful property to preserve.

### Shared C/P allocation does not destroy the non-limiting resource

For growth, the current code computes an offered fraction of each pool, then realises `min(C_offer/gamma_C, P_offer/gamma_P)`. It subtracts **only resources actually used** (`base_mycor.py:700`). Remaining non-limiting C or P stays in the free pool (`base_mycor.py:718`). This is explicitly intended in the older interface plan (`docs/implementation-docs/ippo-actor-interface-implementation-plan.md:266`).

The real restriction is the set of accessible decisions. A common reproduction rate forces the same proportional reproductive extraction of remaining C and P, although trade modifies only one donor resource. That prevents an agent from independently changing reproductive composition. It is distinct from violating resource conservation.

Essential-resource growth itself legitimately links C and P. If an organism has surplus P but insufficient C and no P-export route, the P cannot all become fixed-composition biomass. Retention is the physically correct result unless the model adds another destination, such as excretion or explicit storage. Increasing action dimensions cannot remove the underlying stoichiometric constraint.

### Finite-rate retention is not missing allocation

For an isolated first-order sink, `Q(T) = Q(0) exp(-k T)`. Any finite rate and finite time leave a positive remainder. This is part of the rate model, not a missing-resource accounting term. At every boundary the entire resource stock must be accounted for as a flux, living structure, retained pool, or explicitly recorded loss; it need not all leave its pool.

Numerically exhausting pools every environment step would reintroduce timestep-dependent biology and can deprive the next step of maintenance reserves. If exact finite-time liquidation is a scientific requirement, it needs an explicit finite-capacity or event process, not an infinite hazard or a hidden end-of-step sweep.

### Current storage is a timestep-dependent phantom process

In `_apply_allocation` (`base_mycor.py:689`), let `H = k_g + k_r + k_s`. The offered growth fraction is

`a_g(dt) = k_g/H * (1 - exp(-H dt))`.

Storage enters `H` but transfers nothing out of either free pool. The alleged stored portion remains available to every process on the next substep. Thus it suppresses growth/reproduction in a finite step while having no first-order effect as `dt -> 0`:

`a_g(dt) = k_g dt - 0.5 k_g H dt^2 + O(dt^3)`.

With balanced initial C/P equivalent to one unit biomass, no sources or maintenance, `k_g=1`, `k_r=0`, over one day:

| Storage rate | One step | Ten steps | 100 steps | 1,000 steps |
| --- | --- | --- | --- | --- |
| 0 | 0.632121 | 0.632121 | 0.632121 | 0.632121 |
| 1 | 0.432332 | 0.613291 | 0.630277 | 0.631937 |
| 10 | 0.090908 | 0.465087 | 0.613829 | 0.630282 |

These are cumulative growth from the recurrence `Q_next = Q * [1 - k_g/H * (1 - exp(-H/n))]`, repeated `n` times. A storage policy trained at one resolution can lose its protective effect at finer resolution. This is a defect in the mathematical interpretation of storage as a competing hazard, even though the implementation follows ADR-0012's stated retention-in-free-pool choice.

For real storage, introduce distinct C/P storage compartments, import/export fluxes, mobilisation, costs, and access rules for maintenance. That is a new biological model requiring evidence and calibration. Implicit retention is the appropriate minimal default.

## Recommended mathematical contract

For organism `i`, let `C_i`, `P_i` denote free resource stocks, `B_i` living biomass, and `gamma_C,i`, `gamma_P,i` fixed structural resource costs. For the four-control model:

```text
G_i   = k_growth,i * min(C_i / gamma_C,i, P_i / gamma_P,i)
R_C,i = k_reproduction_c,i * C_i
R_P,i = k_reproduction_p,i * P_i
T_C   = k_trade,plant * C_plant          if both partners operational
T_P   = k_trade,fungus * P_fungus        if both partners operational

dB_i/dt = G_i - structural_loss_i

dC_plant/dt = photosynthesis - M_C,plant - gamma_C,plant*G_plant - R_C,plant - T_C
dP_plant/dt = uptake_plant - M_P,plant - gamma_P,plant*G_plant - R_P,plant + T_P
dC_fungus/dt = -M_C,fungus - gamma_C,fungus*G_fungus - R_C,fungus + T_C
dP_fungus/dt = uptake_fungus - M_P,fungus - gamma_P,fungus*G_fungus - R_P,fungus - T_P

fitness_rate_i = (R_C,i / gamma_C,i)^alpha * (R_P,i / gamma_P,i)^(1-alpha)
reward_i = integral(fitness_rate_i dt)
```

The maintenance terms above denote accepted resource consumption; starvation and structural loss require a separately specified boundary/event rule. These equations do not silently decide whether current maintenance priority or mortality should change.

Trade is a simultaneous pair of independently commanded transfers, preserving current semantics. It is not matched exchange at a price: positive transfer by one agent does not require positive transfer by the other. If matching, transport capacity, interface costs, or reciprocal commitments are desired, they require additional model rules.

The growth expression is the current allocation model's continuous-time limit. Its single extent guarantees stoichiometric debits. The proposed reproductive controls allow composition to vary while retaining the existing reward law. They also permit an unproductive one-element export when the other rate is zero; that is allowed by the current constitutive model too and should be observable rather than silently repaired. Fixed-composition propagule production avoids that behavior by changing the reproductive model.

With fixed-composition reproduction, replace the two reproductive exports with one extent `R_i = k_reproduction,i * min(C_i/rho_C,i, P_i/rho_P,i)` and subtract `rho_C,i R_i` and `rho_P,i R_i`. A direct reward proportional to `R_i` would then have a clear propagule interpretation. Reusing growth stoichiometry or choosing a new reproductive stoichiometry must be an explicit scientific choice.

## Numerical integration and larger timesteps

### Agreed timing requirement — 15 October 2026

The user expects cumulative growth resource use over a fixed physical horizon to be independent of how numerical steps are grouped into policy intervals, provided the same physical rate commands are executed. In particular, over 0.1 days, `dt=0.025` with one held 0.1-day decision must match `dt=0.025` with four 0.025-day decisions that each supply the identical rate vector. Both execute the same four numerical updates. Resource acquisition must remain eligible at the same numerical boundaries in both cases. Compare cumulative realised growth C/P use, biomass, and final pools; offered allocation is a separate diagnostic. Keep initial state, physical episode horizon, and any random forcing identical, and compare before reset. This is a grouping-invariance requirement, including when storage is absent; it does not require the policy to produce identical actions when given more opportunities to observe and act.

Changing numerical `dt` is a different requirement. Pure isolated first-order growth without maintenance, trade, reproduction, acquisition, or a biomass cap has exact subdivision invariance: for initial growth-equivalent resource `Q0`, growth over `T` is `Q0*(1-exp(-kg*T))`. This also holds for initially imbalanced C/P pools with fixed costs, using `Q0=min(C0/gammaC,P0/gammaP)` and retaining the stoichiometric surplus. The full coupled model should converge under numerical refinement, rather than promising exact equality for arbitrary numerical step sizes. No-storage alone does not eliminate process-splitting error.

Rate commands separate decisions from numerical resolution; they do not by themselves make the coupled environment accurate at arbitrary `dt`. The current exponential formula is exact for an isolated sink, not for the whole coupled system.

Today trade is applied before growth/reproduction, incoming trade is delayed until after allocation, and photosynthesis/soil uptake occur last. This grants trade a finite-step priority and creates a one-numerical-step delay for essential inputs. Refinement changes the delay in physical time. Maintenance can kill an organism before same-step photosynthesis, uptake, or transfer could sustain it (`base_mycor.py:323`, `base_mycor.py:369`, `base_mycor.py:487`). Post-growth biomass receives a full step of photosynthesis and uptake capacity. These are splitting choices that need convergence assessment, not additional policy dimensions.

Specify the common flux system first, then choose a positivity-preserving, conservative integrator or sufficiently small biological substeps. Exact exponential updates are appropriate for verified isolated linear processes; retain paired equal-and-opposite trade fluxes. A generic explicit Euler update with unbounded policy hazards is not safe for nonnegative pools. Clipping pools after overdrawing them would conceal conservation errors and change the decision semantics.

An implicit method, conservative flux limiter, or controlled internal biological substepping are implementation candidates, not interchangeable scientific specifications. Flux limiting must preserve growth stoichiometry, specify priority or proportional competition, and converge to the stated equations. Maintenance/death events and resource acquisition must receive the same scrutiny as allocation. Soil already has its own substep scheduler; biological processes need their own accuracy/stability treatment.

Large outer environment intervals may contain smaller internal biological and soil steps while the action stays held. There is no requirement that a larger API interval be a single inaccurate biological update. Unbounded hazards can make the problem stiff; a justified physiological upper bound could help, but should not be invented merely to conceal a solver limitation.

Integrate fitness as a rate. Current Cobb–Douglas has degree one, so `F(h*x,h*y)=h*F(x,y)` and the current per-step computation is consistent with a rate interpretation when composition is constant. Applying Cobb–Douglas once to the total exports of an entire held interval changes the objective when export composition changes during that interval. Keep the integral/sum of instantaneous reproductive fitness, as the wrapper currently does (`policy_interval.py:102`).

## Rates, fractions, and PPO differentiability

The strongest reason for rates is physical-time semantics. For an isolated hazard over reference horizon `T`, a fraction is just another parameterization: `f=1-exp(-k*T)`, `k=-log(1-f)/T` for `0<=f<1`. The repository already provides this conversion (`trade_only.py:17`). A fraction defined over a fixed physical horizon can therefore be used without numerical-step coupling; repeatedly applying that fraction per numerical step cannot.

Differentiability does not distinguish rates from fractions. Softplus, sigmoid, and the earlier simplex transform are differentiable in their interiors. The growth `min` has a kink at co-limitation under either parameterization; mortality and event rules add further nonsmooth behavior.

More importantly, current PPO does not obtain its actor update by differentiating the environment trajectory. It stores sampled latents and computes Gaussian log likelihood ratios at those samples (`algos/ppo.py:817`, `algos/ppo.py:873`), then differentiates that loss (`algos/ppo.py:909`). The environment explicitly stops gradients through returned state and observations (`base_mycor.py:612`). Thus action/environment differentiability is not required by the present PPO estimator. It could matter for a future pathwise or differentiable-planning algorithm, which would need a broader audit of the full environment.

Keep the policy distribution and scored random variable explicit. Fixed softplus transforms can remain. Exact zero is not reached by softplus at finite latents; an exact no-trade or association-off decision needs an explicit gate if scientifically meaningful. Association mode is already a separate concept in `CONTEXT.md`; do not conflate it with a tiny trade rate.

## Additional action-interface findings

The storage resolution artifact and lost partial-interval actor execution are the highest-priority findings: both can change learned strategies when numerical resolution or decision duration changes. Whole-interval provenance and horizon overrun are next. Constructor policy, exact-zero access, reproduction law, and matched-versus-independent transfer are explicit interface or scientific choices, rather than automatically bugs.

* **Wrapper loses partial-interval execution.** `policy_interval.py:112` builds `allocation_executed` from final operational status. If an agent executes earlier substeps and dies later, its action may have changed state and reward but the whole interval is marked invalid for the biological actor. Aggregate whether it executed at any point, alongside lifecycle facts and integrated fluxes.
* **Wrapper realised trade disagrees with base semantics.** The base zeros the realised trade rate when the partner is unavailable (`base_mycor.py:541`); the wrapper reports the entire requested action for an end-alive agent (`policy_interval.py:117`), even in a single-consumer mode where trade never ran. A single rate vector also cannot adequately describe partial-interval execution duration. Preserve requested controls, enabled duration, and actual integrated resource transfers separately.
* **Terminal substeps are not frozen by the wrapper.** The scan runs the complete configured number of numerical steps without checking administrative termination (`policy_interval.py:66`). The base also continues biological updates on an already terminal state. When a horizon lies inside the held interval, this can execute actions beyond it. Require aligned horizons or stop/freeze at the boundary; validate and test the chosen contract.
* **Most flux diagnostics aggregate, but several remain last-step values.** The wrapper explicitly sums reproduction, uptake, trade, growth, and plant carbon fixed (`policy_interval.py:92`) but retains only final-substep growth-resource usage/allocation, maintenance diagnostics, and cancellation flags. Whole-interval budgets therefore cannot safely be reconstructed from all current info fields. Define which are states, integrated fluxes, rates, or event flags, then aggregate accordingly.
* **The constructor hides malformed non-policy commands.** `rate_action` replaces invalid values by zero, and tests require it (`tests/test_actions.py:19`). This differs from strict invalid-action failure discussed in ADR-0003. Decide explicitly whether permissive construction is desired; retain a strict validation route for scientific callers.
* **Existing convergence coverage does not exercise the problematic action modes.** The analytical held-action test uses growth only with zero storage (`tests/test_decision_intervals.py:59`); the fixed coupled numerical qualification also uses zero reproduction/storage (`docs/qualification/rate-action-49/phosphate-numerical/phosphate-numerical-qualification.md:84`). These checks cannot validate the claimed storage semantics or general competing process behavior.
* **Static 90/10 biology is a hazard ratio, not a guaranteed realised budget ratio.** `trade_only.py:12` and `trade_only.py:30` split a total hazard into 90 percent growth and 10 percent reproduction. With stoichiometric imbalance, unrealised growth remains in the free pools while reproduction exports both offered resource quantities, so realised C/P consumption or biomass production need not follow 90/10. Report it as the commanded hazard split and expose actual fluxes separately.

## Validation required for a redesign

1. Account independently for C and P: initial pools plus sources equal final pools plus structural incorporation, reproductive export, recorded maintenance/mortality losses, and net trade. Never count an offered but unrealised growth budget as consumption.
2. Check exact stoichiometric growth, no negative pools, paired trade conservation, zero flux for non-operational participants, and explicit cap behavior.
3. Test pool-only analytic cases: single sink, multiple true sinks, no-op retention, fixed stoichiometric imbalance, independently varied reproductive C/P, incoming trade, and limiting-resource switches.
4. Hold one physical-time policy schedule fixed while refining numerical steps over the same horizon. Compare every stock, cumulative flux, reward, lifecycle outcome, and policy observation. Include nonzero trade, reproduction, starvation, extreme rates, and resource acquisition; test near death boundaries separately because events need special tolerances.
5. Test whole-interval transition aggregation and PPO masks when an agent dies midway, a partner is absent, or an administrative horizon occurs inside an interval.
6. Confirm latent sampling, likelihood ratios, executed controls, static controls, and artifact version checks share the same action schema. Four dimensions with changed meanings are still an incompatible contract; old checkpoints must not load as though equivalent.
7. Requalify the numerical timestep and retrain policies after changing dynamics. A policy trained to exploit phantom storage or finite-step process priority is not an evaluation of the new model.

## Decisions before implementation

The 21 October decision above selects fixed-composition reproduction, growth-based costs with `reproduction_cost_scale=1.0` by default, and three rate controls. The earlier both-model requirement is superseded. Preserve implicit retention, independently commanded trade, and held rate actions. Separately decide whether maintenance priority/death timing is intended biology or a numerical approximation, and whether finite-time exact liquidation or protected storage is scientifically required. These decisions change the model and should precede integrator selection.

Supersede the storage portion of ADR-0012 explicitly; maintain ADR-0001's environment/PPO separation and ADR-0003's executed-command/likelihood alignment. Update `CONTEXT.md` only once meanings are agreed. The parallel biological simplification work in `docs/biological-environment-review.md` explicitly depends on this action contract and should consume the agreed result.
