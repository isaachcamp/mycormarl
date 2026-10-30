# Rate actions, grid resolution, and update ordering

> **Implementation update, 22 October 2026:** This is the historical integration investigation; wider continuous source/soil reordering remains a proposal.
> The current action/reproduction baseline is [ADR-0020](../adr/0020-use-three-rate-actions-and-stoichiometric-reproductive-output.md),
> verified in the [allocation report](../qualification/three-rate-allocation.md).
> Original analysis is retained below with its original provenance.

15 October 2026. Design investigation requested by the user; no model code changed. Read alongside [the action-space proposal](action-space-design-review.md). **Update 21 October:** the user selected fixed-composition reproduction and three rate controls, superseding the earlier both-model requirement. Reproductive costs are growth costs multiplied by the positive dimensionless `reproduction_cost_scale`, default `1.0`; interval fitness is realised reproductive output. The flexible-model discussion below is historical. File/line references describe the checkout inspected during this investigation.

## Recommendation

Keep rate actions in physical units, independent policy decisions, and conservative finite-volume soil amounts. Treat numerical integration as a shared physical clock for the coupled biological and soil processes. Numerical refinement must improve the approximation to one declared model; it must not change initial resources, action meanings, reproductive fitness, or physical episode duration.

Adopt a coupled flux formulation as the reference model. For implementation, first establish a conservative, positivity-preserving first-order shared-stage solver with small qualified internal steps. Only add higher-order or implicit methods if measured error/cost justifies them. Soil diffusion stability alone is not an adequate biological timestep criterion. Do not promise exact timestep independence for the complete coupled model.

## Four different intervals

**Static-experiment exception, agreed 21 October:** new static-policy screens require `K(delta_t)=-log(0.001)/delta_t`, using the declared outer environment interval (equal to the policy interval for these runs). This intentionally fixes nominal mobilisation per experiment interval and changes physical processing speed if that interval changes. Internal solver refinement must hold the derived rates fixed. See the required static-experiment configuration in [the action-space design](action-space-design-review.md). This exception is not a decision about trade-only runs, which remain deferred.

| Quantity | Meaning | What changing it should do |
| --- | --- | --- |
| `Delta_policy` | Time between opportunities to choose rates | Change feedback opportunities; identical rate schedules must produce identical physics |
| `dt` | Public numerical environment interval | Change output grouping or integration resolution, according to an explicit internal-step contract |
| `h` | Actual coupled integration step | Control numerical error; split exactly at action changes and physical terminal times |
| `dr`, `dz` | Spatial cell widths | Change spatial approximation, not physical uptake capacity or total initial inventory |

Today `dt` is the biological integration step, and soil uses `h_soil = dt / n_soil`; only the policy interval is a grouping layer. `PolicyIntervalMycorMarl` requires an integer number of numerical steps (`environments/policy_interval.py:38–46`). `BaseMycorMarl` computes `n_soil` from diffusion CFL (`environments/base_mycor.py:102–116`). The soil loop advances diffusion and uptake only, with biomass and geometry frozen (`soil/soil.py:260–322`). Increasing `n_soil` therefore does **not** improve maintenance, growth, trade, photosynthesis, or the delay before acquired resources can be spent.

Use an internal step schedule based on absolute physical time, with action/terminal boundaries inserted. Resetting an adaptive integrator or its substep schedule at every policy call can introduce grouping-dependent rounding/approximation. The strongest grouping test keeps the actual integration partition identical while changing policy-wrapper grouping.

## Physical rate model

For each organism, use free stocks `C` in g C, `P` in mg P, and biomass `B` in g dry mass. Growth is

```text
G = k_growth min(C/gamma_C, P/gamma_P)       [g dry mass/day]
T_C = k_trade,plant C_plant                 [g C/day]
T_P = k_trade,fungus P_fungus               [mg P/day]
```

Trade is enabled only while both partners are operational. Each transfer is one flux, debited from donor and credited to recipient using the same accepted amount. Growth similarly has one extent with exactly paired structural C/P costs. Do not independently clip its C and P debits.

The flexible model adds `R_C = k_reproC C`, `R_P = k_reproP P`; instantaneous fitness is `(R_C/gamma_C)^alpha (R_P/gamma_P)^(1-alpha)`. The fixed-composition model adds `R = k_repro min(C/rho_C,P/rho_P)`, consumes `rho_C R`, `rho_P R`, and rewards a specified amount per reproductive unit. Reproductive costs and reward scaling remain scientific decisions. Both models share the same integrator, soil coupling, trade, maintenance, and conservation accounting. Remove the phantom storage hazard as proposed in the action-space review; retained free stock is an outcome rather than a sink.

Together these define balances such as

```text
dC_plant/dt = A(B_plant) - M_C,plant - gamma_C G_plant - R_C - T_C
dP_plant/dt = U_plant   - M_P,plant - gamma_P G_plant - R_P + T_P
dC_fungus/dt = T_C     - M_C,fungus - gamma_C G_fungus - R_C
dP_fungus/dt = U_fungus- M_P,fungus - gamma_P G_fungus - R_P - T_P
```

Substitute fixed-composition reproductive costs when selected. Biomass changes by growth minus structural loss. Maintenance and starvation require an explicit boundary law: an unconstrained constant maintenance sink cannot continue subtracting from an empty pool. Keep maintenance priority if intended, but implement it as a declared resource/boundary constraint, not an accidental priority created by the order of large whole-step updates. Recommended boundary semantics: acquisition and trade arrive continuously; maintenance has first claim on inflow whenever a free pool is exhausted; the remaining maintenance deficit causes structural loss with the existing maximum-of-C/P-deficits rule. Resolve death when biomass reaches the declared death threshold, rather than before crediting all interval inflow. This is a proposed scientific decision, not a claim about current behaviour; accept or amend it explicitly before implementation. Do not invent a within-step starvation rescue simply by putting all sources first in a large step.

## Why order matters even without storage

The current order is maintenance and possible death, outgoing trade, growth/reproduction, incoming trade, photosynthesis, geometry reconstruction, then soil diffusion/uptake (`environments/base_mycor.py:323–514`). This grants trade first access to donor stock, delays acquired resources by one biological step, and grants post-growth tissue a full step's acquisition. These are finite-step splitting effects; they disappear only under suitable refinement away from boundary events. Reversing the order merely chooses a different approximation.

A source-only example shows the acquisition lag: with growth-equivalent pool initially zero, source `I=1/day`, growth rate `2/day`, and duration `.1 day`, growth-then-source updates produce cumulative growth `0` with one step, `.00708063` with four, and `.00922369` with 64. The continuous balance gives `.00936538`. Every update conserves resource; coarse steps still delay growth. These values follow the scalar recurrence, not a full environment rollout.

A simple check isolates trade/growth competition. Start with one growth-equivalent unit of donor C and sufficient P, set trade and growth to `1/day`, disable all other processes. Current trade-first updates over `h=0.1 day` export `1-exp(-0.1)=0.095163` C-equivalent units, then grow `exp(-0.1)*(1-exp(-0.1))=0.086107`. Simultaneous competing first-order C sinks instead give each `(1-exp(-0.2))/2=0.090635`. Total depletion agrees in this special example; allocation to processes does not. The example was evaluated with Python's `math.exp`, not an environment rollout. It shows why exact exponential formulas for individual updates do not remove process-priority error.

For pure isolated growth, fixed costs, no other process, and no cap, `Delta_B = Q0*(1-exp(-k_growth*T))`, where `Q0=min(C0/gamma_C,P0/gamma_P)`. This has exact subdivision invariance. The user's `dt=.025`, policy intervals `.1` versus `.025`, identical commands, should match because each runs four identical numerical steps. More frequent policy sampling can produce different commands and therefore different outcomes.

## Spatial scaling: what is already correct

The axisymmetric cell volume is

`V_ij = pi*(r_outer^2-r_inner^2)*(z_outer-z_inner)`.

Canonical soil stock is `M_ij = c_ij V_ij (theta+b_P)` in micromol P (`soil/phosphate_grid.py:108–131, 209–246`). Annuli of equal widths have unequal volumes. Mean concentrations and inventory comparisons must therefore be volume-weighted, not cell averages.

Uptake integrates represented absorber length `L_ij=lambda_ij V_ij`, multiplies by lateral area `2*pi*a*L_ij`, a surface flux in micromol/cm²/s, and the actual step duration in seconds (`soil/phosphate_uptake.py:25–53, 190–219`). No extra `dr`, `dz`, or number-of-cells multiplier should be added to action rates or physiological traits. Root and fungal requests share one proportional cap against each cell's current inventory (`soil/phosphate_uptake.py:357–381`). This is conservative and avoids root-first competition.

Diffusion uses face conductances `G_ij=D_l*theta*f*A_face/d_centres` and conservative paired transfers `G_ij*(c_i-c_j)*h_seconds` (`soil/phosphate_diffusion.py:46–98, 120–163`). Buffered apparent diffusivity is `D_app=D_l*theta*f/(theta+b_P)`. Buffering is included in the amount/concentration relation; do not divide the amount-flux coefficient by buffering a second time.

The exact explicit diffusion positivity ceiling is

`h_CFL,seconds = min_i[V_i*(theta+b_P)/sum_j G_ij]`.

Current code enforces a safety fraction of this ceiling by subcycling (`soil/phosphate_diffusion.py:166–207`). On a uniform interior grid it scales approximately with `1/[D_app*(1/dr²+1/dz²)]`; halving both widths requires roughly four times as many explicit steps when the CFL bound controls. The cylindrical centre has its own face geometry; use the implemented cellwise bound rather than a Cartesian shortcut.

With current defaults `dr=dz=.1 cm`, `D_l=1e-5 cm²/s`, `theta=.3`, `f=.308`, `b_P=239`, the derived `D_app` is `3.8613e-9 cm²/s` and the uniform-grid bound is about `7.4937 days` before the `.8` safety factor. Thus `dt=.025 day` is far below the diffusion ceiling and normally uses one soil substep. This arithmetic is a dimensional check, not a qualification of uptake or biology accuracy.

Grid interval validation already rejects non-divisible requests and gives the nearest admissible interval; it does not silently shorten the boundary cell (`soil/phosphate_grid.py:12–104`). Defaults give 500 radial by 1000 vertical cells over radius 50 cm, depth 100 cm (`params.py:30–47`). Keep physical extent fixed during spatial convergence. Enlarging a domain changes inventory and boundary effects, and is a different experiment.

## Spatial approximation that remains

Root and fungal fronts are represented by fractional cell occupation, preserving represented length well without a whole-cell entry jump (`plant/roots.py:46–136`; `fungus/mycelium.py:125–238`). Geometry is reconstructed from current biomass as required by ADR-0004; no root/hyphal age or historical footprint is tracked.

However the closure consumes **volume-averaged** length density in front cells. It uses that density nonlinearly for territory radius, sparse resistance, and the shared hyphal-overlap blend (`soil/phosphate_uptake.py:56–159`; `soil/soil.py:29–92`). A partially occupied saturated cell is therefore treated as uniformly sparse, and the fungus-dependent blend can also change plant uptake. Refining cells changes this approximation even if total represented length is unchanged. This is a closure/discretization issue, not evidence of an omitted volume factor or an automatically proven bug. Qualify it with fixed biomass/front-position tests before considering occupied/unoccupied subcell quadrature. Do not change the agreed uptake closure incidentally during an action refactor.

Root layer radii also depend on depth-layer averaged root-distribution weights, so changing `dz` refines the geometric shape, not just diffusion. In-domain length is clipped when geometry exceeds the configured domain; ensure boundary clipping is absent or explicitly recorded in convergence experiments.

The production uptake reference time is a fixed physical closure parameter (`uptake_reference_time_days`, default one day). It is **not** numerical `dt`, policy interval, environment age, or absorber cohort age. Sparse propagation distance is `sqrt(D_app*T_ref)` and overlap blending also uses `T_ref` (`soil/phosphate_uptake.py:70–159`). Replacing it by `dt` would change the constitutive model under timestep refinement. Cohort ages would require a separate model and state.

## Practical update scheme

1. Store an absolute physical clock. Hold actions piecewise constant in physical time. Choose internal steps that end exactly at action changes and the physical horizon. Use fixed qualified `h` initially; permit extra conservative subdivisions for large hazards or fast depletion.
2. At each shared integration stage reconstruct living geometry from that stage's biomass; derive soil concentration from canonical inventory. Compute photosynthesis, soil uptake, trade, maintenance demand, growth, and reproduction from the **same stage state**. Shared-stage evaluation prevents plant-first/fungus-first and source/sink ordering from becoming implicit biological rules.
3. Form one stoichiometric transaction ledger. Resolve uptake competition simultaneously within each cell. Share each accepted transfer between its two endpoints. Preserve exact C/P growth and reproduction costs. If a positivity safeguard scales a resource-limited process, scale its full extent and all associated debits/credits together. Do not repair negative pools by clipping after the update.
4. Integrate accepted fluxes over `h`; update all stocks together. Recompute state-dependent fluxes and geometry on the next stage. Resources acquired in one microstep can participate in the next under the held action, irrespective of when the policy next observes them. Treat the lag as a numerical approximation of `h`, not a policy-duration lockout.
5. Resolve resource-exhaustion/death boundaries using the declared maintenance law. Shorten steps or use a compatible nonnegative boundary solver; disable all subsequent processes of a dead participant. Integrate only to the biological event and administrative horizon. A constant-demand maintenance sink at zero stock needs boundary handling, so a generic `h < stock/demand` rule alone is insufficient.
6. Accumulate fluxes and fitness using the same accepted stage amounts and quadrature. Report boundary stocks, integrated fluxes, enabled durations, and event flags separately. Policy grouping only aggregates this record.

A simple common-state forward-Euler transaction is a transparent first reference, with substeps/rejection keeping positivity and error controlled away from events. It will not preserve exact exponential depletion automatically. Preserve an exact isolated-sink kernel or use it as an analytic convergence reference; do not incorrectly require a generic first-order scheme to match exponentials at every finite `h`. If exact isolated-hazard behavior is retained with exponential subtransactions, document their splitting error and show convergence of process allocations. A positivity-preserving SSPRK2 or midpoint-based method can be evaluated later, but limiting-resource switches, caps, and death boundaries prevent a blanket second-order guarantee. Symmetric splitting alone is not second order if its component solvers remain first order.

The shared scheme is a target architecture; current diffusion-first/uptake-second soil subcycling is a valid first-order approximation with its own splitting error. If performance requires separate soil and biology clocks, qualify a multirate coupling with consistent exchanged flux integrals and geometry staging; finer soil steps alone cannot substitute for that coupling.

## Step selection and qualification

Diffusion CFL controls positivity of diffusion, not accuracy of the complete dynamics. Also monitor `h*k` for active hazards, `h*U_i/M_i` for cell depletion, maintenance exhaustion times, relative biomass/geometry changes, and limiting-resource switches. Inventory capping remains a last-resort positivity/conservation safeguard, not proof of accurate depletion timing or root/fungus shares. Pick tolerances from convergence requirements; no arbitrary physiological constants are proposed here. If unbounded softplus rates make explicit steps impractical, either constrain physical rates on scientific grounds or use a stiff-safe sink solver. Do not silently cap rates for numerical convenience.

Validate the selected fixed-composition reproduction model with:

- **Grouping:** identical `dt`, action schedule, initial state, physical horizon, and stochastic forcing; compare `.025` versus `.1` policy intervals over the same four numerical updates. Include growth/reproductive C/P use, trades, all stocks, soil field, reward, and death provenance. Use physical-time-indexed random forcing if stochastic dynamics are introduced.
- **Time:** fixed fine spatial grid and commands; refine coupled `h` and demonstrate convergence of stocks, integrated fluxes, rewards, and event times. Include nonzero trade/reproduction, source-fed growth, starvation, high rates, imbalance, cap contact, and changing resource limitation.
- **Space:** fixed domain, identical continuous initial concentration field, fixed physical parameters and biomass; refine `dr` and `dz` independently with sufficiently fine `h`. Compare total inventory/length, local concentration profiles, front-cell uptake, organism budgets, and reward. Project fields conservatively to common comparison cells.
- **Joint:** refine both space and time after their separate tests. More expensive spatial runs must not hide temporal error because the scheduler happened to add soil substeps.
- **Budget:** pair trade debits/credits; soil loss equals accepted organism gain after the micromol-to-mg conversion; account separately for retained free stocks, structural incorporation/loss, reproduction, and maintenance. Diagnose cap activation and domain-clipped geometry.

For flexible reproduction integrate instantaneous Cobb–Douglas fitness. In general `sum_h CD(Delta_C_h,Delta_P_h)` differs from `CD(sum_h Delta_C_h,sum_h Delta_P_h)` when composition varies; aggregating components before applying the reward would make policy grouping change the objective. The current wrapper correctly sums the numerical rewards. Fixed-composition reproduction uses cumulative reproductive extent directly, with its declared scaling.

## Additional findings to retain in the bug notes

Verified coupling in source, separate from proposed scientific changes:

- **Reset inventories depend on numerical timestep.** Unspecified initial free pools become `kappa * B0 * dt` (`environments/base_mycor.py:247–254`). With default nonzero C maintenance, refining `dt` changes initial C stocks. Explicitly pin initial pools for convergence studies; replace the fallback with a physical reserve specification if numerical independence is intended.
- **Trade-history observation is tied to numerical resolution.** `last_received` is overwritten with final-step incoming trade (`environments/base_mycor.py:452–457`) and normalized by `kappa * B * dt` (`149–203`; `observations.py:19–42`). The policy wrapper returns the last substep observation, not a whole-interval exchange rate. For a steady transfer and biomass the amount/reference factors cancel `dt` to leading order, so this is not automatically a unit error. Changing `dt` nevertheless changes the observation window and its finite-step bias, and can change the feature even for matched boundary stocks. Choose a physical-time exchange rate/window, explicitly declare final-integration-step semantics, or provide a fixed physical reference; do not multiply a whole-interval amount by a numerical-step denominator.
- **Existing known storage/wrapper bugs remain relevant.** See the action-space review for phantom storage, partial-interval execution loss, reported realised trade, horizon overrun, and incomplete diagnostic aggregation. This investigation does not reimplement or close them.

Whole-step maintenance priority, memoryless uptake geometry, fixed `T_ref`, independent bilateral gifts, and a shared hyphal-dependent uptake blend are documented model choices. They should not be filed as verified bugs solely because an alternative biology is possible.

## ADR changes before implementation

Supersede ADR-0012's storage rule and its fixed statement that acquisition first participates at the next numerical substep with the chosen coupled-stage semantics. Preserve its separation of policy decisions from physical rates. ADR-0004 currently specifies post-allocation geometry before uptake; evaluating geometry at common integration stages changes that staging statement while retaining its memoryless biomass principle, so amend explicitly. ADR-0010's finite complete initial soil inventory and volume-weighted spatial treatment remain compatible. Keep the environment independent of PPO (ADR-0001); physical action validation and provenance still follow ADR-0003.
