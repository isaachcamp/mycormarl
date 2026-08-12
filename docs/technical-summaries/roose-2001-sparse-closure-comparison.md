# Roose et al. (2001) versus the repository sparse uptake closure

> **Question.** Do the sparse analytical closure in MycorMARL and the
> single-root solution of Roose, Fowler & Darrah (2001) solve the same problem,
> are their formulae equivalent, and is time dependence missing from the
> implementation?
>
> **Conclusion.** They share the same local cylindrical diffusion plus
> Michaelis--Menten uptake mechanism, after neglecting pore-water advection,
> but they do **not** solve the same complete initial-boundary-value problem and
> the implemented formula is not Roose et al.'s formula as written. Their
> surface-concentration quadratics become algebraically identical if the
> repository's effective outer radius is replaced by Roose et al.'s
> time-dependent matched radius. MycorMARL's cell bulk concentration already
> evolves with diffusion and uptake; what it lacks is the *age/history of the
> unresolved radial depletion profile*. Roose et al.'s prescribed far-field
> concentration does not itself decline.

## Sources and scope

This comparison uses the authors' deposited copy of the paper, [Roose, Fowler
& Darrah, *A mathematical model of plant nutrient uptake*, Journal of
Mathematical Biology 42, 347--360 (2001)](https://people.maths.ox.ac.uk/fowler/papers/2001.2.pdf),
DOI [10.1007/s002850000075](https://doi.org/10.1007/s002850000075), and the
repository implementation and tests. No secondary account is used. The paper
explicitly excludes mycorrhizae and root hairs, so only its single cylindrical
root result can be compared to the generic root/hypha sparse closure; its root
system branching model is a separate upscale calculation (paper pp. 348 and
355--358, especially eqs. 3.11--3.18).

## Are they solving the same problem?

Only at the level of the local physical ingredients.

| Feature | Roose et al. (2001) | MycorMARL sparse closure |
|---|---|---|
| Absorber | One uniform cylindrical root of radius $a$ | A represented cylindrical root or hypha of radius $r_a$ |
| Soil domain | Infinite radial domain, $r\ge a$ | Finite quasi-steady annulus, $r_a\le r\le R_{\mathrm{eff}}$ |
| Transport | Transient radial diffusion; radial pore-water advection is present initially and then normally neglected for small $Pe$ | Steady radial diffusion is used only as a sub-grid resistance; the coarse soil field separately undergoes transient finite-volume diffusion |
| Buffering | Instantaneous linear adsorption, $c_s=bc$, giving capacity $\phi+b$ | Linear capacity $\theta+b_p$ |
| Root boundary | Michaelis--Menten flux $F_m c/(K_m+c)$ | Michaelis--Menten flux $J_{\max}C_s/(K_m+C_s)$ |
| Outer/initial data | Initially uniform $c=c_0$ and prescribed $c\to c_0$ as $r\to\infty$ | Current cell bulk concentration $C_b$ is imposed at $R_{\mathrm{eff}}$ for the temporary annular profile |
| Time | Time since uptake began controls depletion-layer spreading | Fixed configurable $T_{\mathrm{ref}}$ controls the resistance; no local absorber/depletion age is stored |
| Other absorbers | Assumed far enough away for an isolated-root, "dilute network" approximation | A density-derived territory caps $R_{\mathrm{eff}}$; a separate hypha-derived weight blends sparse and continuous requests |

Roose et al.'s dimensional model is (paper eqs. 2.4--2.7)

$$
(\phi+b)\frac{\partial C}{\partial t}
=\phi D\frac1r\frac{\partial}{\partial r}
\left(r\frac{\partial C}{\partial r}\right)
$$

when pore-water advection is neglected, with

$$
\phi D\left.\frac{\partial C}{\partial r}\right|_{r=a}
=F_m\frac{C(a,t)}{K_m+C(a,t)},\qquad
C(r,0)=c_0,\qquad C(\infty,t)=c_0.
$$

The paper derives this from conservation of dissolved plus instantaneously
buffered nutrient (eqs. 2.1--2.4), estimates that $Pe$ is usually small (pp.
350--351), and obtains its explicit approximation by a large-time matched
asymptotic solution (eqs. 2.14--2.24). Thus its time dependence is the radial
profile's response to a root placed in an initially uniform, effectively
infinite reservoir. The far-field value $c_0$ is a constant boundary datum,
not an evolving finite "bulk pool."

The repository instead sets

$$
D_{\mathrm{flux}}=D_l\theta f_l,
\qquad
D_{\mathrm{app}}=\frac{D_{\mathrm{flux}}}{\theta+b_p},
$$

then solves the steady annular balance

$$
\frac{D_{\mathrm{flux}}(C_b-C_s)}{r_a\ln(R_{\mathrm{eff}}/r_a)}
=J_{\max}\frac{C_s}{K_m+C_s}.
$$

The finite radius is

$$
R_{\mathrm{eff}}=r_a+
\min\left(\sqrt{D_{\mathrm{app}}T_{\mathrm{ref}}},
R_{\mathrm{soil}}-r_a\right),
\qquad
R_{\mathrm{soil}}=(\pi\lambda_{\mathrm{length}})^{-1/2}.
$$

These choices are implemented in
[`effective_uptake_radius_cm`](../../mycormarl/mycormarl/soil/phosphate_uptake.py#L70-L90)
and
[`sparse_uptake_resistance`](../../mycormarl/mycormarl/soil/phosphate_uptake.py#L93-L121).
The density-derived cap, fixed reference time, and later sparse/continuous
blend have no counterpart in Roose et al. (2001).

## Where the effective uptake resistance comes from

The resistance is the exact steady diffusive resistance of a cylindrical
annulus, combined with the scale of the surface uptake kinetics. Consider one
straight absorber of length $L$. In the temporary sub-grid problem there is no
production or consumption between its surface $r=r_a$ and an outer matching
radius $r=R$, so steady radial conservation gives

$$
\frac{1}{r}\frac{d}{dr}\left(r\frac{dC}{dr}\right)=0.
$$

Integrating twice and applying $C(r_a)=C_s$ and $C(R)=C_b$ gives the
logarithmic cylindrical profile

$$
C(r)=C_s+(C_b-C_s)
\frac{\ln(r/r_a)}{\ln(R/r_a)}.
$$

The inward diffusive supply through the absorber surface is therefore

$$
Q_{\mathrm{diff}}
=2\pi r_aL D_{\mathrm{flux}}
\left.\frac{dC}{dr}\right|_{r=r_a}
=\frac{2\pi L D_{\mathrm{flux}}}
{\ln(R/r_a)}(C_b-C_s).
$$

Equivalently, the purely transport part is a resistance per unit length,

$$
\mathcal R_{\mathrm{diff}}
=\frac{\ln(R/r_a)}{2\pi D_{\mathrm{flux}}},
\qquad
\frac{Q_{\mathrm{diff}}}{L}
=\frac{C_b-C_s}{\mathcal R_{\mathrm{diff}}}.
$$

Surface uptake over the same length is

$$
Q_{\mathrm{uptake}}
=2\pi r_aLJ_{\max}\frac{C_s}{K_m+C_s}.
$$

Equating supply and uptake cancels $2\pi L$ and yields

$$
C_b-C_s
=\underbrace{\frac{r_aJ_{\max}}{D_{\mathrm{flux}}}
\ln\left(\frac{R}{r_a}\right)}_{k}
\frac{C_s}{K_m+C_s}.
$$

Thus the code's `resistance` $k$ is not the bare transport resistance
$\mathcal R_{\mathrm{diff}}$. It is that resistance multiplied by the maximum
uptake capacity per unit length, $2\pi r_aJ_{\max}$, and consequently has
units of concentration. It measures the concentration drop that diffusion
would have to sustain at the kinetic scale $J_{\max}$. Increasing absorber
radius, maximum influx, or the logarithmic travel distance increases $k$ and
lowers $C_s$; increasing $D_{\mathrm{flux}}$ decreases $k$ and brings $C_s$
closer to $C_b$. Absorber length does not appear in $k$ because both diffusive
supply and absorbing area scale linearly with $L$.

### Why the closure needs an effective outer radius

A nonzero steady flux to an isolated cylinder cannot be matched to a finite
constant concentration at radial infinity: the cylindrical solution varies as
$\ln r$. The steady annular closure therefore needs a finite outer location
where its unresolved local profile is joined to the resolved cell value
$C_b$. $R_{\mathrm{eff}}$ is this **matching scale**. It is not a larger
physical absorber radius, and it is not an explicitly tracked sharp depletion
front.

The repository bounds that scale in two different ways:

1. Over the proxy exposure time $T_{\mathrm{ref}}$, a concentration
   disturbance can propagate only a distance of order
   $\sqrt{D_{\mathrm{app}}T_{\mathrm{ref}}}$. The apparent diffusivity
   $D_{\mathrm{app}}=D_{\mathrm{flux}}/(\theta+b_p)$ is used here because
   dissolved diffusion must also change the instantaneously buffered pool as
   the profile advances. Measured outward from the absorber surface, this
   gives the candidate radius
   $r_a+\sqrt{D_{\mathrm{app}}T_{\mathrm{ref}}}$.
2. Absorbers at length density $\lambda_{\mathrm{length}}$ cannot each be
   assigned an unlimited independent soil volume. One unit of absorber length
   has soil volume $1/\lambda_{\mathrm{length}}$ and hence cross-sectional
   area $1/\lambda_{\mathrm{length}}$. Replacing that area by an equivalent
   circle gives

   $$
   \pi R_{\mathrm{soil}}^2
   =\frac{1}{\lambda_{\mathrm{length}}},
   \qquad
   R_{\mathrm{soil}}
   =(\pi\lambda_{\mathrm{length}})^{-1/2}.
   $$

   Capping the matching radius at $R_{\mathrm{soil}}$ prevents neighbouring
   absorbers from each claiming the same surrounding soil as an independent
   sparse territory.

Together these arguments give the implemented rule

$$
R_{\mathrm{eff}}
=r_a+\min\left[
\sqrt{D_{\mathrm{app}}T_{\mathrm{ref}}},
\max(R_{\mathrm{soil}}-r_a,0)
\right].
$$

The two bounds have different status. The diffusion length is a dimensional
propagation estimate, while the territory radius is an area-partitioning
construction. Imposing $C_b$ at their minimum is the closure assumption; a
real territory boundary in an interacting array would generally require a
symmetry or coupled-neighbour condition, not necessarily $C=C_b$. When the
territory cap is reached, depletion zones are beginning to interact and the
isolated sparse-annulus picture is least secure. The repository handles that
regime separately through its sparse/continuous blend.

Finally, $T_{\mathrm{ref}}$ is a fixed proxy rather than the age of a local
root or hypha. The effective radius is therefore reconstructed from current
geometry on every step; it does not carry a depletion front through time. The
Roose matched radius below supplies a more precise transient matching rule for
an isolated root, and makes clear which part of the implemented radius is a
project-specific approximation.

## Exact algebraic relationship

Let $C_s$ denote concentration at the absorber and define the repository
resistance

$$
k=\frac{r_aJ_{\max}}{D_{\mathrm{flux}}}
\ln\left(\frac{R_{\mathrm{eff}}}{r_a}\right).
$$

The annular balance gives

$$
C_b-C_s=k\frac{C_s}{K_m+C_s},
$$

or

$$
C_s^2-(C_b-K_m-k)C_s-C_bK_m=0.
$$

The repository evaluates the stable physical root of this quadratic in
[`sparse_surface_concentration`](../../mycormarl/mycormarl/soil/phosphate_uptake.py#L162-L187).

For $Pe=0$, Roose et al.'s matched inner solution gives, in their dimensionless
variables (paper eqs. 2.22--2.24),

$$
c_\infty-c_1=\frac{F}{2}L(t),\qquad
F=\lambda\frac{c_1}{1+c_1},
$$

with

$$
L(t)=\ln(1+4e^{-\gamma}t),\qquad
t=\frac{D_{\mathrm{app}}t_D}{a^2},\qquad
\lambda=\frac{aF_m}{D_{\mathrm{flux}}K_m}.
$$

Returning to dimensional concentration gives

$$
C_b-C_s=
\frac{aF_m}{D_{\mathrm{flux}}}\frac{L(t)}2
\frac{C_s}{K_m+C_s}.
$$

This is **exactly the same quadratic** as the repository closure if

$$
\ln(R_{\mathrm{eff}}/a)=\frac{L(t)}2,
$$

that is, if

$$
\boxed{R_{\mathrm{Roose}}(t_D)
=\sqrt{a^2+4e^{-\gamma}D_{\mathrm{app}}t_D}}.
$$

This mapping also reproduces the paper's closed flux formula (eq. 2.24). A
direct numerical substitution through the repository quadratic gave agreement
between the boundary flux and eq. 2.24 to $3.1\times10^{-9}$ in dimensionless
flux for a representative phosphate parameter set. The shared quadratic is
therefore not a coincidence: the quasi-steady logarithmic annulus is the inner
part of Roose et al.'s long-time matched solution. What differs is the rule for
the outer matching radius.

The current uncapped repository rule,

$$
R_{\mathrm{repo}}=a+\sqrt{D_{\mathrm{app}}T_{\mathrm{ref}}},
$$

is not $R_{\mathrm{Roose}}$. At long time the latter has propagation distance
approximately $2\sqrt{e^{-\gamma}}\sqrt{D_{\mathrm{app}}t_D}\approx
1.50\sqrt{D_{\mathrm{app}}t_D}$, whereas the repository uses coefficient one
and adds the distance to $a$. The two can happen to be close for a particular
radius and reference time, but are not generally equivalent. Once
$R_{\mathrm{soil}}$ caps the repository radius they are explicitly different
problems.

## What actually evolves in each model?

### Roose et al.

- $c_0$ (and hence dimensionless $c_\infty$) is both the initial concentration
  and the prescribed concentration at radial infinity (paper eqs. 2.6--2.9).
  It does not decline with uptake.
- The surface concentration $c_1(t)$, radial depletion profile, and root flux
  $F(t)$ evolve. The logarithmic factor $L(t)$ increases, so flux from a fixed
  far-field concentration decreases as the depletion layer spreads.
- In the paper's whole-root-system calculation, root length distributions also
  evolve through growth and branching (section 3), adding a separate source of
  time dependence to system uptake.

### MycorMARL

- `State.soil_labile_p` is a finite cellwise amount. Every soil substep first
  redistributes that amount by conservative diffusion, derives a fresh solution
  concentration, removes accepted root/fungal uptake, and stores the remaining
  amount. See
  [`soil_diffusion_uptake_substep_with_diagnostics`](../../mycormarl/mycormarl/soil/soil.py#L190-L230),
  [`labile_amount_to_solution_concentration`](../../mycormarl/mycormarl/soil/phosphate_grid.py#L221-L235),
  and the inventory update in
  [`_apply_blended_uptake_with_diagnostics`](../../mycormarl/mycormarl/soil/soil.py#L115-L151).
  Therefore the bulk concentration **does evolve**.
- The sparse surface concentration also changes whenever current $C_b$ changes,
  because it is recalculated each substep. Its resistance, however, is held
  fixed over the biological step and reconstructed from current geometry with
  the same fixed $T_{\mathrm{ref}}$ on later steps
  ([`evolve_soil_p`](../../mycormarl/mycormarl/soil/soil.py#L233-L285)).
- There is no state for time since a root/hypha arrived, no stored radial
  profile, and no depletion-front radius. Newly represented absorbing length
  therefore immediately receives the same reference-age resistance as older
  length at the same density.

The concern is consequently real but narrower than "the bulk concentration
doesn't evolve": macro-scale bulk depletion is implemented; transient
*sub-grid depletion age* is not.

## Does something need to change?

### No correction is needed merely to make bulk concentration evolve

That behaviour is already present and is covered by repository tests: uptake
reduces canonical labile amount and thus its derived concentration, while
finite-volume diffusion redistributes it. Replacing the current closure solely
because Roose et al.'s $F(t)$ contains time would conflate their fixed far-field
reservoir with MycorMARL's finite evolving cells.

### A change is needed if the intended claim is “we implement Roose et al. (2001)”

The present implementation should continue to be described as a project-specific
quasi-steady sparse closure, not as Roose et al.'s transient analytical solution.
The existing
[`phosphate-uptake-closures.md`](phosphate-uptake-closures.md)
already makes this distinction, but the mathematical mapping above should be
used anywhere stronger provenance is claimed.

### A small change can match Roose at one fixed reference age, but not add memory

If a literature-grounded fixed-age closure is desired without adding state,
replace the uncapped propagation rule by

$$
R_{\mathrm{eff}}=
\min\left[R_{\mathrm{soil}},
\sqrt{a^2+4e^{-\gamma}D_{\mathrm{app}}T_{\mathrm{ref}}}
\right].
$$

Before the territory cap, this makes the repository quadratic exactly equal to
Roose et al. eq. 2.24 evaluated at $T_{\mathrm{ref}}$ and current $C_b$. It is
still a fixed pseudo-age closure, and using a changing $C_b$ is a frozen-bulk
coupling approximation rather than the original fixed-$c_\infty$ solution.
The territory cap and sparse/continuous blend remain project-specific.

### Genuine Roose-style time dependence requires new state and semantics

A truly transient closure would need a local exposure age or depletion-front
state for roots and fungi. It would also need rules for newly grown length,
mixed-age cohorts in one cell, turnover/retraction, recolonisation, merging
depletion zones, and whether diffusion/replenishment can reset the effective
age. Feeding global simulation time into eq. 2.24 would be incorrect: roots and
hyphae appear at different times, and Roose et al.'s derivation assumes a fixed
initial/far-field concentration.

This would reopen the repository's accepted memoryless-geometry decision:
[`ADR-0004`](../adr/0004-derive-uptake-geometry-from-current-biomass.md)
explicitly says the model does not currently retain a historical front and
that equal current biomass and traits imply equal uptake geometry regardless
of history. It should therefore be treated as a modelling extension requiring
an ADR decision and validation, not as a one-line bug fix.

## Recommendation

1. **Keep the current macro concentration evolution.** It addresses finite
   inventory and coarse spatial redistribution, which Roose et al.'s isolated
   fixed-far-field problem does not.
2. **Do not call the current sparse formula equivalent to Roose et al.** Call it
   a steady-annulus closure whose algebra matches the inner Roose relation under
   a different effective-radius rule.
3. **For the smallest scientifically motivated correction**, use
   $R_{\mathrm{Roose}}(T_{\mathrm{ref}})$ before the existing territory cap and
   add a regression comparison against eq. 2.24. This improves provenance but
   does not resolve depletion age.
4. **Only add time-dependent resistance if depletion history is an intended
   state variable.** First specify the cohort/reset/overlap semantics and reopen
   ADR-0004; then qualify the coupled model against the full transient PDE, not
   only against eq. 2.24 with global time.

On present evidence, the implementation is not broken in the way suspected:
its bulk concentration is dynamic. It is, however, a different and more
memoryless sub-grid approximation than Roose et al.'s transient isolated-root
solution. Whether to change it is therefore a model-scope decision; the
fixed-age Roose radius is the defensible minimal revision, while true transient
depletion requires a larger state-model change.
