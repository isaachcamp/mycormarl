# Continuous and sparse closures for phosphate uptake

> **Scope:** This summary covers the continuous and sparse phosphate-uptake approximations, their smooth blending rule, and the origin of that rule. It does not cover the finite-volume diffusion solver except where its transport coefficients enter the uptake closures.
>
> **Evidence note:** The uptake physics and numerical formulas are implemented and tested. The sparse-to-continuous transition rule, its one-day reference time, exponent, and use of hyphal density alone are explicitly provisional modelling choices rather than a transition law taken directly from one paper.

## What is a closure?

A model is *closed* when every unknown needed to advance it can be calculated from the state variables it actually stores. Here, the soil grid stores one labile phosphate amount per cell. It does **not** resolve the microscopic phosphate concentration profile around every root or fungal hypha. Yet uptake depends on concentration at an absorber's surface, not just the cell-average concentration.

A **closure** supplies the missing sub-grid relationship. It replaces an unresolved spatial problem—many cylindrical absorbers, local diffusion, sorption, and surface uptake—with an algebraic request for phosphate based on resolved cell quantities. It is therefore an approximation chosen to make the coarse model solvable, not merely a software module or a boundary condition.

This code provides two alternative closures:

- The **continuous closure** assumes depletion fields are sufficiently coupled or homogenised that absorber-surface concentration equals cell bulk-solution concentration.
- The **sparse closure** assumes absorbers have separate cylindrical territories and estimates a locally depleted surface concentration from a quasi-steady diffusion–uptake balance.

The model computes both requests and interpolates between them. It never adds them as separate sinks.

## Shared uptake kinetics and geometry

For absorber length density $\lambda$ (cm absorber $cm^{-3}$ soil) in a cell of volume $V$ ($cm^3$), represented length and lateral area are

$$
L = \lambda V,
\qquad
A = 2\pi r_a L,
$$

where $r_a$ is root or hyphal radius. Surface influx follows Michaelis–Menten kinetics,

$$
J(C_s) = J_{\max}\frac{C_s}{K_m+C_s},
$$

where $C_s$ is phosphate concentration at the absorbing surface, $J_{\max}$ is maximum influx, and $K_m$ is the half-saturation concentration. The two closures differ principally in how they obtain $C_s$.

The common flux–area–time calculation is visible in [`continuous_uptake_request`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L25-L53) and [`sparse_uptake_request`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L190-L219). The kinetic kernel follows the classical mechanistic root-uptake tradition of Barber and Tinker–Nye; the repository's numerical defaults are attributed via Schnepf & Roose (2006), rather than independently fitted for this model.

## Continuous closure

The continuous closure makes the strongest homogenisation assumption:

$$
C_s \approx C_b,
$$

where $C_b$ is the cell's bulk solution concentration. Its requested uptake over timestep $\Delta t$ is

$$
U_{\mathrm{cont}}
= J_{\max}\frac{C_b}{K_m+C_b}
  (2\pi r_a\lambda V)\Delta t.
$$

In plain language, every represented centimetre of root or hypha sees the same cell-average phosphate concentration. The closure is cheap and appropriate when neighbouring depletion zones overlap enough that treating each absorber independently would be artificial. It does not solve or store a radial concentration profile.

```python
surface_flux = michaelis_menten_surface_flux(concentration, j_max, k_m)
return surface_flux * absorbing_area * days_to_seconds(dt_days)
```

Implemented by [`continuous_uptake_request`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L25-L53). Its arithmetic, zero limits, monotonic response, and JAX compilation are tested in [`test_continuous_phosphate_uptake.py`](../tests/test_continuous_phosphate_uptake.py#L18-L96).

## Sparse closure

The sparse closure assigns each absorber an equivalent circular soil territory. Since one unit of absorber length is allocated cross-sectional soil area $1/\lambda$,

$$
R_{\mathrm{soil}} = \frac{1}{\sqrt{\pi\lambda}}.
$$

The radial depletion profile is allowed to propagate only as far as either that territory boundary or a buffered diffusion distance over reference time $T_{\mathrm{ref}}$:

$$
R_{\mathrm{eff}}
= r_a + \min\!\left[
\sqrt{D_{\mathrm{app}}T_{\mathrm{ref}}},
\max(R_{\mathrm{soil}}-r_a,0)
\right].
$$

Here $D_{\mathrm{app}}$ is the apparent diffusivity governing propagation through buffered soil,

$$
D_{\mathrm{app}} = \frac{D_l\theta f_l}{\theta+B},
$$

with solution diffusion coefficient $D_l$, water content $\theta$, impedance factor $f_l$, and linear buffer power $B$. The code deliberately uses $D_{\mathrm{app}}$ for propagation but $D_{\mathrm{flux}}=D_l\theta f_l$ for amount supply, because buffering is already represented when stored labile amount is converted to solution concentration. This separation is implemented in [`apparent_diffusivity_cm2_s`](../mycormarl/mycormarl/soil/phosphate_diffusion.py#L20-L38) and [`sparse_uptake_resistance`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L93-L121).

The closure then balances quasi-steady cylindrical diffusive supply with surface uptake:

$$
\frac{2\pi D_{\mathrm{flux}}L(C_b-C_s)}
{\ln(R_{\mathrm{eff}}/r_a)}
= 2\pi r_aLJ_{\max}\frac{C_s}{K_m+C_s}.
$$

Defining the concentration-like resistance

$$
k = \frac{r_aJ_{\max}\ln(R_{\mathrm{eff}}/r_a)}{D_{\mathrm{flux}}},
$$

gives a quadratic whose physical root is

$$
C_s = \frac{C_b-K_m-k+
\sqrt{(C_b-K_m-k)^2+4C_bK_m}}{2},
\qquad 0\le C_s\le C_b.
$$

Finally,

$$
U_{\mathrm{sparse}}
=J_{\max}\frac{C_s}{K_m+C_s}
 (2\pi r_a\lambda V)\Delta t.
$$

The code evaluates two algebraically equivalent forms of the quadratic root depending on the sign of $C_b-K_m-k$, avoiding catastrophic cancellation, and explicitly handles zero and infinite resistance:

```python
rationalised_root = 2.0 * bulk * k_m_array / safe_denominator
direct_root = 0.5 * (a + discriminant_root)
surface = jnp.where(a >= 0.0, direct_root, rationalised_root)
```

Implemented by [`territory_radius_cm`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L56-L67), [`effective_uptake_radius_cm`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L70-L90), [`sparse_uptake_resistance`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L93-L121), [`sparse_surface_concentration`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L162-L187), and [`sparse_uptake_request`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L190-L219). Tests check territory geometry, finite propagation, the quadratic residual and bounds, cancellation resistance, flux arithmetic, and zero-supply behaviour in [`test_sparse_phosphate_uptake.py`](../tests/test_sparse_phosphate_uptake.py#L28-L174).

## How the model moves between closures

There is no Boolean switch. Hyphal density defines a smooth, shared continuous-regime weight. For local hyphal length density $\lambda_h$, hyphal radius $r_h$, and apparent diffusivity $D_{\mathrm{app}}$, the estimated time for adjacent hyphal depletion zones to meet is

$$
R_h=\frac{1}{\sqrt{\pi\lambda_h}},
\qquad
\ell_{\mathrm{gap}}=\max(R_h-r_h,0),
\qquad
t_{\mathrm{diff}}=\frac{\ell_{\mathrm{gap}}^2}{D_{\mathrm{app}}}.
$$

This is the diffusion scaling $\ell\sim\sqrt{Dt}$ rearranged as a timescale. The dimensionless overlap ratio and continuous weight are

$$
\Omega=\frac{T_{\mathrm{ref}}}{t_{\mathrm{diff}}},
\qquad
w_{\mathrm{cont}}
=\frac{\Omega^p}{1+\Omega^p}
=\frac{1}{1+(t_{\mathrm{diff}}/T_{\mathrm{ref}})^p}.
$$

Thus $t_{\mathrm{diff}}\ll T_{\mathrm{ref}}$ gives $w_{\mathrm{cont}}\to1$, while $t_{\mathrm{diff}}\gg T_{\mathrm{ref}}$ gives $w_{\mathrm{cont}}\to0$. The midpoint is $t_{\mathrm{diff}}=T_{\mathrm{ref}}$, where $w_{\mathrm{cont}}=1/2$. The exponent $p$ controls transition sharpness; it does not change the midpoint.

Each consumer is blended with the **same hypha-derived weight**:

$$
U_i=(1-w_{\mathrm{cont}})U_{i,\mathrm{sparse}}
 +w_{\mathrm{cont}}U_{i,\mathrm{cont}},
\qquad i\in\{\mathrm{root},\mathrm{fungus}\}.
$$

```python
return (1.0 - weight) * sparse + weight * continuous
```

The overlap time and stable weight are implemented in [`hyphal_overlap_time_seconds`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L124-L142) and [`continuous_regime_weight`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L145-L159); blending is in [`blend_uptake_requests`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L222-L235). [`uptake_geometry_coefficients`](../mycormarl/mycormarl/soil/soil.py#L45-L78) computes both sparse resistances and this weight from post-growth geometry. The complete transaction computes both alternatives, blends them, and only then caps combined root–fungal demand against cell inventory in [`blended_uptake_transaction`](../mycormarl/mycormarl/soil/phosphate_uptake.py#L258-L354).

The limits and nominal overlap calculation are tested in [`test_sparse_phosphate_uptake.py`](../tests/test_sparse_phosphate_uptake.py#L39-L67); a mixed-cell integration test independently reconstructs both closures, the shared weight, and the post-blend cap in [`test_p3_environment_uptake.py`](../tests/test_p3_environment_uptake.py#L160-L232).

## Where the switching condition comes from

The condition has three layers of provenance:

1. **Established physical idea:** mechanistic rhizosphere models represent nutrient transport to cylindrical roots and nonlinear surface uptake. Tinker & Nye (2000) synthesize this framework, while Roose, Fowler & Darrah (2001) derive long-time cylindrical-root uptake approximations. Schnepf & Roose (2006) extend mechanistic phosphate modelling to external mycorrhizal hyphae and explicitly discuss overlapping hyphal depletion zones.
2. **Repository-derived geometry and parameters:** the code converts local hyphal length density to an equivalent cylindrical territory and uses buffered diffusivity. Its nominal saturated density is a provisional 2-D-to-3-D conversion from the imaged fungal networks of Oyarte-Galvez et al. (2025), not a direct soil measurement. With $\lambda_h=168.75\ \mathrm{cm}^{-2}$, $r_h=5\times10^{-4}\ \mathrm{cm}$, and $D_{\mathrm{app}}\approx3.86\times10^{-9}\ \mathrm{cm^2\,s^{-1}}$, the repository obtains $t_{\mathrm{diff}}\approx5.5$ days.
3. **Project-specific closure decision:** the repository compares that estimated overlap time with configurable $T_{\mathrm{ref}}$, then uses a Hill-type smooth weight. The defaults $T_{\mathrm{ref}}=1$ day and $p=2$ are set in [`EnvConfig`](../mycormarl/mycormarl/params.py#L27-L42). They were chosen provisionally as a static proxy for absorber exposure/lifetime because local colonisation age is not stored. The design rationale says this explicitly in [`phosphate-uptake-todos.md`](../implementation-docs/phosphate-uptake-todos.md#L306-L350).

At the nominal saturated density these defaults give $\Omega\approx0.18$ and $w_{\mathrm{cont}}\approx0.032$: still overwhelmingly sparse, despite hyphae being present. No hyphae or no apparent diffusion gives $t_{\mathrm{diff}}=\infty$ and exactly $w_{\mathrm{cont}}=0$, so root-only cells always use the sparse closure. Very high hyphal density collapses the gap to zero and gives exactly $w_{\mathrm{cont}}=1$.

This should not be described as a literature-derived universal threshold. Schnepf & Roose reported overlapping depletion zones in their own model, but the exact territory formula, shared root–fungus weight, one-day timescale, exponent two, and use of current density without colony age are the present repository's modelling choices. The documentation also notes that $T_{\mathrm{ref}}$ affects both the blend and the sparse propagation radius, so sensitivity is not necessarily monotonic.

## Relationship to the literature

| Source | Relevant contribution | Relationship to this implementation |
|---|---|---|
| [Tinker & Nye (2000)](https://doi.org/10.1093/oso/9780195124927.001.0001) | Mechanistic treatment of solute movement and uptake in the rhizosphere. | Foundation for cylindrical transport plus Michaelis–Menten-type uptake; the code reduces the spatial problem to algebraic cell closures. |
| [Roose, Fowler & Darrah (2001)](https://doi.org/10.1007/s002850000075) | Explicit long-time approximation for uptake by a single cylindrical root and upscaling to root systems. | Closest mathematical precedent for replacing a resolved radial uptake problem with a tractable analytical approximation; the repository's sparse quadratic is its own simplified balance. |
| [Schnepf & Roose (2006)](https://doi.org/10.1111/j.1469-8137.2006.01771.x) | Coupled root and external-hyphal phosphate model; parameterisation and depletion-zone overlap analysis. | Main direct scientific precedent and source chain for several provisional transport and kinetic parameters. It motivates overlap as a regime diagnostic, but not the exact implemented smooth weight. |
| [Oyarte-Galvez et al. (2025)](https://doi.org/10.1038/s41586-025-08614-x) | Quantitative imaging of architecture and trade in arbuscular-mycorrhizal networks. | Source for the planar saturation-density observation that the repository converts, provisionally, into a 3-D hyphal length density used by the overlap calculation. |

## Assumptions and limitations

- **Code-backed:** continuous uptake sets $C_s=C_b$; sparse uptake computes a temporary $C_s\le C_b$; both alternatives are blended before one shared inventory cap.
- **Documented choice:** only hyphal density determines regime weight. Root density affects root sparse resistance but cannot by itself make a cell continuous.
- **Interpretation:** the continuous closure is best read as a homogenised-depletion approximation, not proof that concentration is literally uniform within a cell.
- **Limitation:** the sparse closure assumes quasi-steady, cylindrical, non-interacting territories; residual sub-cell root–fungus interference is not resolved.
- **Limitation:** $T_{\mathrm{ref}}$ is both the transition comparison time and the sparse propagation horizon, coupling two modelling roles.
- **Limitation:** the transition is static. It does not track time since hyphal arrival, ageing, turnover, or an explicit depletion-front state.
- **Calibration need:** $T_{\mathrm{ref}}$, $p$, the density conversion, buffering, diffusivity, and kinetic parameters require sensitivity analysis and experimental calibration before the regime classification can be treated quantitatively.

## References

1. Tinker, P. B., & Nye, P. H. *Solute Movement in the Rhizosphere*. Oxford University Press (2000). [DOI](https://doi.org/10.1093/oso/9780195124927.001.0001).
2. Roose, T., Fowler, A. C., & Darrah, P. R. “A mathematical model of plant nutrient uptake.” *Journal of Mathematical Biology* 42, 347–360 (2001). [DOI](https://doi.org/10.1007/s002850000075).
3. Schnepf, A., & Roose, T. “Modelling the contribution of arbuscular mycorrhizal fungi to plant phosphate uptake.” *New Phytologist* 171, 669–682 (2006). [DOI](https://doi.org/10.1111/j.1469-8137.2006.01771.x).
4. Oyarte-Galvez, L. et al. “A travelling-wave strategy for plant–fungal trade.” *Nature* (2025). [DOI](https://doi.org/10.1038/s41586-025-08614-x).
