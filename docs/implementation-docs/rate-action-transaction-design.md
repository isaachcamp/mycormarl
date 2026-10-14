# Rate-action transaction design

This note records the proposed implementation semantics for GitHub issue #49,
consistent with [ADR-0012](../adr/0012-adopt-rate-actions-for-held-policy-intervals.md).
It is a design proposal pending agreement on the upper bounds of the rates.

## Action contract

A Rate action is

$$
[k_{\mathrm{trade}}, k_{\mathrm{growth}}, k_{\mathrm{reproduction}},
k_{\mathrm{storage}}],
$$

where every component has units of $\mathrm{d}^{-1}$ and is non-negative.
`storage` means retention in the existing free carbon and phosphorus pools; it
does not introduce a new compartment or a mobilisation process.

The proposed validity range is $[0, \infty)\,\mathrm{d}^{-1}$ for each
component, with non-finite values rejected. The numerical update below remains
bounded for every finite rate. If a finite physiological maximum is preferred,
it must be specified separately for each component and recorded as part of the
public action contract.

## First-order integration

For a rate $k$ held during one numerical step of duration $\Delta t$, the
selected fraction is

$$
f(k, \Delta t) = 1 - \exp(-k\Delta t).
$$

Consequently, integration over a fixed physical duration $T$ gives the same
total first-order response $1 - \exp(-kT)$ independent of the subdivision of
$T$ into numerical steps.

## Trade

Trade is a donor-specific first-order transfer, evaluated after maintenance:

$$
C_{\mathrm{plant,out}} = C_{\mathrm{plant,available}}
f(k_{\mathrm{trade,plant}}, \Delta t),
$$

$$
P_{\mathrm{fungus,out}} = P_{\mathrm{fungus,available}}
f(k_{\mathrm{trade,fungus}}, \Delta t).
$$

The existing bilateral rule remains: if either organism is non-operational
after maintenance, neither proposed transfer leaves its donor. Each accepted
transfer is subtracted once from its donor and credited once to its recipient.

Incoming trade is credited after growth and reproduction for the current
numerical substep. It therefore participates in the held Rate action from the
next numerical substep, rather than being allocated twice in the substep in
which it arrives.

## Competing growth, reproduction, and storage rates

Let

$$
K = k_g + k_r + k_s.
$$

For either current free pool $Q$ (carbon or phosphorus), the amount selected
for outcome $i \in \{g, r, s\}$ during one numerical step is

$$
Q_i =
\begin{cases}
Q\,(1 - \exp(-K\Delta t))\dfrac{k_i}{K}, & K > 0, \\
0, & K = 0.
\end{cases}
$$

Growth uses the C/P-selected amounts under the existing essential-resource
stoichiometry: realised biomass growth is limited by the scarcer allocated
resource, and any selected but unneeded amount remains in its free pool.
Reproduction is exported and recorded using its selected C and P amounts.
Storage is a selected retention outcome: its amount is not removed from the
free pool. It therefore competes with growth and reproduction without a new
state variable.

New photosynthate and direct soil-phosphate uptake are likewise credited after
allocation and become eligible under the held rates on the next numerical
substep.

## Non-negativity and conservation

The transaction is structurally safe:

- $0 \leq 1 - \exp(-K\Delta t) < 1$ for finite non-negative $K$.
- The competing-rate shares sum to one when $K > 0$, so total selected outflow
  cannot exceed the current pool.
- Storage leaves its selected amount in the pool; growth removes only the
  stoichiometrically realised amount; reproduction is explicitly recorded as
  an export.
- Trade is equal and opposite between donor and recipient.
- Maintenance, mortality, reproduction exports, biomass incorporation, and
  soil uptake remain explicit terms in the existing carbon and phosphorus
  accounting identities.

Thus free pools remain non-negative without clipping, while conservation is
preserved by transaction construction rather than post-hoc correction.

## Policy and numerical timesteps

The policy-interval wrapper holds the Rate action constant. The numerical
environment applies the equations above over its own $\Delta t$. This keeps
the physiological programme independent of numerical refinement at a fixed
policy decision interval and makes newly acquired resources eligible on the
next numerical substep, not according to policy-decision frequency.
