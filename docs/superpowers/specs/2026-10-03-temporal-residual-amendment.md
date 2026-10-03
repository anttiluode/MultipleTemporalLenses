# Multiple Temporal Lenses — temporal-residual amendment

Date: 2026-10-03
Status: approved correction before learned-model comparisons
Parent spec: `docs/superpowers/specs/2026-10-03-multiple-temporal-lenses-design.md`

This amendment is binding for the first learned-model gate and supersedes the parent spec where the two differ.

## Why this amendment exists

A raw leaky state is a low-pass summary, not a lag-specific bin. For one event at lag `L`, a lens with decay `rho` contributes

\[
w_\rho(L)=(1-\rho)\rho^L.
\]

Every such kernel is recent-biased. Choosing the slowest raw low-pass therefore does not isolate old evidence; it retains old evidence while still superposing newer evidence. A positive softmax over raw low-passes cannot itself subtract those overlapping recent contributions.

The original primary architecture would therefore confound **memory geometry** with **query routing** on the benchmark designed to test it.

## Corrected state/read geometry

The physical resident states remain three heterogeneous low-passes ordered fast to slow:

\[
z_f,\;z_m,\;z_s.
\]

The primary query-visible coordinates are now the invertible temporal residual basis

\[
b_f=z_f-z_m,\qquad b_m=z_m-z_s,\qquad b_s=z_s.
\]

For `K` ordered low-passes, this generalizes to adjacent fast-minus-slower differences plus the slowest tail. The transform is exactly invertible by reverse cumulative sum, so raw and residual coordinates contain identical information.

These residuals are signed, overlapping difference-of-exponential kernels. They may be called **temporal residual** or **band-pass-like** coordinates, but they are not perfectly disjoint rectangular age bins.

## Gate-0 analytical sanity check

Before any learned comparison, report analytical impulse weights for raw and residual coordinates. The diagnostic must establish:

- raw leaky kernels follow `(1-rho)*rho**L`;
- every raw lens remains recent-biased;
- adjacent residual kernels can change sign across lag, allowing cancellation of overlapping low-pass contributions;
- the raw↔residual transform is invertible and adds no information.

The committed diagnostic uses both the motivating example `rho=(0.5,0.95,0.995)` at lags `(0,2,20,200)` and the canonical long-timescale decays derived from `tau=(2,20,200)`.

## Corrected learned-model comparison

All multiscale learned models receive the same encoded event stream, the same three physical low-pass states, the same current query, and the same 36-scalar resident-state budget. The first comparison is a 2×2 read-geometry test:

| coordinates | reader | purpose |
|---|---|---|
| raw low-passes | query-gated softmax | asks whether routing alone can overcome overlap |
| temporal residuals | query-gated softmax | primary lens-routing hypothesis |
| raw low-passes | query-conditioned concat MLP | strong ungated/non-routing control |
| temporal residuals | query-conditioned concat MLP | asks whether residual geometry alone is sufficient |

The concat reader **must receive the query**. It is not allowed to lose merely because only the gated model sees present context.

The gated and concat variants must be fed identical resident tensors before their reader-specific operation. Parameter counts must be reported and matched as closely as integer widths permit.

## Interpretation matrix

- `residual-gated > raw-gated`: coordinate accessibility matters; routing over raw low-passes was structurally disadvantaged.
- `residual-concat ≈ residual-gated`: temporal decomposition helps, but explicit softmax routing is unnecessary.
- `residual-gated > residual-concat` under matched conditions: evidence specifically for query-directed routing among temporal residuals.
- raw and residual readers performing alike: the coordinate transform does not matter for the tested reader/task.
- matched single-state recurrence performing alike: multiple explicit temporal states are unnecessary for the tested task.

No result may be interpreted as increased memory capacity when comparing raw and residual coordinates, because the transform is invertible.

## Selective-state-space control

A selective-state recurrence is promoted to a first-class interpretation control. Input-dependent retention/update already captures the broad idea that present input changes temporal propagation. The repository may describe its control as **selective-SSM/Mamba-like**, but must not call it Mamba unless an actual Mamba implementation is run.

If the selective-state control matches or exceeds the residual-lens models, the architecture-level conclusion narrows to an interpretable decomposition of computation already available to selective state-space recurrence.

## Claim boundary

The corrected first-stage question is no longer simply whether "the present chooses a timescale." It is:

> **Does exposing fixed-size multiscale memory in explicit temporal-residual coordinates, and optionally routing among those coordinates with the current query, improve access to history compared with raw low-passes, generic query-conditioned readout, single recurrence, and selective-state recurrence?**
