# Temporal Residual Correction Plan Addendum

Parent plan: `docs/superpowers/plans/2026-10-03-core-temporal-lens-gates.md`
Spec amendment: `docs/superpowers/specs/2026-10-03-temporal-residual-amendment.md`

This addendum supersedes the coordinate/readout portions of Tasks 3, 4, 5, and 7 before any learned Gate-1 result is inspected.

## Completed corrective slice

### A. Correct Task-3 coordinate orientation

- Keep raw physical states ordered fast→slow.
- Define `band`/temporal-residual coordinates as `(z_fast-z_medium, z_medium-z_slow, z_slow)` for three lenses.
- Reconstruct raw states by reverse cumulative sum.
- Test exact orientation and invertibility.

### B. Add analytical impulse-kernel diagnostic

- Add `impulse_kernel(decays, lags, mode)` with raw formula `(1-rho)*rho**L`.
- For `mode="band"`, apply adjacent fast-minus-slower residuals plus slowest tail.
- Add a CLI diagnostic covering `rho=(0.5,0.95,0.995)`, lags `(0,2,20,200)`, and canonical `tau=(2,20,200)`.
- Freeze the analytical receipt at `results/gate0_impulse_kernel.json` before learned models exist.

## Revised Task 4 — fair reader comparison

Implement one shared encoder/lens-state producer, then four multiscale readers:

1. `RawGatedLensModel` — raw low-passes + query-softmax routing.
2. `ResidualGatedLensModel` — temporal residuals + query-softmax routing; primary routing model.
3. `RawConcatLensModel` — MLP over `[all raw states, query]`.
4. `ResidualConcatLensModel` — MLP over `[all residual states, query]`; primary strong control.

Keep `SingleStateModel(total_state_dim=36)` as the fixed-state recurrence control.

Required tests before implementation:

- all fixed-state models report 36 resident scalars;
- copying the shared encoder gives bit-identical raw lens states across all multiscale variants;
- raw and residual variants differ only by the invertible coordinate transform before the reader;
- concat readers demonstrably receive the query;
- gated weights sum to 1 and only gated models expose `lens_weights`;
- parameter counts for gated vs concat readers are reported and matched as closely as practical.

## Revised Task 5 — promote selective state

Implement the 36-state input-selective recurrence before any architecture-level novelty interpretation. Treat it as a first-class control, not a late appendix. Keep the tiny causal transformer as the explicit-history comparison.

## Revised Gate 1 classification

Do not use the old single `PASS_QUERY_GATING` criterion. Report the factorial comparisons separately:

- residual vs raw within gated reader;
- residual vs raw within concat reader;
- gated vs concat within residual coordinates;
- each against single-state recurrence;
- each against selective-state recurrence when available.

Possible labels:

- `RESIDUAL_COORDINATES_HELP`
- `ROUTING_ADDS_VALUE`
- `GENERIC_QUERY_READOUT_SUFFICIENT`
- `SELECTIVE_STATE_EXPLAINS_RESULT`
- `NO_MULTILENS_ADVANTAGE`

A model may receive more than one descriptive label; the receipt must include the underlying metrics rather than compressing the outcome to one winner label.

## Stop condition for this addendum

The corrective slice ends after the residual transform, analytical diagnostic, tests, docs, and analytical receipt are committed. The learned readers are the next task and must not be implemented in the same corrective commit.
