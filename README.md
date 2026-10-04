# Multiple Temporal Lenses

**Status:** core Gates 0–4 executed and frozen. The original query-routing claim did **not** survive the preregistered controls. The repository records the narrower result rather than promoting the failed hypothesis.

## Question

Can a small bank of fixed-size temporal states act as a compact alternative to explicit history when the current query determines which age of the past matters?

The minimal lens bank uses leaky states:

```text
z_t^(i) = rho_i z_(t-1)^(i) + (1-rho_i) x_t
```

Two coordinate systems are compared:

```text
raw:       (z_fast, z_medium, z_slow)
residual:  (z_fast-z_medium, z_medium-z_slow, z_slow)
```

The residual transform is exactly invertible. It changes the coordinates presented to the reader, not the information stored by the bank.

## Why the residual correction mattered

A slow leaky state is a low-pass, not a lag-specific bin. For an impulse at lag `L`, its raw contribution is `(1-rho) * rho^L`, so every raw lens remains biased toward recent events. `experiments/impulse_kernel_diagnostic.py` freezes the analytical correction before learned comparisons: adjacent fast-minus-slower differences plus the slowest tail. These are signed overlapping temporal bands, not rectangular age windows.

## Models

| model | history representation | resident state |
|---|---|---:|
| single-state GRU | learned recurrent state | 36 scalars |
| raw + gated | three raw leaky states + query softmax | 36 |
| residual + gated | three residual coordinates + query softmax | 36 |
| raw + concat | three raw leaky states + query-conditioned MLP | 36 |
| residual + concat | three residual coordinates + query-conditioned MLP | 36 |
| selective-state | input-selective diagonal recurrence | 36 |
| tiny causal transformer | explicit token history | sequence-growing |

The selective-state model is a **minimal Mamba-like control, not Mamba**. The transformer is **not** memory-matched: its explicit-history proxy grows from 1,752 scalars at `T=72` to 5,016 at `T=208`.

## Frozen results

### Gate 1 — Query Chooses Timescale

Held-out lags are disjoint from development lags.

| model | development | held-out |
|---|---:|---:|
| single-state GRU | 0.4543 | 0.4488 |
| raw + gated | 0.5035 | 0.4430 |
| residual + gated | 0.3965 | 0.4211 |
| raw + concat | 0.6062 | **0.5258** |
| residual + concat | **0.6594** | 0.5051 |

**Result:** `NO_CLEAR_PRIMARY_ADVANTAGE`. The development residual-coordinate advantage did not survive the frozen lag shift, and softmax routing did not beat the generic query-conditioned concat reader.

The narrow surviving observation is that the fixed multiscale bank with a generic query-conditioned readout beat the matched-state GRU and the repository's minimal selective-state control on this lag-shift task. That is a result about this benchmark and these small controls, not an architecture-level claim over modern SSMs.

### Gate 2 — Same Present, Different History

Each pair has an identical present observation and query, but the relevant historical value is changed. A matched control shuffles histories within query-scale groups while preserving present/query/labels.

| model | paired | shuffled | history effect |
|---|---:|---:|---:|
| raw + concat | **0.5250** | 0.1207 | +0.4043 |
| residual + concat | 0.5029 | 0.1193 | +0.3836 |
| single-state GRU | 0.4500 | 0.1299 | +0.3201 |
| raw + gated | 0.4449 | 0.1256 | +0.3193 |
| residual + gated | 0.4213 | 0.1252 | +0.2961 |
| selective-state | 0.4213 | 0.1252 | +0.2961 |

Primary pass criterion: residual-gated paired accuracy `>0.75` and at least `+0.15` over shuffled history.

**Result:** `FAIL_HISTORY_DEPENDENCE`. The shuffle effects are large—history is causally useful—but none of the compact readers converts it into sufficiently reliable prediction. The primary residual-gated model reaches 0.4213 vs 0.1252 shuffled; raw concat is best at 0.5250, still far below the absolute bar.

### Gate 3 — Later Context Bends Earlier Meaning

An earlier observation represents two candidate values with a 50/50 target. A later context bit selects which candidate should become correct. Training supervises both the ambiguous prefix and resolved final probe.

| model | final accuracy | context-consistent probability shift |
|---|---:|---:|
| raw + concat | **0.4996** | +0.0001 |
| single-state GRU | 0.4895 | +0.0006 |
| residual + concat | 0.4492 | +0.0000 |
| residual + gated | 0.4359 | +0.0018 |
| raw + gated | 0.4137 | +0.0030 |
| selective-state | 0.1484 | -0.0000 |

Primary criterion: probability shift `>=0.25` and final accuracy `>0.80`.

**Result:** `FAIL_REINTERPRETATION`. The primary residual-gated reader shifts probability by only +0.0018 and reaches 0.4359 final accuracy. No compact model shows the intended reinterpretation trajectory.

### Gate 4 — Selective-state and explicit-history controls

On the short held-out lag-shift task, the tiny transformer reaches only **0.4250** despite an extremely low development validation loss. Explicit access to tokens therefore did not guarantee lag-shift generalization for this tiny model.

Residual-gated and the minimal selective-state recurrence both score 0.4211 on the short held-out task. Under the preregistered one-standard-error rule this is `SELECTIVE_STATE_EXPLAINS_RESULT` for the **routing-specific** interpretation. It does not explain away raw + concat's higher Gate-1 score.

The preregistered long stress test uses exact anchors `(2,20,200)` with `T=208` and no architecture/hyperparameter search:

| model | overall | lag 2 | lag 20 | lag 200 |
|---|---:|---:|---:|---:|
| tiny transformer | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| residual + gated | 0.5461 | 1.0000 | 0.4042 | 0.2287 |
| raw + concat | 0.4496 | 1.0000 | 0.2104 | 0.1317 |
| selective-state | 0.4211 | 1.0000 | 0.1259 | 0.1304 |

The explicit-history transformer solves the exact-anchor stress task. The compact 36-scalar models degrade sharply with age. Residual-gated is the strongest compact model in this stress run, but lag-200 accuracy is only 0.2287; this does not rescue the failed short-horizon routing gate.

## What survived

1. **Compact multiscale state carries usable history.** Gate 2's large history-vs-shuffle gaps show that the stored state is not decorative.
2. **Reader geometry matters.** Generic query-conditioned concat is stronger than the tested softmax routing on the main short benchmark.
3. **The original “query chooses a temporal lens” mechanism is not supported.** It loses the critical comparisons and fails reinterpretation.
4. **Compression has a visible price.** Explicit history solves the exact-anchor long stress while the 36-scalar models deteriorate with temporal distance.

A next architecture needs a stronger way to separate or address old information than a convex softmax over overlapping low-pass states. Signed temporal residuals make subtraction possible, but here they were not sufficient.

## Transformer comparison

| Transformer | Temporal lens bank |
|---|---|
| retains explicit token/item representations | retains a fixed number of evolving states |
| query addresses positions/content | query/readout addresses temporal summaries |
| strong random access | strong compression bias |
| history memory grows with sequence length | resident state stays fixed |
| old items remain distinct | old events superpose and decay |

## Biological motivation — not biological evidence

The abstraction was motivated by real observations, but none of these synthetic results establishes a cortical mechanism:

- extended cortico-thalamo-cortical loop structures — Shepherd & Yamawaki (2021), https://doi.org/10.1038/s41583-021-00459-3
- heterogeneous intrinsic cortical timescales — Murray et al. (2014), https://doi.org/10.1038/nn.3862
- nonlinear apical-dendritic recruitment by corticocortical feedback — Fişek et al. (2023), https://doi.org/10.1038/s41586-023-06007-6

Relevant ML comparisons: Universal Transformers (Dehghani et al., 2018, https://arxiv.org/abs/1807.03819) and Mamba/selective state spaces (Gu & Dao, 2023, https://arxiv.org/abs/2312.00752).

This repository does **not** claim cortex is a transformer, apical dendrites are Q/K/V, loop length determines oscillation frequency, PAC is attention, or these experiments explain consciousness.

## Reproduce

```bash
python -m pip install -e '.[test]'
pytest -q

python experiments/impulse_kernel_diagnostic.py
python experiments/query_chooses_timescale.py --dry
python experiments/same_present_history.py --dry
python experiments/later_context_bends_meaning.py --dry
python experiments/explicit_history_controls.py --dry
```

Frozen JSON receipts live in `results/`. Heavy canonical training is CPU-runnable and the later Gate-4 runner supports deterministic chunked checkpoints so interrupted sessions can resume without changing the training stream.

## Next experiment

Do **not** jump directly to PAC. The next clean question is whether explicit delayed/recurrent temporal states create more separable, query-addressable modes than overlapping leaky summaries. That requires a new preregistered delayed-loop/spectral plan; oscillatory/PAC analysis should remain downstream of that result.

See `docs/superpowers/` for the approved design, residual correction, and implementation plans.
