# Core temporal-lens receipts

This directory is the measurement ledger for the core non-oscillatory experiment. Numbers below are copied from the frozen JSON receipts; the JSON files remain authoritative.

## Gate summary

| gate | outcome | narrow observation |
|---|---|---|
| Gate 0 — mechanics | analytical sanity check | raw leaky states are recent-biased; residual coordinates are invertible signed differences |
| Gate 1 — query chooses timescale | `NO_CLEAR_PRIMARY_ADVANTAGE` | raw + concat is best held-out fixed-state lens reader at 0.5258 |
| Gate 2 — same present, different history | `FAIL_HISTORY_DEPENDENCE` | large history-vs-shuffle gaps, best paired accuracy 0.5250 |
| Gate 3 — later context bends meaning | `FAIL_REINTERPRETATION` | no compact reader produces the required probability revision |
| Gate 4 — controls/stress | routing narrowed; explicit history wins exact-anchor stress | transformer reaches 1.0000 on `(2,20,200)` stress with sequence-growing memory |

## Gate 1 — corrected factorial

| model | development | held-out |
|---|---:|---:|
| single-state GRU | 0.4543 | 0.4488 |
| raw + gated | 0.5035 | 0.4430 |
| residual + gated | 0.3965 | 0.4211 |
| raw + concat | 0.6062 | **0.5258** |
| residual + concat | **0.6594** | 0.5051 |

The development residual-coordinate advantage did not survive frozen lag shift. Classification: `NO_CLEAR_PRIMARY_ADVANTAGE`.

## Gate 2 — history intervention

| model | paired | shuffled | delta |
|---|---:|---:|---:|
| raw + concat | **0.5250** | 0.1207 | +0.4043 |
| residual + concat | 0.5029 | 0.1193 | +0.3836 |
| single-state GRU | 0.4500 | 0.1299 | +0.3201 |
| residual + gated | 0.4213 | 0.1252 | +0.2961 |

All models fail the preregistered `>0.75` absolute paired-accuracy bar. The large shuffle deltas nevertheless show that history representations carry task-relevant information.

## Gate 3 — reinterpretation

Primary residual-gated: final accuracy 0.4359; context-consistent probability shift +0.0018. Required: final accuracy `>0.80`, shift `>=0.25`.

Best final accuracy is raw + concat at 0.4996; its probability shift is +0.0001. Gate 3 is a clear negative result.

## Gate 4 — controls and long stress

Short held-out lag shift: residual-gated 0.4211; minimal selective-state 0.4211; tiny transformer 0.4250. Routing-vs-selective classification: `SELECTIVE_STATE_EXPLAINS_RESULT`.

Long exact-anchor `(2,20,200)` stress:

| model | overall | lag 2 | lag 20 | lag 200 |
|---|---:|---:|---:|---:|
| tiny transformer | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| residual + gated | 0.5461 | 1.0000 | 0.4042 | 0.2287 |
| raw + concat | 0.4496 | 1.0000 | 0.2104 | 0.1317 |
| selective-state | 0.4211 | 1.0000 | 0.1259 | 0.1304 |

Transformer explicit-history proxy grows from 1,752 scalars (`T=72`) to 5,016 (`T=208`), 2.863×. It is not a fixed-state memory match.

## Claim boundary

Measured here: compact multiscale state contains usable history; generic query-conditioned concat is stronger than the tested softmax router on the short benchmark; the proposed routing mechanism fails its primary/reinterpretation gates; fixed-state compression degrades at older lags.

Not established here: superiority over Mamba/modern SSMs; a cortical implementation; delayed-loop eigenmodes; oscillatory computation/PAC; consciousness claims.
