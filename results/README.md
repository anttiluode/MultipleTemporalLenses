# Core temporal-lens receipts

Gate 0 and Gate 1 use the corrected temporal-residual amendment. Gate 1 is a 2×2 factorial over coordinate basis (`raw`, `temporal residual`) and reader (`query-gated softmax`, `query-conditioned concat`), plus a 36-scalar single-state GRU.

## Corrected Gate 1

| model | development accuracy | held-out accuracy |
|---|---:|---:|
| single-state GRU | 0.4543 | 0.4488 |
| raw + gated | 0.5035 | 0.4430 |
| residual + gated | 0.3965 | 0.4211 |
| raw + concat | 0.6063 | **0.5258** |
| residual + concat | **0.6594** | 0.5051 |

The development result suggested a residual-coordinate advantage for the generic concat reader, but it did **not** survive held-out evaluation: residual concat lost to raw concat on all five held-out seeds. Residual gating also lost to raw gating on all five held-out seeds. Under the preregistered rule (higher mean plus at least 4/5 seed wins), Gate 1 is `NO_CLEAR_PRIMARY_ADVANTAGE` for residual coordinates or routing.

The strongest measured result at this gate is narrower: the query-conditioned concat reader is the best of the fixed multiscale readers on held-out data, with raw coordinates slightly ahead of residual coordinates. Query-gated routing is not supported by this benchmark.

The freeze marker was committed before the corrected held-out run and no settings were changed after development. An abandoned pre-correction local execution had previously touched seed identities 3001–3005 under the superseded design; corrected-model settings were not tuned from those results, so this run is best described as the preregistered canonical held-out replication rather than claiming those seeds were never executed anywhere.
