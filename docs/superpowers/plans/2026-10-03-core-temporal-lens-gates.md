# Core Temporal Lens Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the smallest CPU-runnable system that can falsify the claim that query-gated heterogeneous temporal states provide a useful compact memory mechanism when the current query determines which temporal scale matters.

**Architecture:** A deterministic synthetic benchmark feeds the same event stream into fixed-size recurrent baselines, an ungated multiscale bank, and a query-gated temporal-lens bank. All fixed-state models expose the same `forward(sequence, query)` contract and report resident-state size separately from parameter count. A tiny causal transformer receives the same sequence plus a final query token as an explicit-history control; a simple selective-state recurrence is the SSM-like control. Delayed loops, eigenmodes, and PAC are deliberately excluded from this first plan and will receive their own follow-on plan only after the core gates are frozen.

**Tech Stack:** Python 3.11+, PyTorch 2.x, NumPy 1.26+, pytest 8+, standard-library JSON/argparse/dataclasses only beyond those dependencies.

**Spec:** `docs/superpowers/specs/2026-10-03-multiple-temporal-lenses-design.md`

## Global Constraints

- CPU-runnable canonical gates; no GPU-only dependency.
- Primary task uses fixed resident state for recurrent/lens/SSM controls; transformer explicit-history memory is reported separately rather than called equal-budget.
- The primary comparison is query-gated multi-lens vs ungated multi-lens and matched-resident-state single recurrence.
- Benchmark generation, metric definitions, and held-out seeds are frozen before final held-out results are inspected.
- Artificial-task success must not be presented as a biological result.
- No delayed-loop, spectral, PAC, consciousness, quantum, image/shadow, or production-KV-cache claims in this implementation stage.
- Canonical short-horizon benchmark uses anchor timescales `(2, 16, 64)` with sequence length `72`; the long-horizon stress test uses `(2, 20, 200)` with sequence length `208` only after the short gate runs.
- Primary lens time constants are the task anchors and decay factors are computed as `rho_i = exp(-1 / tau_i)`.
- Canonical content vocabulary size is `8`; there are `3` query scales; lens state dimension is `12` per lens, for exactly `36` resident scalars per example.
- Fixed-state baselines use exactly `36` resident scalars per example unless a task explicitly documents why that is impossible.
- Canonical training uses AdamW, learning rate `3e-3`, no weight decay, batch size `128`, maximum `1000` optimizer steps, early stopping on validation loss with patience `100`; the same optimizer-step ceiling applies to every learned primary/control model.
- Frozen evaluation seeds are `3001, 3002, 3003, 3004, 3005`; development uses only seeds below `3000`.

## Review Focus

- **Leakage through the present/query token:** paired trials must be byte-identical at the present observation and query while retaining different labels because of history; Task 2 adds an explicit equality test.
- **State-budget mismatch:** all fixed-state models must report exactly 36 resident scalars; Task 5 tests accounting from live model metadata rather than hard-coded table values.
- **Ungated baseline accidentally denied the query:** the ungated bank still receives the query at the output readout; only query-dependent *memory selection* is ablated; Task 4 tests this contract.
- **Transformer comparison mislabeled as equal memory:** Task 8 reports sequence-dependent explicit-history activation/KV proxy separately and never calls it a 36-scalar resident-state match.
- **Held-out contamination after tuning:** Task 6 refuses evaluation seeds `<3000` for frozen receipts and records config+seed hashes in result JSON.

---

## File map

- `pyproject.toml` — package metadata and test dependencies.
- `src/multiple_temporal_lenses/config.py` — frozen benchmark/model/training dataclasses and canonical constants.
- `src/multiple_temporal_lenses/tasks.py` — deterministic generators for query-timescale, paired-history, and later-context tasks.
- `src/multiple_temporal_lenses/lenses.py` — leaky temporal state bank and invertible raw/band coordinate transform.
- `src/multiple_temporal_lenses/models.py` — shared model output type, single-state, ungated-bank, query-gated-bank, and strong query-concat control.
- `src/multiple_temporal_lenses/baselines.py` — GRU/single-state alias if needed, selective-state recurrence, and tiny causal transformer.
- `src/multiple_temporal_lenses/metrics.py` — accuracy/loss, lens entropy, parameter count, resident-state count, explicit-history memory proxy.
- `src/multiple_temporal_lenses/train.py` — deterministic training/evaluation loop and receipt writer.
- `experiments/query_chooses_timescale.py` — Gates 0–1 canonical run.
- `experiments/same_present_history.py` — Gate 2 paired-history run.
- `experiments/later_context_bends_meaning.py` — Gate 3 reinterpretation run.
- `experiments/explicit_history_controls.py` — Gate 4 selective-state and transformer controls plus long-horizon stress.
- `tests/test_config.py`, `tests/test_tasks.py`, `tests/test_lenses.py`, `tests/test_models.py`, `tests/test_metrics.py`, `tests/test_train.py` — TDD coverage.
- `results/README.md` plus frozen JSON receipts — measured outcomes only after execution.
- `README.md` — research question, equations, comparison table, claim boundary, and measured results.

### Task 1: Scaffold the package and freeze canonical configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/multiple_temporal_lenses/__init__.py`
- Create: `src/multiple_temporal_lenses/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `BenchmarkConfig`, `ModelConfig`, `TrainConfig`, `canonical_short_config()`, `canonical_long_config()`.
- `BenchmarkConfig` fields: `sequence_length: int`, `anchor_lags: tuple[int, int, int]`, `vocab_size: int`, `state_dim: int`, `num_scales: int`, `eval_seeds: tuple[int, ...]`.
- `ModelConfig` fields: `state_dim: int`, `num_lenses: int`, `coordinate_mode: Literal["raw", "band"]`.
- `TrainConfig` fields: `lr: float`, `batch_size: int`, `max_steps: int`, `patience: int`.

- [ ] **Step 1: Write failing configuration tests** asserting short anchors `(2,16,64)`, long anchors `(2,20,200)`, state budget `3*12 == 36`, eval seeds `3001..3005`, and every lag `< sequence_length`.
- [ ] **Step 2: Run** `pytest tests/test_config.py -v`; expect import/config failures.
- [ ] **Step 3: Implement the dataclasses and canonical constructors** with the exact Global Constraints above; validate strictly increasing positive lags and `max(anchor_lags) < sequence_length`.
- [ ] **Step 4: Run** `pytest tests/test_config.py -v`; expect PASS.
- [ ] **Step 5: Commit** with `feat: freeze core temporal lens configuration`.

### Task 2: Build deterministic synthetic tasks with leakage-proof paired fixtures

**Files:**
- Create: `src/multiple_temporal_lenses/tasks.py`
- Create: `tests/test_tasks.py`

**Interfaces:**
- Consumes: `BenchmarkConfig`.
- Produces: `SequenceBatch(sequence, query, target, metadata)` where `sequence: FloatTensor[B,T,V]`, `query: FloatTensor[B,3]`, `target: LongTensor[B]`.
- Produces: `make_query_timescale_batch(config, batch_size, seed, split) -> SequenceBatch`.
- Produces: `make_same_present_pairs(config, pair_count, seed) -> tuple[SequenceBatch, SequenceBatch]`.
- Produces: `LaterContextBatch(prefix_sequence, full_sequence, neutral_target, resolved_target, context)` and `make_later_context_batch(...)`.

- [ ] **Step 1: Write failing task tests** proving one content event is placed at each scale anchor, the one-hot query chooses one scale, the label equals that scale's content, generation is seed-reproducible, and held-out generation uses lag values not seen by the development split.
- [ ] **Step 2: Add the leakage test**: for every paired-history item assert `pair_a.sequence[:, -1] == pair_b.sequence[:, -1]`, `pair_a.query == pair_b.query`, and `pair_a.target != pair_b.target`; also assert at least one historical timestep differs.
- [ ] **Step 3: Run** `pytest tests/test_tasks.py -v`; expect FAIL.
- [ ] **Step 4: Implement deterministic generators.** Short development lags are sampled within scale buckets `{2,3}`, `{12,16}`, `{48,56}`; frozen held-out lags are `{4}`, `{20}`, `{64}`. Each example contains one value event in each scale bucket and zero present sensory input at the final query step. The long stress version uses exact anchors `(2,20,200)` without tuning.
- [ ] **Step 5: Implement later-context examples** where an earlier ambiguous item encodes two candidate values, a later binary context selects which candidate becomes correct, and the prefix target is the 50/50 soft distribution over both candidates.
- [ ] **Step 6: Run** `pytest tests/test_tasks.py -v`; expect PASS.
- [ ] **Step 7: Commit** with `feat: add deterministic temporal lens benchmark tasks`.

### Task 3: Implement the temporal-state mechanics and invertible coordinates

**Files:**
- Create: `src/multiple_temporal_lenses/lenses.py`
- Create: `tests/test_lenses.py`

**Interfaces:**
- Consumes: sequence tensors shaped `[B,T,D]`.
- Produces: `decays_from_timescales(timescales: Sequence[float]) -> Tensor[K]`.
- Produces: `LeakyLensBank(state_dim: int, timescales: Sequence[float])` with `run(encoded_sequence) -> Tensor[B,K,state_dim]` and `resident_state_scalars -> int`.
- Produces: `to_coordinates(states, mode: Literal["raw","band"]) -> Tensor` and `from_band_coordinates(bands) -> Tensor`.

- [ ] **Step 1: Write failing recurrence tests** against hand-computed two-step updates and the exact rule `z_t = rho*z_{t-1} + (1-rho)*x_t`.
- [ ] **Step 2: Add limiting/ordering tests**: `rho=0` tracks the current encoded event exactly; larger timescale retains a single impulse longer than a smaller one; 3x12 reports 36 resident scalars.
- [ ] **Step 3: Add coordinate tests** asserting `from_band_coordinates(to_coordinates(z,"band")) == z` within `1e-6` and raw mode is identity.
- [ ] **Step 4: Run** `pytest tests/test_lenses.py -v`; expect FAIL.
- [ ] **Step 5: Implement minimal lens mechanics** with no learned recurrence parameters; only the input encoder in later models is learned.
- [ ] **Step 6: Run** `pytest tests/test_lenses.py -v`; expect PASS.
- [ ] **Step 7: Commit** with `feat: add multiscale resident lens states`.

### Task 4: Implement primary models and isolate query-dependent lens selection

**Files:**
- Create: `src/multiple_temporal_lenses/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: `sequence: Tensor[B,T,V]`, `query: Tensor[B,3]`.
- Produces: `ModelOutput(logits: Tensor[B,V], diagnostics: dict[str, Tensor])`.
- Produces models sharing `forward(sequence, query) -> ModelOutput` and `resident_state_scalars: int`:
  - `SingleStateModel(total_state_dim=36)`.
  - `UngatedLensModel(num_lenses=3, state_dim=12)`.
  - `QueryGatedLensModel(num_lenses=3, state_dim=12)`.
  - `QueryConcatLensControl(num_lenses=3, state_dim=12)`.
- `QueryGatedLensModel` diagnostics must expose `lens_weights: Tensor[B,3]` summing to 1.

- [ ] **Step 1: Write failing shape/state-budget tests** for all four models and verify identical input sequences give identical underlying lens states in gated and ungated lens models when their encoders are copied.
- [ ] **Step 2: Add the critical ungated-query test**: changing `query` while holding sequence fixed must be allowed to change ungated logits, but must not change its history-mixture weights; changing query in the gated model may change `lens_weights`.
- [ ] **Step 3: Add the query-concat control test** proving it sees all three lens states plus query but has no softmax lens-selection variable; this is the strong control against merely rediscovering generic multiplicative/query-state interaction.
- [ ] **Step 4: Run** `pytest tests/test_models.py -v`; expect FAIL.
- [ ] **Step 5: Implement the four models.** The gated model uses `q=Wq(query)`, per-lens learned key/value transforms, scaled dot-product scores over the three resident lens states, softmax over lenses, then a linear classifier. Ungated uses a learned query-independent mixture plus the same query embedding at the classifier. Query-concat uses all lens states and query in a parameter-matched MLP as closely as integer widths allow.
- [ ] **Step 6: Run** `pytest tests/test_models.py -v`; expect PASS.
- [ ] **Step 7: Commit** with `feat: add query gated temporal lens models`.

### Task 5: Add selective-state and explicit-history baselines plus honest budget accounting

**Files:**
- Create: `src/multiple_temporal_lenses/baselines.py`
- Create: `src/multiple_temporal_lenses/metrics.py`
- Create: `tests/test_metrics.py`
- Extend: `tests/test_models.py`

**Interfaces:**
- Produces: `SelectiveStateBaseline(state_dim=36)` implementing input-dependent diagonal retention/update and final-query readout.
- Produces: `TinyCausalTransformer(d_model=12, nhead=3, num_layers=1, ff_dim=24)`; append the query as a final causal token and classify from that position.
- Produces: `count_parameters(model) -> int`, `resident_state_scalars(model) -> int | None`, `explicit_history_scalar_proxy(model, sequence_length) -> int | None`, `lens_entropy(weights) -> Tensor`.

- [ ] **Step 1: Write failing tests** asserting `SelectiveStateBaseline.resident_state_scalars == 36`, all fixed-state models report 36, and transformer resident state returns `None` while its explicit-history proxy strictly grows with sequence length.
- [ ] **Step 2: Add metric tests** for zero-entropy one-hot lens selection, `log(3)` entropy for uniform selection, and parameter counting against a tiny known linear module.
- [ ] **Step 3: Run** `pytest tests/test_models.py tests/test_metrics.py -v`; expect FAIL.
- [ ] **Step 4: Implement the selective recurrence** `a_t=sigmoid(W_a x_t+b)`, candidate `u_t=tanh(W_u x_t)`, state `s_t=a_t*s_{t-1}+(1-a_t)*u_t`; query enters only the final readout, matching the lens-task information contract.
- [ ] **Step 5: Implement the causal transformer** with an explicit triangular mask and a dedicated query token formed from the same query vector used by fixed-state models.
- [ ] **Step 6: Implement budget/entropy metrics** from live tensor shapes/model attributes, not table constants.
- [ ] **Step 7: Run** the two test files; expect PASS.
- [ ] **Step 8: Commit** with `feat: add selective state and transformer controls`.

### Task 6: Build deterministic training/evaluation and frozen receipt machinery

**Files:**
- Create: `src/multiple_temporal_lenses/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Produces: `seed_everything(seed: int) -> None`.
- Produces: `fit_classifier(model, batch_factory, train_config, train_seed, validation_seed) -> TrainResult`.
- Produces: `evaluate_classifier(model, batch_factory, seeds: Sequence[int]) -> EvalResult`.
- Produces: `write_receipt(path, *, experiment, config, metrics, seeds, model_metadata) -> None`.

- [ ] **Step 1: Write failing determinism tests**: two 20-step smoke trainings with identical seed/config produce identical losses/logits; changing seed changes at least one trained parameter.
- [ ] **Step 2: Add frozen-seed protection tests**: `write_receipt(... seeds=[2999])` is allowed only for development-tagged receipts; final/frozen receipt APIs reject any seed `<3000`; frozen evaluation must use exactly `3001..3005` unless explicitly named `stress`.
- [ ] **Step 3: Add early-stopping test** on a constant-loss dummy model/batch.
- [ ] **Step 4: Run** `pytest tests/test_train.py -v`; expect FAIL.
- [ ] **Step 5: Implement the shared trainer/evaluator** with cross-entropy for query-timescale and paired-history tasks, plus soft-target cross-entropy for the ambiguous prefix in later-context.
- [ ] **Step 6: Implement JSON receipt writing** including timestamp, Python/torch versions, full frozen config, model class, parameter count, resident-state count, explicit-history proxy when applicable, and seeds.
- [ ] **Step 7: Run** `pytest tests/test_train.py -v`; expect PASS.
- [ ] **Step 8: Run full suite** `pytest -q`; expect PASS.
- [ ] **Step 9: Commit** with `feat: add reproducible training and result receipts`.

### Task 7: Execute Gate 0 and Gate 1 — Query Chooses Timescale

**Files:**
- Create: `experiments/query_chooses_timescale.py`
- Create after run: `results/gate0_state_mechanics.json`
- Create after run: `results/gate1_query_timescale.json`
- Create: `results/README.md`

**Interfaces:**
- Consumes all core model/task/training APIs.
- CLI: `python experiments/query_chooses_timescale.py --mode dev|frozen --coordinate raw|band`.

- [ ] **Step 1: Add a CLI smoke test** in `tests/test_train.py` or a small `tests/test_experiments.py` that imports the experiment and builds every primary model without training.
- [ ] **Step 2: Implement Gate 0 receipt** measuring impulse-retention ordering, exact 36-scalar budget, and raw↔band reconstruction error.
- [ ] **Step 3: Run development training only** for `SingleStateModel`, `UngatedLensModel`, `QueryGatedLensModel`, and `QueryConcatLensControl`; adjust only training stability settings permitted by Global Constraints, never held-out seeds/task rules.
- [ ] **Step 4: Freeze the selected development settings in config/source before held-out evaluation.**
- [ ] **Step 5: Run frozen seeds `3001..3005` once** and write Gate 1 metrics: mean/per-seed accuracy, cross-entropy, parameter count, state budget, lens entropy by query scale, and pairwise deltas.
- [ ] **Step 6: Classify without spin:** `PASS_QUERY_GATING` only if gated beats both ungated and single-state on mean frozen accuracy and wins against each on at least 4/5 seeds; otherwise record `NARROWED` or `FAIL_PRIMARY` with exact reason. Query-concat is reported as a strong control, not part of the pass criterion.
- [ ] **Step 7: Run** `pytest -q`; expect PASS.
- [ ] **Step 8: Commit** with `exp: freeze query chooses timescale gate`.

### Task 8: Execute Gate 2 — Same Present, Different History

**Files:**
- Create: `experiments/same_present_history.py`
- Create after run: `results/gate2_same_present_history.json`

**Interfaces:**
- CLI: `python experiments/same_present_history.py --mode frozen`.

- [ ] **Step 1: Reuse the trained/frozen model configuration from Gate 1**; do not change architecture after seeing paired-history outcomes.
- [ ] **Step 2: Evaluate paired examples** and record paired classification accuracy, logit separation between histories, hidden/lens-state separation, and lens-selection differences.
- [ ] **Step 3: Add matched shuffled-history control** where history/label pairing is permuted while present/query remain unchanged.
- [ ] **Step 4: Gate criterion:** `PASS_HISTORY_DEPENDENCE` requires gated-model paired accuracy above 0.75 and at least 0.15 absolute above shuffled-history accuracy; otherwise narrow/fail explicitly.
- [ ] **Step 5: Run** `pytest -q`; expect PASS.
- [ ] **Step 6: Commit** with `exp: add same present different history gate`.

### Task 9: Execute Gate 3 — Later Context Bends Earlier Meaning

**Files:**
- Create: `experiments/later_context_bends_meaning.py`
- Create after run: `results/gate3_later_context.json`

**Interfaces:**
- CLI: `python experiments/later_context_bends_meaning.py --mode dev|frozen`.

- [ ] **Step 1: Add/extend tests** verifying prefix and full sequence share exactly the same ambiguous historical event and differ only by the later context event.
- [ ] **Step 2: Train with two supervised probe points:** prefix target is `[0.5,0.5]` over the two candidate values; final target is one-hot after context.
- [ ] **Step 3: Frozen evaluation records** probability mass on both candidates at prefix, probability shift toward context-consistent candidate at final time, final accuracy, and comparison across primary models.
- [ ] **Step 4: Gate criterion:** `PASS_REINTERPRETATION` requires mean gated-model context-consistent probability to increase by at least `0.25` from prefix to final and final accuracy to exceed `0.80`; compare but do not require superiority over query-concat.
- [ ] **Step 5: Run** `pytest -q`; expect PASS.
- [ ] **Step 6: Commit** with `exp: add later context reinterpretation gate`.

### Task 10: Execute Gate 4 — selective-state and transformer controls, then long-horizon stress

**Files:**
- Create: `experiments/explicit_history_controls.py`
- Create after run: `results/gate4_controls.json`
- Create after run: `results/gate4_long_horizon_stress.json`

**Interfaces:**
- CLI: `python experiments/explicit_history_controls.py --benchmark short|long --mode dev|frozen`.

- [ ] **Step 1: Train/evaluate `SelectiveStateBaseline` and `TinyCausalTransformer`** under the same optimizer-step ceiling and data splits as Gate 1.
- [ ] **Step 2: Report fairness metadata**: fixed-state resident scalars, parameter counts, transformer explicit-history scalar proxy at T=72, and never label transformer memory as matched.
- [ ] **Step 3: Compare the gated model against selective-state baseline**; if selective state matches/exceeds it within one standard error, classify the architecture-level novelty claim as `SELECTIVE_STATE_EXPLAINS_RESULT` even if Gate 1 passed.
- [ ] **Step 4: After short results are frozen, run the preregistered long stress task `(2,20,200), T=208`** with no architecture/hyperparameter search beyond extending sequence length/task config.
- [ ] **Step 5: Write long-stress receipt** including accuracy by lag scale and transformer memory growth from T=72 to T=208.
- [ ] **Step 6: Run** `pytest -q`; expect PASS.
- [ ] **Step 7: Commit** with `exp: add explicit history and selective state controls`.

### Task 11: Publish the measured claim boundary and core paper-style README

**Files:**
- Create/Modify: `README.md`
- Modify: `results/README.md`
- Modify if needed for factual corrections only: `docs/superpowers/specs/2026-10-03-multiple-temporal-lenses-design.md`

**Interfaces:**
- Consumes only committed frozen receipts; README numbers must be generated/copied from those receipts.

- [ ] **Step 1: Write README sections**: question; minimal equations; transformer vs temporal-lens comparison; relationship to selective SSMs; biological motivation with citations; benchmark design; exact pass/narrow/fail criteria; measured results; limitations; next experiment.
- [ ] **Step 2: Add a results table sourced only from JSON receipts** and clearly separate `measured in this repo`, `established external neuroscience`, and `analogy/hypothesis`.
- [ ] **Step 3: State the strongest surviving conclusion mechanically** according to receipts, including negative or narrowing outcomes without changing gate criteria after the fact.
- [ ] **Step 4: Add follow-on note:** delayed recurrent loops, spectral eigenmodes, and PAC require a new implementation plan; do not sneak them into this branch.
- [ ] **Step 5: Run** `pytest -q` and all four experiment CLIs in receipt-reading/dry mode; expect success.
- [ ] **Step 6: Commit** with `docs: publish core temporal lens findings`.

## Plan self-review

- **Spec coverage:** This plan implements the spec's immediate objective plus Gates 0–4. Gates 5–6 (delayed loops and spectral/PAC) are intentionally split into a later plan because they form a separable dynamical subsystem and the spec explicitly says to add them only after the core gate is frozen.
- **Primary interaction isolated:** The ungated model still receives the query, and the query-concat control tests whether generic query/state interaction can explain any gated-model advantage.
- **State fairness:** All fixed-state primary/SSM controls expose 36 resident scalars; parameter count is reported separately. Transformer memory is explicitly non-fixed and sequence dependent.
- **Leakage:** Present observation/query equality is tested for paired-history examples.
- **Held-out discipline:** final seeds are mechanically protected and frozen settings are committed before use.
- **Biological boundary:** no artificial result is allowed to establish a cortical mechanism; delayed loops/PAC remain deferred.
