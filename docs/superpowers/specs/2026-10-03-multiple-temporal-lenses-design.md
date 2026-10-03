# Multiple Temporal Lenses — research and implementation design

Date: 2026-10-03
Status: design approved in conversation; implementation not started

## 1. Purpose

This repository will test a narrow computational hypothesis inspired by recurrent cortical loops, heterogeneous neural timescales, active dendritic feedback, and transformer/state-space sequence models:

> **Query-gated states with heterogeneous temporal dynamics can provide a compact alternative to explicit history when the relevance of past information depends jointly on content and temporal scale.**

The repository is not intended to establish that cortex is literally a transformer, that apical tufts are biological Q/K/V units, that recurrent loops necessarily oscillate, or that phase-amplitude coupling (PAC) is equivalent to attention. Biological observations are hypothesis generators; the scientific core is a set of explicit computational comparisons.

The central object is a bank of resident temporal states — “temporal lenses” — whose dynamics differ and whose readout depends on the current query/state.

## 2. Biological motivation and claim boundary

Three empirical observations motivate the abstraction.

1. **Extended recurrent loops exist.** Cortico-thalamo-cortical loops are tightly interlinked with local cortical and corticocortical circuits and can form extended chains of loops. This supports studying recurrent computation over physically different routes without assuming that each route corresponds to one frequency or one cognitive function.
2. **Cortical dynamics span heterogeneous timescales.** Intrinsic spiking timescales vary across cortical areas, with shorter timescales in sensory regions and longer timescales in prefrontal/association regions. This motivates multiple state variables with different retention dynamics.
3. **Feedback can be integrated nonlinearly in apical dendrites.** In mouse visual cortex, cortico-cortical feedback can recruit branch-specific nonlinear events in apical tufts of layer-5 pyramidal neurons. This motivates keeping contextual/feedback input computationally distinct from immediate feedforward drive.

These observations do **not** establish the proposed architecture. In particular:

- a pyramidal neuron is not asserted to implement transformer attention;
- an apical tuft is not identified with a query, key, or value vector;
- transformer depth is not equated with biological time;
- cortical loops are not assumed to have frequencies given by inverse path length;
- PAC is not treated as evidence for nested loops unless alternative signal-processing explanations are controlled.

### Key references

- Shepherd & Yamawaki (2021), *Untangling the cortico-thalamo-cortical loop: cellular pieces of a knotty circuit puzzle*, Nature Reviews Neuroscience. https://doi.org/10.1038/s41583-021-00459-3
- Murray et al. (2014), *A hierarchy of intrinsic timescales across primate cortex*, Nature Neuroscience. https://doi.org/10.1038/nn.3862
- Fişek et al. (2023), *Cortico-cortical feedback engages active dendrites in visual cortex*, Nature. https://doi.org/10.1038/s41586-023-06007-6
- Dehghani et al. (2018), *Universal Transformers*. https://arxiv.org/abs/1807.03819
- Gu & Dao (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. https://arxiv.org/abs/2312.00752
- Jensen, Spaak & Park (2016/2017), *Discriminating Valid from Spurious Indices of Phase-Amplitude Coupling*, eNeuro. https://doi.org/10.1523/ENEURO.0334-16.2016

## 3. Computational abstraction

Let the present input be `x_t`, the current model state be `h_t`, and let there be `K` resident temporal states:

\[
z_t^{(1)}, z_t^{(2)}, \ldots, z_t^{(K)}.
\]

The simplest lens uses a leaky recurrence:

\[
z_{t+1}^{(i)} = \rho_i z_t^{(i)} + (1-\rho_i) B_i x_t,
\]

with distinct `rho_i`. A small `rho_i` is a fast lens; a value near 1 is a slow lens.

A lens bank by itself is only a multiscale filter. The core hypothesis requires **query-dependent access**. Define

\[
q_t = W_Q h_t,
\]

\[
k_t^{(i)} = W_K^{(i)} z_t^{(i)},\qquad
v_t^{(i)} = W_V^{(i)} z_t^{(i)}.
\]

The current state selects among temporal histories:

\[
a_i(t) = \operatorname{softmax}_i\left(\frac{q_t^\top k_t^{(i)}}{\sqrt d}\right),
\]

\[
c_t = \sum_i a_i(t) v_t^{(i)}.
\]

The next state is then

\[
h_{t+1}=h_t+F(h_t,x_t,c_t).
\]

The defining computation is therefore not merely “remember at several timescales.” It is:

> **construct several continuously evolving views of history, then let the present decide which view matters.**

## 4. Comparison with transformer attention

For a causal transformer, the current query addresses explicit stored past items:

\[
q_t=W_Q h_t,
\]

\[
c_t = \sum_{j\le t}\operatorname{softmax}_j(q_t^\top k_j)v_j.
\]

The temporal-lens architecture instead addresses resident dynamical summaries:

\[
c_t = \sum_i\operatorname{softmax}_i(q_t^\top k_i(z_t^{(i)}))v_i(z_t^{(i)}).
\]

The conceptual distinction is:

| Transformer | Multiple Temporal Lenses |
|---|---|
| Stores explicit token/item history | Stores fixed-size dynamical history states |
| Query addresses individual past positions | Query addresses temporal modes/summaries |
| Old K/V items remain individually accessible | Old events continuously dissolve into evolving state |
| Strong random access | Strong compression/integration bias |
| Time represented via position/content and causal structure | Time encoded directly in state dynamics |
| Computational depth is mostly architectural depth | Recurrent passes may turn physical/algorithmic time into depth |

This is not expected to dominate transformers at exact episodic recall. Its plausible advantage is compact resident memory when the task depends on temporal scale rather than exact retrieval of arbitrary old items.

## 5. Relationship to state-space models and previous repository work

The closest modern ML control is not only the vanilla transformer. Selective state-space models, especially Mamba, allow state transitions to depend on current input and can selectively propagate or forget information. A serious test must therefore include an SSM-like control so that the repository does not relabel established selective recurrence.

The project also directly follows the existing `TransformerToX` / `Rytmi` line. That work found that a frozen GPT-2-derived query/read mechanism used `(fast-slow, slow)` temporal coordinates much more effectively than the invertibly related `(fast, slow)` coordinates, even though they contained the same information. Later equal-budget cache controls narrowed early claims and showed the importance of attention-sink effects, while still leaving a narrower result in which temporal summaries improved a sink-preserving recent window on the tested head/contexts.

The lesson carried forward is:

> **History being present in state is not enough; its coordinate system can determine whether a current query can use it.**

`MultipleTemporalLenses` will therefore test both **state content** and **query/read geometry**, not only memory capacity.

## 6. Model ladder

Implementation should proceed from the least committed mechanism to the more biological/dynamical variants.

### Model A — single recurrent state

One fixed-size recurrent state. This asks whether ordinary recurrence is sufficient.

### Model B — ungated temporal bank

Several leaky states with different `rho_i`, concatenated or averaged through a fixed readout. This tests whether multiple timescales alone explain performance.

### Model C — query-gated temporal lenses (primary)

Several temporal states plus query-dependent addressing over lenses. This is the core hypothesis.

### Model D — recurrent/delayed temporal lenses

Replace simple leaky summaries with explicit recurrent loops or delayed state dynamics, e.g.

\[
\tau_i \dot z_i(t)=-z_i(t)+W_i\phi(z_i(t-d_i))+B_i x(t).
\]

In discrete time, use ring buffers / fixed delays so the mechanism remains auditable.

### Model E — oscillatory/PAC extension

Only after Models A–D are understood, permit recurrent parameters that produce oscillatory modes and test whether slow-state phase/gain can modulate faster activity. PAC is an optional consequence, not a prerequisite.

### Controls

- small causal transformer with explicit history;
- GRU or equivalent recurrent baseline;
- selective-SSM/Mamba-like control under matched state/parameter budget;
- fixed multiscale filter bank without query gating;
- shuffled or mismatched lens labels where appropriate;
- query ablations and same-information coordinate transforms.

## 7. Primary benchmark: Query Chooses Timescale

Construct synthetic sequences in which the same current observation can require information from different temporal distances. The current query specifies or implies which temporal scale is relevant.

Example structure:

- a cue/value pair occurs at lag 2, lag 20, or lag 200;
- distractors appear at the other lags;
- the current query determines which lag family is relevant;
- train/test splits include unseen combinations of content and lag where possible.

The benchmark should include both discrete symbolic and low-dimensional continuous versions.

### Primary scientific comparison

The predeclared interaction of interest is:

\[
\text{query-gated multi-lens} > \text{ungated multi-lens}
\]

and

\[
\text{query-gated multi-lens} > \text{single-state recurrent control},
\]

under matched state/parameter budgets, specifically on examples where the same present requires different historical scales depending on query/context.

A win by the ungated filter bank would narrow the claim to ordinary multiscale memory. A win by the SSM control would indicate that the useful mechanism is already captured by selective state-space recurrence. A large transformer advantage would quantify the cost of compressed state relative to explicit history.

## 8. Benchmark 2: Same Present, Different History

Create paired trials with identical current input and identical current query but different prior histories:

\[
H_A \ne H_B,\qquad x_t^A=x_t^B,\qquad q_t^A=q_t^B.
\]

The correct output differs between the pair.

This gate distinguishes true history dependence from present-input shortcuts. It should report paired accuracy, representation separation, and whether the selected lens changes in a history-appropriate way.

## 9. Benchmark 3: Later Context Bends Earlier Meaning

Present an ambiguous event that initially permits two interpretations. Later evidence disambiguates it. The model must revise its interpretation without replaying the complete raw history.

Measure the trajectory

\[
P(y\mid t_0) \rightarrow P(y\mid t_1) \rightarrow P(y\mid t_2)
\]

rather than only final accuracy.

The central question is whether slower resident lenses preserve enough information about the ambiguous earlier event for later context to alter its interpretation.

This benchmark operationalizes the phrase:

> **A loop is not merely feedback; a later state can become a new lens on an earlier measurement.**

## 10. Spectral and oscillatory analysis

The delayed-loop model can be linearized around a fixed point. For a linear recurrent system, eigenvalues

\[
\lambda_k=\sigma_k+i\omega_k
\]

separate retention/damping (`sigma_k`) from oscillation (`omega_k`). The repository should explicitly test the following rather than assuming them:

- whether useful lenses require oscillatory modes at all;
- whether learned/selected delays alter the useful eigenmodes;
- whether different timescale tasks recruit different modes;
- whether a slow state multiplicatively gating a fast process creates measurable PAC;
- whether apparent PAC survives controls for nonsinusoidal waveform shape and harmonics.

The project must not infer “loop period = oscillation period” from path length alone.

## 11. Metrics

Each benchmark should report more than task accuracy.

Core metrics:

- held-out task accuracy / loss;
- parameter count;
- resident-state scalar budget;
- training compute proxy and inference-step cost;
- memory as a function of sequence length;
- lens-selection entropy and per-query selection distribution;
- ablation deltas (no gating, one lens, shuffled lens, equal-information coordinate transform);
- generalization across unseen lags/timescales.

For dynamical variants:

- state autocorrelation time;
- eigenvalue / spectral radius diagnostics where defined;
- damping and oscillatory-mode estimates;
- PAC only with surrogate and waveform-shape controls.

## 12. Fair-comparison rules

1. **Equal resident-state budget** for recurrent/SSM/lens variants whenever possible.
2. **Report explicit transformer memory separately**, since KV memory grows with sequence length and is not equivalent to fixed resident state.
3. **Freeze benchmark generation and primary metrics before reading final held-out results.**
4. **Use fresh held-out seeds** after any hyperparameter selection.
5. **Distinguish mechanism gates from model-quality claims.** A synthetic mechanism win is not a language-model result.
6. **Include zero/simple baselines** where regression metrics can otherwise look impressive despite poor absolute performance.
7. **No biological conclusion from artificial-task success.**

## 13. Repository architecture

Target structure:

```text
MultipleTemporalLenses/
├── README.md
├── pyproject.toml
├── src/multiple_temporal_lenses/
│   ├── lenses.py              # leaky and coordinate-based temporal states
│   ├── gating.py              # query-to-lens addressing
│   ├── recurrent.py           # delayed/recurrent loop variants
│   ├── baselines.py           # RNN/filter/SSM/transformer controls
│   ├── tasks.py               # deterministic benchmark generators
│   ├── metrics.py             # task, memory, gating and dynamical metrics
│   └── spectral.py            # eigenmode / PAC diagnostics
├── experiments/
│   ├── query_chooses_timescale.py
│   ├── same_present_history.py
│   ├── later_context_bends_meaning.py
│   └── delayed_loop_spectrum.py
├── tests/
│   ├── test_lenses.py
│   ├── test_gating.py
│   ├── test_tasks.py
│   ├── test_budget.py
│   └── test_spectral.py
├── results/
│   └── README.md
└── docs/superpowers/specs/
```

Implementation should favor NumPy/PyTorch and CPU-runnable canonical gates. GPU-heavy language-model work is explicitly deferred until the small mechanism is understood.

## 14. Testing strategy

Use test-driven development for implementation.

Tests must include:

- exact recurrence checks for each lens;
- limiting cases (`rho=0`, `rho→1` within numeric tolerance);
- deterministic state-budget accounting;
- proof-by-construction that ungated and gated models receive identical lens states when comparing only the readout mechanism;
- same-present paired fixtures with no current-input leakage;
- benchmark reproducibility under fixed seeds;
- delayed-loop ring-buffer correctness;
- spectral checks against analytically known small matrices;
- PAC null controls where a nonsinusoidal single oscillator must not be interpreted as evidence for two interacting oscillators.

## 15. Staged gates

### Gate 0 — temporal-state mechanics

Verify that fast/medium/slow states respond with the intended retention ordering and that state budgets are exact.

### Gate 1 — query chooses timescale

Run Models A–C and core controls on the synthetic primary benchmark. Do not add oscillations.

### Gate 2 — same present, different history

Verify that the mechanism uses retained history rather than current-input shortcuts.

### Gate 3 — later context bends earlier meaning

Test iterative reinterpretation using resident state rather than raw-history replay.

### Gate 4 — explicit transformer / selective-SSM comparison

Match parameter and resident-state budgets as fairly as the architecture permits; separately report transformer KV growth.

### Gate 5 — delayed recurrent loops

Replace leaky summaries with explicit delayed loops and measure whether delayed recurrence adds anything beyond the simpler lens bank.

### Gate 6 — spectral/PAC extension

Only if Gate 5 yields useful loop dynamics, analyze oscillatory modes and controlled PAC. A negative result does not invalidate Gates 0–5.

## 16. Success, narrowing, and failure outcomes

### Strong result

Query-gated heterogeneous states beat both ungated multiscale memory and matched single-state recurrence specifically when the current query determines which temporal scale matters, and the advantage survives a selective-SSM control.

### Useful narrowing

- Ungated bank ≈ gated bank: multiple timescales matter; query gating does not.
- Selective SSM ≈ lens model: the mechanism is a recognizable selective-state-space computation, not a distinct architecture.
- Transformer wins strongly: explicit random access is worth the memory cost for these tasks.
- Delayed loops add nothing: biological loop language is unnecessary for the computational result.
- Oscillations/PAC add nothing: temporal lenses do not require rhythmic dynamics.

### Failure

If a matched single-state recurrent model solves the tasks as well as the lens bank, or if apparent gains vanish under budget/leakage controls, the central multiple-lens claim fails for the tested setting. That is a valid repository result and should be documented as such.

## 17. Deferred work

Not in the first implementation:

- biologically detailed multicompartment neurons;
- large language-model training;
- claims about consciousness or quantum mechanics;
- fitting cortical frequency bands to hand-picked loops;
- replacing a production transformer KV cache;
- image/shadow inverse-model experiments.

Those can be connected later only if the minimal computational gates justify them.

## 18. Immediate implementation objective

Build the smallest reproducible version that can falsify the primary claim:

1. deterministic synthetic `query chooses timescale` task;
2. single-state, ungated multi-lens, and query-gated multi-lens models;
3. matched state/parameter budgets where feasible;
4. held-out results plus lens-selection diagnostics;
5. README that separates established neuroscience, computational analogy, and measured repository results.

Only after this first gate is frozen should the project expand into delayed loops, spectral modes, and PAC.
