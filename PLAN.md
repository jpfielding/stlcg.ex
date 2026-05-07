# Plan: `stlcg.ex` — Idiomatic Elixir Port of Stanford ASL's STLCG

## Context

[StanfordASL/stlcg](https://github.com/StanfordASL/stlcg) is a PyTorch toolbox that computes **robustness** of Signal Temporal Logic (STL) formulas on signal traces via computation graphs, enabling gradient-based learning over temporal-logic specifications. This port delivers an idiomatic Elixir-native equivalent — `stlcg.ex` — using `Nx` + `Nx.Defn` for tensor ops and autodiff. Target: public repo `github.com/jpfielding/stlcg.ex` with full feature parity, fixture-backed numerical parity against upstream, and a Livebook demo mirroring `demo.ipynb`.

**Working directory**: `/Users/fieldingj/projects/stlcg.ex` (empty; greenfield).

## Semantics contract (locked before any operator code)

Before any temporal-operator implementation lands, a short `docs/semantics.md` commits to the exact upstream conventions, written as a contract our port MUST satisfy:

- **Input trace layout**: `{batch, time, features}`, time increasing with index.
- **Output trace layout**: same shape; entry `t` is the robustness evaluated over the suffix starting at time `t` (upstream convention). `STLCG.robustness(formula, inputs)` returns `trace[..., -1, ...]` (final time slice), not index 0. Confirmed against `src/stlcg.py` `Temporal_Operator._run_cell` output ordering.
- **Intervals**: `nil ≡ [0, ∞)`; `{a, :infinity}` is `[a, ∞)`; `{a, b}` is the inclusive closed interval with `steps = b - a + 1`; indices count discrete time steps, not wall-clock.
- **Until overlap**: `overlap: true` means the witness time `t*` for ψ may coincide with the last φ satisfaction; `false` requires strict precedence. Default `true` (upstream).
- **Edge cases fixed by decree** (authoritative; supersedes any upstream quirk):
  - AGM all-zero / all-non-positive → `0.0` (follows from `neg_branch = 1 - exp(mean(log(1 - 0))) = 0`).
  - AGM input domain: caller guarantees all values `< 1` in the neg branch (positive robustness values are not subject to this constraint because they take the pos branch). This is a *caller precondition*, not a runtime check inside `defn` — Elixir is free to raise before the `defn` call if it can cheaply verify (e.g., caller passed a concrete literal > 1), but parity fixtures assume the caller obeys the precondition.
  - `time_len = 1` trace on any interval → degenerate passthrough robustness (single-element min/max is identity).
  - `{a, b}` with `a > time_len - 1` → final entries clipped to `-∞` (Always) / `+∞` (Eventually) per upstream masking.
- **Hand-computable oracle set**: 5 short traces (`time_len ∈ {3, 5, 8}`) with analytic robustness for each operator landing before Python-generated fixtures. These encode the semantics contract independently of upstream code.

## Decisions (confirmed)

1. **Backend**: `{:nx, "~> 0.11"}` with `Nx.Defn` for autodiff (0.11.0 is the current Hex release; `~> 0.11` allows 0.11.x patch bumps, blocks 0.12 API churn). No Axon (optimizers handled directly in demos via `Nx.Defn.grad`).
2. **Scope for v0.1.0**: Full parity — all predicates, logical ops, `Always`/`Eventually`/`Until`/`Then`, `Integral1d`, `Maxish`/`Minish` (true, logsumexp-soft, AGM, distributed-grad), DSL macro, demo Livebook, parity test harness. **stlviz graphviz export is deferred.**
3. **Parity validation**: Python venv + upstream `stlcg` clone produces canonical fixtures (`fixtures/*.json`); Elixir tests load and compare within tolerance.

## Architecture

```
stlcg.ex/
├── mix.exs                              # project, deps: nx, ex_doc, credo, dialyxir, stream_data, jason
├── README.md                            # usage, install, attribution to Stanford ASL
├── LICENSE                              # MIT (port); upstream MIT preserved in NOTICE
├── NOTICE                               # upstream copyright attribution
├── .formatter.exs .credo.exs .github/workflows/ci.yml
├── config/config.exs
├── lib/
│   ├── stlcg.ex                         # public facade: robustness/3, robustness_trace/3
│   ├── stlcg/expression.ex              # %Expression{name, value}; Nx.Container impl
│   ├── stlcg/formula.ex                 # protocol Formula w/ robustness_trace/3
│   ├── stlcg/aggregation.ex             # maxish/minish defns
│   ├── stlcg/predicates.ex              # LessThan, GreaterThan, Equal, Identity
│   ├── stlcg/logical.ex                 # And, Or, Not, Implies
│   ├── stlcg/temporal.ex                # Always, Eventually (reverse-time DP via Nx while)
│   ├── stlcg/until_then.ex              # Until, Then (4-D nested min/max)
│   ├── stlcg/integral.ex                # Integral1d
│   ├── stlcg/dsl.ex                     # `use STLCG.DSL` sugar: always/1, eventually/1, band/bor/bnot, lt/gt
│   └── stlcg/inspect.ex                 # Inspect protocol impls for pretty formulas
├── test/
│   ├── expression_test.exs predicates_test.exs logical_test.exs
│   ├── aggregation_test.exs temporal_test.exs until_then_test.exs integral_test.exs
│   ├── autodiff_test.exs                # Nx.Defn.grad through each operator
│   ├── dsl_test.exs
│   └── parity_test.exs                  # loads fixtures/*.json, compares
├── fixtures/
│   ├── gen_fixtures.py                  # reproducer script — runs upstream stlcg
│   ├── requirements.txt                 # torch, numpy, stlcg from git
│   └── *.json                           # canonical robustness traces
├── notebooks/
│   └── demo.livemd                      # Livebook mirroring demo.ipynb
└── scripts/
    └── bootstrap_fixtures.sh            # venv + clone stlcg + run gen_fixtures.py
```

### Core types

- **`STLCG.Expression`** — struct `%Expression{name: String.t(), value: Nx.Tensor.t()}`. Derives `Nx.Container` with `containers: [:value], keep: []` (name is *not* kept in numerical calls — it is formula metadata consumed only outside `defn`, to avoid per-name compiler cache churn). Public functional API: `Expression.new/2`, `Expression.add/2`, `Expression.lt/2`, etc. `name` is threaded through formula construction for Inspect/DSL ergonomics; before a call into `defn`, the formula walker extracts `value` tensors and passes them as plain args.
- **`STLCG.Formula`** — protocol with callback `compile(formula)` that returns a `{defn_fun, param_shape_spec}` closure plus metadata. **Plain Elixir walks the struct tree**; the compiled closure runs the numerics inside `defn`. (Protocol dispatch is not available inside `defn` — this is the key structural constraint.)
- **`STLCG.robustness_trace/3`** is a regular Elixir function that pattern-matches on operator structs and composes per-operator `defn` kernels. `STLCG.robustness/3` takes the terminal time slice from the trace — the exact slice follows upstream's `_run_cell` output convention, which will be lifted verbatim from Python and pinned by fixtures rather than reasoned about abstractly.
- **Operator structs** — `%Always{subformula, interval}`, `%Eventually{subformula, interval}`, `%Until{lhs, rhs, interval, overlap}`, `%Then{...}`, `%And{lhs, rhs}`, `%Or{lhs, rhs}`, `%Not{subformula}`, `%Implies{lhs, rhs}`, `%LessThan{lhs, val}`, `%GreaterThan{lhs, val}`, `%Equal{lhs, val}`, `%Identity{name}`, `%Integral1d{subformula, interval}`. Intervals: `nil | {a, :infinity} | {a, b}` (tagged tuples, not lists, for pattern matching).

### Temporal operator implementation (the hard part)

Upstream's RNN maps to `Nx.Defn.Kernel.while/4` with three shape-stable loop states per interval kind:

- **`interval = nil`** → scan accumulating running min (Always) / max (Eventually) over the growing suffix.
- **`interval = {a, :infinity}`** → two-part hidden state `{d0, h0}` where `h0` is a fixed-length shift buffer (length `a`) and `d0` is the aggregated delay region; at each step append new `x`, matrix-shift `h0`, aggregate over `[d0, h0[0]]`.
- **`interval = {a, b}`** → fixed-length buffer of size `steps = b - a + 1`; slice aggregate at each step.

Key `defn` constraints the plan respects:
- **Shape stability**: `output_acc` is preallocated `Nx.broadcast(0.0, {batch, time_len, features})` and written via `Nx.put_slice` at each step; it does not grow.
- **Static intervals**: `a`, `b`, `steps`, `rnn_dim` are compile-time constants in the closure (captured before `defn` is entered) — they cannot be tensor-valued.
- **No axis-reversal gymnastics**: rather than pre-reversing the time axis, the `while` iterates `t` in the direction upstream iterates and reads `inputs[.., time_len - 1 - t, ..]` when the semantics demand it. The exact indexing is transcribed from upstream `_run_cell` and locked by fixtures before any stylistic refactoring.

### Aggregation (`Maxish` / `Minish`)

The four modes are *separate `defn` kernels* (not runtime-branched) — the mode is selected in plain Elixir before the `defn` call so that each kernel is shape-static:

- `true` → `Nx.reduce_max/2` or `Nx.reduce_min/2`.
- `distributed` → `mask = stop_grad(Nx.equal(x, Nx.reduce_max(x, ...)))`; return `Nx.sum(x * mask) / Nx.sum(mask)`. `stop_grad` documents intent (the equality comparator has zero gradient anyway, but making this explicit prevents accidental backward-breakage). Autodiff distributes gradient uniformly across tied entries.
- `soft` (`scale > 0, agm = false`) → `Nx.logsumexp(x * scale) / scale`; for Minish, negate in and out.
- `agm` → sign-partitioned AGM without dynamic indexing. Reduction is **per-window** on the caller-supplied `axes` (same axes the corresponding `reduce_max` would use), `keep_axes: keepdim`:
  - `pos_mask = Nx.greater(x, 0)` (strict `>`, matching upstream `torch.gt`; zeros excluded).
  - `pos_count = Nx.sum(pos_mask, axes: axes, keep_axes: true)` and `pos_sum = Nx.sum(x * pos_mask, axes: axes, keep_axes: true)`.
  - `pos_mean = pos_sum / Nx.max(pos_count, 1)`.
  - `neg_branch = 1 - Nx.exp(Nx.mean(Nx.log(Nx.max(1 - x, tiny_eps)), axes: axes, keep_axes: true))`. Upstream is only well-defined for `x ∈ (-∞, 1)`; we guard `log(1-x)` with `tiny_eps = 1.0e-12` so numerics never NaN. Caller precondition documented above: values must be `< 1` when the neg branch is taken (parity fixtures respect this). No runtime check inside `defn` (data-dependent host branching is not allowed there).
  - Final: `Nx.select(Nx.any(pos_mask, axes: axes, keep_axes: true), pos_mean, neg_branch)`; then `Nx.squeeze` if `keepdim = false`.
  - The all-zero case (`any_positive = false`, all `x ≤ 0`) falls into the neg branch, giving `1 - exp(mean(log(1))) = 0` — matches upstream empty-positive behavior. Fixture covers both branches plus the all-zero edge case.

### Autodiff

Nx is functional — there is no `requires_grad=True` mutable tensor. Instead, `STLCG.robustness/3` takes a `params` keyword/map argument and returns an `Nx.Tensor`; gradients are obtained by wrapping it in a `defn` function:

```elixir
grad_fn = Nx.Defn.grad(fn params ->
  STLCG.robustness(formula, inputs_with(params), opts)
end)
grad_fn.(%{c: Nx.tensor(1.0), d: Nx.tensor(0.9)})
```

Because `robustness/3` itself is plain Elixir (walking the struct tree), the *composition* of per-operator `defn` kernels must be differentiable end-to-end. This works only if all per-operator kernels are `defn`-defined (not `jit`'d per-call) and their composition is invoked inside a single outer `defn`/`Nx.Defn.grad` frame — `defn` functions called from a `defn` frame inline into the same expression graph, preserving autodiff. The implementation path is:

1. `STLCG.compile(formula) :: (inputs -> tensor)` — plain-Elixir function returning a closure.
2. The closure body calls `defn`-defined operator kernels directly (no `jit` wrapping).
3. `Nx.Defn.grad(params, fn params -> compiled.(inputs_with(params)) end)` — single frame, single graph.

**Phase 0 spike (gate before any operator implementation)**: build *one* minimal nested formula (`Always(x <= c)`) using two kernels (predicate + temporal), compose via the above pattern, run `Nx.Defn.grad` on `c`, and compare against finite differences. If that does not produce a single differentiable graph, revisit the architecture before building out the operator set. `autodiff_test.exs` verifies each operator's gradient against finite differences (tolerance `1e-4` for float32, `1e-8` for float64).

**Vectors 2 and 7 (resolved, noted for implementation)**:
- `Nx.Defn.Kernel.while` + preallocated `output_acc` + `Nx.put_slice` with loop-variable `t` is supported — slice lengths must be compile-time integers but slice starts may be scalar tensors.
- `sum(x * mask) / sum(mask)` with `mask = stop_grad(x == reduce_max(x))` correctly distributes gradient across tied entries; a tied-max gradient fixture pins this behavior.

### DSL sugar

Elixir's overridable-operator set is fixed: `&&&`, `|||`, `^^^`, `~~~`, `<<<`, `>>>`, `<~`, `~>`, `<<~`, `~>>`, `<~>`, `<|>`. Of those, we use:

```elixir
use STLCG.DSL
# builds plain STLCG.* structs — no protocol dispatch, no defn

x = expr("x", x_tensor); c = expr("c", c_tensor)
w = expr("w", w_tensor); d = expr("d", d_tensor)

# Predicates (binary): <~ lt   ~> gt
# Logical: &&& and   ||| or   ~~~ not (unary)   implies/2 (word form — no clean glyph)
# Temporal: always/1, always/2, eventually/1, eventually/2, until/2, until/3, then/2

formula = always(x <~ c) &&& always(w <~ d)
```

`<` / `>` / `==` stay as Kernel comparison (they compile to booleans on tensors, not formulas) — upstream-style dunder overloading is explicitly *not* available. A `dsl_test.exs` snapshot asserts every DSL form produces a struct tree identical to calling the plain constructors.

### Parity fixtures

Upstream `stlcg` has no `setup.py` — `demo.ipynb` uses `sys.path.insert(0, "src")`. `pip install git+...` may not expose the module. Bootstrap therefore clones at a **pinned commit SHA** and sets `PYTHONPATH`:

`scripts/bootstrap_fixtures.sh`:
1. `python3 -m venv .venv` — uses whatever `python3` resolves to (3.9+). `torch==2.2.2` and `numpy==1.26.4` both ship CPython 3.9 wheels, so no hard 3.11 requirement.
2. `pip install torch==2.2.2 numpy==1.26.4`.
3. `git clone --depth=1 https://github.com/StanfordASL/stlcg .cache/stlcg && git -C .cache/stlcg checkout abd16c92108f1b57a72d66c58492c949b6c5a8ea`. (Upstream HEAD at plan time; re-resolve only if a parity gap surfaces.)
4. `PYTHONPATH=.cache/stlcg/src python -c "import stlcg; stlcg.LessThan(lhs='x', val=1.0)"` — smoke import gate before fixture generation.
5. Runs `fixtures/gen_fixtures.py` dumping JSON `{formula_id, formula_ast, required_operators: ["LessThan", "Always", ...], inputs_shape, inputs_data, opts, dtype, backend: "pytorch-cpu-fp32", expected_trace, expected_robustness, meta: {depth, aggregation_modes}}`. The `required_operators` array drives skip-logic in `parity_test.exs`.

Fixtures are **tagged by the operator set required**. `test/parity_test.exs` skips any fixture whose required operators are not yet implemented (vs failing at repo-init when only the scaffolding exists). A separate `schema_test.exs` validates the JSON AST round-trips to a canonical `STLCG.Formula` struct — this gate catches oracle-side bugs without requiring operator numerics.

**Tolerance matrix** (replaces single `rtol=1e-5` claim):

| Regime | Aggregation | Depth | dtype | rtol | atol |
| ------ | ----------- | ----- | ----- | ---- | ---- |
| Hard   | true max/min | any   | f32   | 1e-6 | 1e-7 |
| Hard   | true max/min | any   | f64   | 1e-10 | 1e-12 |
| Soft   | logsumexp    | 1–2   | f32   | 1e-5 | 1e-6 |
| Soft   | logsumexp    | ≥3    | f32   | 1e-4 | 1e-5 |
| AGM    | agm          | any   | f32   | 1e-5 | 1e-6 |
| Distributed | mask/sum  | any   | f32   | 1e-6 | 1e-7 |

Each fixture carries its regime tuple; the harness reads it and applies the matching tolerance. A fixture failing its matrix-defined gate fails the build — no `10×` override. (An earlier draft proposed that; it renders the matrix decorative and is now removed.)

## Tickets (GitHub Issues)

Issues created on repo init; closed by commits referencing `Closes #N`.

| #  | Title                                                   | Gate                                    | Wave |
| -- | ------------------------------------------------------- | --------------------------------------- | ---- |
| 1  | Project scaffold (mix, CI, formatter, credo, dialyxir)  | `mix test` green on empty suite         | 0    |
| 2  | `STLCG.Expression` + derives `Nx.Container`             | doctest + non-tensor leaf test          | 0    |
| 3  | `STLCG.Formula` protocol + plain-Elixir tree walker     | no-op test                              | 0    |
| 4a | `fixtures/gen_fixtures.py` + bootstrap + smoke-import   | `bootstrap_fixtures.sh` runs end-to-end | 1    |
| 4b | JSON AST schema + `schema_test.exs` + `parity_test.exs` harness | round-trip + tolerance-matrix loader | 1    |
| 5  | Predicates (`LessThan`/`GreaterThan`/`Equal`/`Identity`)| fixture + analytic                      | 1    |
| 6  | `Maxish`/`Minish` (true/soft/AGM/distributed) — 4 kernels| fixture each mode                      | 1    |
| 7  | Logical ops (`And`/`Or`/`Not`/`Implies`)                | fixture + analytic                      | 2    |
| 8  | `Always` (all 3 interval cases)                         | fixture per case                        | 3    |
| 9  | `Eventually` (all 3 interval cases)                     | fixture per case                        | 3    |
| 10 | `Until` + `Then`                                        | fixture each                            | 3    |
| 11 | `Integral1d`                                            | fixture                                 | 3    |
| 12 | Autodiff validation across all operators                | finite-diff agreement                   | 4    |
| 13 | DSL macro (`use STLCG.DSL`)                             | sugar → same struct tree                | 4    |
| 14 | Livebook demo (`notebooks/demo.livemd`)                 | runs end-to-end                         | 5    |
| 15 | Optimization example (gradient descent on `c`)          | converges in Livebook                   | 5    |
| 16 | README, ExDoc, usage guide, attribution                 | `mix docs` clean                        | 5    |
| 17 | (stretch) stlviz-equivalent graphviz export             | deferred to v0.2                        | post |

**Execution-order issue numbers follow the table above**: autodiff is #12, DSL is #13, Livebook #14, optimization #15, README #16, stlviz #17. (Earlier drafts drifted; this is the canonical numbering.)

## Execution order

Issues are opened in waves as each phase begins, not all upfront.

1. **Phase 0 — Repo bootstrap + semantics**: `mix new`, deps, CI, `gh repo create`, commit semantics contract (`docs/semantics.md`), open #1–#3 (scaffold, Expression, Formula protocol).
2. **Phase 0.5 — Autodiff spike (gate)**: Build a minimal `Always(x <= c)` using a predicate `defn` kernel + a one-case temporal `defn` kernel composed inside a single outer `defn`. Run `Nx.Defn.grad` on `c`; compare to finite diff. **If this does not produce a single differentiable graph, revisit architecture before anything else.** No new issue — this is internal risk reduction and lives on a feature branch with a single test.
3. **Phase 1 — Fixture oracle**: open #4a (generator + smoke-import) and #4b (JSON AST schema + validator). Write `fixtures/gen_fixtures.py`, run via venv, commit JSON. Wave-open #5 (Predicates) and #6 (Maxish/Minish).
4. **Phase 2 — Leaves**: Predicates → Maxish/Minish → Logical ops. Each commit `Closes #N`. Wave-open #7–#10 when #5, #6 land.
5. **Phase 3 — Temporal ops**: Always / Eventually (three interval cases each) → Until / Then → Integral1d. Hand-computable oracle (from semantics contract) gates operator entry; Python-generated fixture gates operator parity.
6. **Phase 4 — Cross-cutting**: open #12 autodiff, #13 DSL.
7. **Phase 5 — Polish**: open #14–#16 (Livebook demo, optimization example, README/ExDoc + CI hardening). Tag `v0.1.0`.

Issue #17 (stlviz graphviz export) tracked as post-v0.1 backlog.

## Critical files to create (initial)

- `mix.exs` — project metadata, deps.
- `lib/stlcg/expression.ex` — data type; all else depends.
- `lib/stlcg/formula.ex` — protocol; shape of everything downstream.
- `lib/stlcg/aggregation.ex` — shared reduction primitive for And/Or/Always/Eventually.
- `lib/stlcg/temporal.ex` — reverse-time DP in `defn`; highest risk for subtle bugs.
- `fixtures/gen_fixtures.py` — golden-source oracle.
- `test/parity_test.exs` — the forcing function for correctness.

## Verification

- **Unit**: `mix test` — analytic cases per operator, including the 5-trace hand-computable oracle set from `docs/semantics.md`.
- **Parity**: `scripts/bootstrap_fixtures.sh` then `mix test --only parity` — every fixture within its tolerance-matrix regime.
- **Autodiff**: `mix test --only autodiff` — `Nx.Defn.grad` vs finite differences per the autodiff tolerances above.
- **Backend**: Default test backend is `Nx.BinaryBackend` (no XLA toolchain required on contributor machines). An optional `MIX_ENV=test NX_BACKEND=exla mix test --only exla_smoke` lane exercises a subset on EXLA-CPU when available (gated on `EXLA` env var so CI doesn't break on absent toolchains). Tolerances are calibrated against BinaryBackend vs PyTorch-CPU-fp32 — EXLA is documented as supported but not authoritative.
- **Demo**: `livebook server`, open `notebooks/demo.livemd`, run all cells; optimization loop converges `c → ~max(x)`.
- **CI**: GitHub Actions runs format check, credo, dialyzer, test on Elixir 1.18.2 / OTP 27 (matches local container; BinaryBackend only; no EXLA step).
- **Docs**: `mix docs` builds without warnings; README renders on github.com/jpfielding/stlcg.ex.

## Out of scope (for this port)

- `stlviz` graphviz export (deferred to v0.2; tracked as issue #17).
- Publication to Hex.pm (deferred until parity is proven in the wild).
- GPU support beyond what EXLA gives for free (documented, not tested).

## Attribution

Port retains upstream Stanford ASL MIT notice in `NOTICE`; README credits the paper ("STLCG: A Toolbox for Specifying Signal Temporal Logic Specifications with Backpropagation") and links back to `StanfordASL/stlcg`.
