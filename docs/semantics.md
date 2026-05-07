# `stlcg.ex` — Semantics Contract

This document locks the *exact numerical semantics* that every operator in
`stlcg.ex` MUST satisfy. It is authoritative: if the code disagrees with
this document, the code is wrong; if an upstream quirk disagrees with this
document, the quirk is out of scope.

Operator implementations follow the upstream Stanford ASL PyTorch
[`stlcg`](https://github.com/StanfordASL/stlcg) library where semantics are
unambiguous, and this contract where upstream is under-specified or
data-dependent in ways `Nx.Defn` cannot express.

---

## 1. Trace layout

- **Input trace shape**: `{batch, time, features}`, row-major, with time
  increasing along axis 1 (index 0 = earliest, index `time - 1` = latest).
- **Output trace shape**: same — `{batch, time, features}`.

  Entry `trace[b, t, f]` is the robustness evaluated over the **suffix
  starting at time `t`** — i.e. the value such that if `t` is the
  "current time", `trace[b, t, f] ≥ 0` iff the formula is satisfied
  at time `t` looking forward.

  This matches upstream's `_run_cell` output ordering. See
  `Temporal_Operator._run_cell` in upstream `src/stlcg.py`.

- **Scalar robustness**: `STLCG.robustness(formula, inputs)` returns
  `trace[..., time - 1, ...]` — the robustness at the *final* time step.
  Equivalently: the full-trace robustness viewed from time 0 after
  upstream's internal time-reversal convention.

## 2. Intervals

Intervals are **discrete time-step counts**, never wall-clock. Callers are
responsible for translating between sampling rate and index count.

| Elixir value        | Meaning                      | Upstream `interval` |
|---------------------|------------------------------|---------------------|
| `nil`               | `[0, ∞)` — full future       | `None`              |
| `{a, :infinity}`    | `[a, ∞)` with `a ≥ 0`        | `[a, np.inf]`       |
| `{a, b}`            | `[a, b]` inclusive, `b ≥ a`  | `[a, b]`            |

For `{a, b}` the *window size* is `steps = b - a + 1`. Intervals are
pattern-matched in plain Elixir before the `defn` call; `a`, `b`, and
`steps` become compile-time constants in the `defn` closure.

## 3. Logical operators

Let `ρ₁(t)`, `ρ₂(t)` be the robustness traces of sub-formulas.

| Operator | Robustness trace                   |
|----------|------------------------------------|
| `And(ϕ, ψ)`     | `min(ρϕ(t), ρψ(t))`          |
| `Or(ϕ, ψ)`      | `max(ρϕ(t), ρψ(t))`          |
| `Not(ϕ)`        | `-ρϕ(t)`                      |
| `Implies(ϕ, ψ)` | `max(-ρϕ(t), ρψ(t))`         |

The `min`/`max` used here is whichever `Maxish`/`Minish` mode the caller
selected (true/soft/AGM/distributed — see §6). Upstream's
`And`/`Or` always dispatch through `Minish`/`Maxish`, so the mode choice
propagates naturally.

## 4. Predicate operators

Let `trace(t)` be the signal (the `.value` of an `Expression`) and `val`
a scalar threshold. `pscale` is an optional positive scalar (default `1.0`).

| Operator           | Robustness trace                     |
|--------------------|--------------------------------------|
| `LessThan(lhs, val)`    | `(val - trace(t)) * pscale`     |
| `GreaterThan(lhs, val)` | `(trace(t) - val) * pscale`     |
| `Equal(lhs, val)`       | `-abs(trace(t) - val) * pscale` |
| `Identity(name)`        | `trace(t) * pscale`             |

## 5. Temporal operators

All temporal operators share the reverse-time dynamic-programming shape:
the output at time `t` depends on the suffix `trace[t..time-1]`, where
aggregation runs over the relevant *future* window.

### 5.1 `Always(ϕ, interval)` — `□_I ϕ`

For each time `t`, aggregate `ρϕ` over `{t + i : i ∈ I ∩ valid}` with `min`:

```
ρ_□(t) = min_{i ∈ I, t + i < time}  ρϕ(t + i)
```

Windows that fall entirely outside `[t, time - 1]` are filled with
`+∞` before the min (so a short trace near its end still produces a
defined value — matching upstream's masking behavior). See §8 for edge
cases.

### 5.2 `Eventually(ϕ, interval)` — `◇_I ϕ`

Mirror of Always with `max` and empty-window fill `-∞`:

```
ρ_◇(t) = max_{i ∈ I, t + i < time}  ρϕ(t + i)
```

### 5.3 `Until(ϕ, ψ, interval, overlap)` — `ϕ U_I ψ`

Upstream's definition (4-D tensor construction, `src/stlcg.py:Until`):

```
ρ_U(t) = max_{t' ∈ I offset from t}  min(  ρψ(t'),   min_{t'' ∈ [t, t']}  ρϕ(t'')  )
```

i.e. there exists some future time `t*` within the interval offset from
`t` where ψ holds, and ϕ held through every step from `t` up to (and by
default *including*) `t*`.

- `overlap: true` (default) — ϕ must hold at *every* time in `[t, t*]`
  including `t*` itself.
- `overlap: false` — ϕ must hold at every time in `[t, t* - 1]`; ψ
  strictly follows.

### 5.4 `Then(ϕ, ψ, interval, overlap)` — `ϕ T_I ψ`

Substitute `max` for the inner `min_{t''}` in §5.3: ϕ must *eventually*
(not *always*) have held in the prefix.

```
ρ_T(t) = max_{t' ∈ I offset from t}  min(  ρψ(t'),   max_{t'' ∈ [t, t']}  ρϕ(t'')  )
```

### 5.5 `Integral1d(ϕ, interval)`

Cumulative trapezoidal integration of `ρϕ` over the interval window at
each `t`. Unit spacing (`dx = 1`) unless the caller rescales. Empty
windows produce `0.0`.

## 6. Aggregation modes (`Maxish` / `Minish`)

Each reduction (`min`, `max`) in §3 and §5 is parameterized by a **mode**
selected in plain Elixir before entering `defn`. The four modes:

### 6.1 `true`

`Nx.reduce_max` / `Nx.reduce_min`. Exact, non-smooth, gradient
concentrated on the argmax/argmin index.

### 6.2 `distributed`

Tied-argmax average:

```
mask = stop_grad( Nx.equal(x, Nx.reduce_max(x, axes: …, keep_axes: true)) )
out  = Nx.sum(x * mask, axes: …, keep_axes: …) / Nx.sum(mask, axes: …, keep_axes: …)
```

Same numerical output as `true` in the no-tie case; gradient is
distributed uniformly across tied entries. The `stop_grad` is
documentation of intent — the equality comparator has zero gradient
anyway, but making this explicit prevents accidental backward-breakage if
a future refactor inlines the mask.

### 6.3 `soft` (log-sum-exp)

```
out = Nx.logsumexp(x * scale, axes: …, keep_axes: …) / scale
```

with `scale > 0`. Symmetric form for Minish: negate in and out. As
`scale → ∞`, soft → true; as `scale → 0⁺`, soft → mean.

### 6.4 `agm` (arithmetic-geometric mean)

Per-window on caller-supplied `axes`, `keep_axes: true` internally:

```
pos_mask  = Nx.greater(x, 0)                            # strict >
pos_count = Nx.sum(pos_mask, axes: …, keep_axes: true)
pos_sum   = Nx.sum(x * pos_mask, axes: …, keep_axes: true)
pos_mean  = pos_sum / Nx.max(pos_count, 1)

neg_branch = 1 - Nx.exp(Nx.mean(Nx.log(Nx.max(1 - x, tiny_eps)),
                                 axes: …, keep_axes: true))

any_pos = Nx.any(pos_mask, axes: …, keep_axes: true)
out     = Nx.select(any_pos, pos_mean, neg_branch)
```

- `tiny_eps = 1.0e-12` — prevents `log(0)` NaN when `x` hits exactly `1`.
- **Caller precondition**: values must be `< 1` when the neg branch is
  taken. `stlcg.ex` does not enforce this at runtime inside `defn`
  (data-dependent host branching is forbidden there). Parity fixtures
  respect this precondition.
- **All-zero / all-non-positive**: `any_pos = false`, neg branch fires,
  `log(1 - 0) = 0`, `exp(0) = 1`, result `= 0.0`. Authoritative; any
  upstream divergence is out of scope.

### Minish

Every mode above has a Minish counterpart: negate input, apply Maxish,
negate output. Implementation is a single `negate -> maxish -> negate`
wrapper, not a separate kernel set.

## 7. Predicate scaling (`pscale`)

`pscale` multiplies every predicate's robustness and passes through the
chain unchanged. Defaults to `1.0`. Must be positive; zero collapses the
formula to constant-zero robustness and is a user error (not checked).

## 8. Edge cases (authoritative)

These rules supersede any under-specified upstream behavior:

| Situation                             | Behavior                          |
|---------------------------------------|-----------------------------------|
| `time_len = 1`                        | Passthrough — aggregation is identity on a singleton window |
| `{a, b}` with `a > time_len - 1`      | Entry clipped to `+∞` (Always) / `-∞` (Eventually) |
| `{a, :infinity}` with `a > time_len - 1` | Same as above                    |
| AGM all-zero / all-non-positive       | `0.0` (follows algebraically)     |
| AGM input `≥ 1` in neg branch         | Caller precondition violation — undefined; callers must not pass such values |
| `Integral1d` empty window             | `0.0`                             |
| `Until` / `Then` with empty interval  | `-∞`  (no witness time exists)    |

## 9. Hand-computable oracle set

Before any Python-generated parity fixture lands, these five traces
encode the contract independently. Each lives in `test/support/oracle.ex`
and is used by every operator's unit test.

| id  | trace `x` (shape `{1, T, 1}`) | purpose |
|-----|------------------------------|---------|
| O1  | `[0.0]`                           | `T = 1`: degenerate passthrough |
| O2  | `[0.5, -0.3, 0.1]`                | mixed-sign, monotone neither direction |
| O3  | `[1.0, 1.0, 1.0, 1.0, 1.0]`       | constant positive — Always robustness = 1.0 |
| O4  | `[-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]` | monotone increasing — Always = -0.2 at t=0, Eventually = 0.5 at t=0 |
| O5  | `[0.3, -0.1, 0.2, -0.4, 0.1]`     | tie-break stress (two equal minima at t=1 and t=3 with threshold 0) |

Expected robustness for each operator × each oracle is codified as
test data in `test/support/oracle.ex`.

## 10. References

- Upstream source pinned to commit
  `abd16c92108f1b57a72d66c58492c949b6c5a8ea` at port time. Re-resolve
  only if a parity gap surfaces.
- Leung et al. *Back-propagation through Signal Temporal Logic
  Specifications.* IJRR 2023.
- Notation and recursive semantics adapted from
  Donzé & Maler, "Robust Satisfaction of Temporal Logic over Real-Valued
  Signals." FORMATS 2010.
