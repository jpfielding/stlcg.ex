# stlcg.ex

[![CI](https://github.com/jpfielding/stlcg.ex/actions/workflows/ci.yml/badge.svg)](https://github.com/jpfielding/stlcg.ex/actions/workflows/ci.yml)

An idiomatic Elixir port of Stanford ASL's
[stlcg](https://github.com/StanfordASL/stlcg) — a toolbox for computing the
**robustness of Signal Temporal Logic (STL) formulas** as differentiable
computation graphs.

Built on [`Nx`](https://hex.pm/packages/nx) + `Nx.Defn`, so STL robustness
can be plugged straight into gradient-based learning loops: every operator
backpropagates.

> **Status:** v0.1.0 — all operators implemented and parity-validated
> against 60 JSON fixtures generated from upstream at pinned SHA
> `abd16c92`. See [PLAN.md](PLAN.md) for the implementation plan and
> [docs/semantics.md](docs/semantics.md) for the semantic contract.

## What is STL robustness?

Given a signal trace `x` and an STL formula `ϕ`, the *robustness value*
`ρ(ϕ, x)` is a real number whose **sign** indicates whether `x` satisfies
`ϕ` and whose **magnitude** indicates the margin of satisfaction.
Because `ρ` is continuous and (sub-)differentiable, it can serve as a
loss term in neural-network training.

## Planned operators (v0.1.0)

- **Predicates:** `LessThan`, `GreaterThan`, `Equal`, `Identity`
- **Logical:** `And`, `Or`, `Not`, `Implies`
- **Temporal:** `Always`, `Eventually`, `Until`, `Then` — each with
  `nil | {a, :infinity} | {a, b}` interval support
- **Integration:** `Integral1d`
- **Aggregation:** `Maxish`/`Minish` in four modes — true max/min,
  log-sum-exp soft, arithmetic-geometric mean (AGM), and
  distributed-gradient tied-max.

## Example (target API)

```elixir
import STLCG.DSL
import Nx.Defn

x = STLCG.Expression.new("x", Nx.tensor([[0.1, 0.3, 0.5, 0.2]]))
c = STLCG.Expression.new("c", Nx.tensor(1.0))

formula = always(x <~ c)                       # □ (x < c)

STLCG.robustness(formula, %{x: x, c: c})
#=> #Nx.Tensor<f32 ...>
```

## Status & roadmap

See [PLAN.md](PLAN.md). Open issues at
<https://github.com/jpfielding/stlcg.ex/issues>.

## Attribution

See [NOTICE](NOTICE). This is a third-party port, not an official
Stanford ASL release.

## License

MIT — see [LICENSE](LICENSE).
