# Changelog

All notable changes to this project are documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the
project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-05-07

Initial port of Stanford ASL's STLCG to Elixir / Nx.

### Added

- `STLCG.Expression` — named signal wrapper, derives `Nx.Container`.
- `STLCG.Formula` protocol — plain-Elixir tree walker + per-operator
  `defn` kernels that compose into a single differentiable graph.
- Predicates: `LessThan`, `GreaterThan`, `Equal`, `Identity`.
- Logical ops: `And`, `Or`, `Not`, `Implies`.
- Temporal ops: `Always`, `Eventually` with `nil / {a, :infinity} / {a, b}`
  interval kinds.
- `Until`, `Then` — past-witness search with bounded and unbounded intervals.
- `Integral1d` — past-window sum with first-value padding.
- Aggregation kernels: `Maxish` / `Minish` in four modes
  (`:true_`, `:distributed`, `:soft`, `:agm`).
- `STLCG.DSL` macro sugar — `<~`, `~>`, `&&&`, `|||`, plus
  `always/eventually/until/then_/integral` helpers.
- `STLCG.Fixtures` — loader for Python-generated parity fixtures with
  regime-based tolerance matrix.
- 60 JSON parity fixtures exercising every operator × interval across
  five hand-computable oracle traces; all green at hard-f32 tolerance.
- Autodiff validation suite (`Nx.Defn.grad` vs central finite difference)
  across all operators.
- Livebook demo (`notebooks/demo.livemd`) mirroring upstream's
  `demo.ipynb`.

### Not included (deferred)

- `stlviz` graph-visualization export (tracked post-v0.1).
- Hex.pm publication (once external parity is validated in the wild).
- EXLA / GPU authoritative coverage (supported but not the default
  reference backend).

[0.1.0]: https://github.com/jpfielding/stlcg.ex/releases/tag/v0.1.0
