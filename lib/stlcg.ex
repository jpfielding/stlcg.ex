defmodule STLCG do
  @moduledoc """
  `STLCG` — an idiomatic Elixir port of Stanford ASL's STLCG toolbox for
  computing the robustness of **Signal Temporal Logic (STL)** formulas as
  differentiable computation graphs, built on `Nx` + `Nx.Defn`.

  ## Overview

  An STL formula `ϕ` is evaluated against a signal trace `x` to produce a
  *robustness value* `ρ(ϕ, x)` — a real number whose sign indicates
  satisfaction and whose magnitude indicates the margin. Because `ρ` is
  (sub-)differentiable, it can be plugged straight into gradient-based
  learning loops.

  ## Status

  `v0.1.0` is a work in progress. See `PLAN.md` at the repository root for
  the implementation plan and `docs/semantics.md` for the full semantic
  contract this port honors.

  ## Planned public entry points

  - `STLCG.robustness_trace/3` — computes the robustness trace (per-time-step).
  - `STLCG.robustness/3`       — final-time robustness scalar (per batch).
  - `STLCG.compile/1`          — lowers a formula to a reusable callable.

  These APIs are not yet wired up — they are placeholders until the
  operator set lands (see `PLAN.md` ticket table).
  """

  @typedoc "An STL formula expressed as a struct tree (see `STLCG.Formula`)."
  @type formula :: struct()

  @typedoc "Inputs to a formula — typically a map of variable-name => `STLCG.Expression` or `Nx.Tensor`."
  @type inputs :: map() | tuple() | struct()

  @doc """
  Returns the version string declared in `mix.exs`.
  """
  @spec version() :: String.t()
  def version, do: Application.spec(:stlcg, :vsn) |> to_string()
end
