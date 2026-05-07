defmodule STLCG do
  @moduledoc """
  `STLCG` — an idiomatic Elixir port of Stanford ASL's STLCG toolbox for
  computing the robustness of **Signal Temporal Logic (STL)** formulas as
  differentiable computation graphs, built on `Nx` + `Nx.Defn`.

  See `PLAN.md` at the repository root for the implementation plan and
  `docs/semantics.md` for the full semantic contract.

  ## Public API

  - `robustness_trace/3` — per-time-step robustness as an `Nx.Tensor` of
    shape `{batch, time, features}`.
  - `robustness/3` — scalar robustness at the terminal time step
    (upstream convention: `trace[..., -1, ...]`).
  - `compile/1` — lower a formula struct tree to a reusable function that
    expects an inputs map and returns a robustness trace. Safe to call
    inside `defn`.

  ## How it works

  `robustness_trace/3` is plain Elixir. It pattern-matches on the
  top-level struct tag, recursively computes sub-formula traces (still in
  plain Elixir), and dispatches each operator's numerics to a `defn`
  kernel via the `STLCG.Formula` protocol. Because `defn` functions called
  from within another `defn` frame inline into the same expression graph,
  the whole composition is a single differentiable graph — protocol
  dispatch happens only at compile time (of the walker), not at numeric
  runtime.
  """

  alias STLCG.Formula

  @type formula :: struct()
  @type inputs :: map() | tuple() | struct()
  @type opts :: keyword()

  @default_opts [pscale: 1.0, scale: -1.0, agm?: false, distributed?: false]

  @doc """
  Returns the version string declared in `mix.exs`.
  """
  @spec version() :: String.t()
  def version, do: Application.spec(:stlcg, :vsn) |> to_string()

  @doc """
  Compute the robustness trace of `formula` against `inputs`.

  `inputs` is typically a map `%{name => Expression_or_Tensor}`. Leaf
  predicates look up their signal by name from this map.

  `opts` supports:

    * `:pscale` — predicate scale (default `1.0`)
    * `:scale` — soft max/min scale for `Maxish`/`Minish` (`< 0` → true,
      `> 0` → log-sum-exp soft, default `-1.0`)
    * `:agm?` — use arithmetic-geometric mean aggregation (default `false`)
    * `:distributed?` — distribute gradient over tied argmax (default `false`)
  """
  @spec robustness_trace(formula(), inputs(), opts()) :: Nx.Tensor.t()
  def robustness_trace(formula, inputs, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts) |> Keyword.put(:inputs, inputs)
    walk(formula, opts)
  end

  @doc """
  Scalar robustness at the terminal time step (final slice along the time axis).
  """
  @spec robustness(formula(), inputs(), opts()) :: Nx.Tensor.t()
  def robustness(formula, inputs, opts \\ []) do
    trace = robustness_trace(formula, inputs, opts)
    terminal_slice(trace)
  end

  @doc """
  Lower a formula tree to a 1-arg function `inputs -> Nx.Tensor` suitable
  for calling inside `defn` or for repeated use with varying inputs.

  The returned closure captures the formula struct tree and the options
  at compile time; only the tensor-bearing inputs are open at call time.
  """
  @spec compile(formula(), opts()) :: (inputs() -> Nx.Tensor.t())
  def compile(formula, opts \\ []) do
    fn inputs -> robustness_trace(formula, inputs, opts) end
  end

  @doc """
  Return every operator tag present in the formula tree, useful for the
  parity-fixture harness's skip-logic.
  """
  @spec operator_tags(formula()) :: MapSet.t(atom())
  def operator_tags(formula) do
    tag = Formula.operator_tag(formula)
    children = Formula.subformulas(formula)

    Enum.reduce(children, MapSet.new([tag]), fn child, acc ->
      MapSet.union(acc, operator_tags(child))
    end)
  end

  # --- Internals -------------------------------------------------------

  @doc false
  @spec walk(formula(), opts()) :: Nx.Tensor.t()
  def walk(formula, opts) do
    subtraces = Enum.map(Formula.subformulas(formula), &walk(&1, opts))
    Formula.robustness_trace(formula, subtraces, opts)
  end

  # The terminal slice is computed in plain Elixir (the shape is statically
  # known after the walker runs; this is just an index into the last time).
  # Upstream keeps the time axis at size 1 — shape is {batch, 1, features}.
  defp terminal_slice(trace) do
    case Nx.shape(trace) do
      {_batch, time, _features} ->
        Nx.slice_along_axis(trace, time - 1, 1, axis: 1)

      {_batch, time} ->
        Nx.slice_along_axis(trace, time - 1, 1, axis: 1)

      _ ->
        trace
    end
  end
end
