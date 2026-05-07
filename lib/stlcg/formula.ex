defprotocol STLCG.Formula do
  @moduledoc """
  Protocol implemented by every STL-formula struct (`%Always{}`,
  `%And{}`, `%LessThan{}`, …).

  ## What this protocol does *not* do

  Protocol dispatch is **not available inside `Nx.Defn`** — there is no
  way to pattern-match on a struct tag from within a `defn` frame.
  Therefore this protocol is *only* consulted by the plain-Elixir tree
  walker in `STLCG` (`robustness_trace/3`), which dispatches to
  per-operator `defn` kernels based on the struct tag and composes them
  into a single differentiable expression graph.

  The canonical implementation pattern for an operator module is:

      defmodule STLCG.SomeOp do
        import Nx.Defn

        defstruct [...]

        # The defn kernel — called by the walker with plain tensors.
        defn kernel(sub_trace, opts \\ []), do: ...

        defimpl STLCG.Formula do
          def robustness_trace(%{} = op, subtraces, opts) do
            STLCG.SomeOp.kernel(subtraces, opts)
          end

          def subformulas(%{sub: s}), do: [s]
          def operator_tag(_), do: :some_op
        end
      end

  ## Callbacks

  - `robustness_trace/3` — given an operator struct, the tensor traces of
    its sub-formulas (pre-computed by the walker), and an option map,
    return the robustness trace for this operator. Must be callable from
    a `defn` frame.

  - `subformulas/1` — return the direct sub-formulas as a list in
    walk order. Used by the walker and by `operator_tags/1` for skip-logic.

  - `operator_tag/1` — return a stable atom naming this operator (e.g.
    `:always`, `:less_than`). Used by the parity-fixture harness to skip
    fixtures whose required operators aren't yet implemented.
  """

  @doc """
  Compute the robustness trace for this operator given pre-computed
  sub-formula traces.

  `subtraces` is an ordered list of `Nx.Tensor`s in the same order as
  `subformulas/1` returns. Leaf operators (predicates) receive an empty
  list and pull signal data from the `inputs` keyword of `opts`.

  `opts` is the options map flowed down from the top-level call, with at
  least `:inputs`, `:pscale`, `:scale`, `:agm?`, `:distributed?`.
  """
  @spec robustness_trace(t(), [Nx.Tensor.t()], keyword()) :: Nx.Tensor.t()
  def robustness_trace(formula, subtraces, opts)

  @doc """
  Return the direct sub-formulas as a list in walk order.
  Leaf operators return `[]`.
  """
  @spec subformulas(t()) :: [t()]
  def subformulas(formula)

  @doc """
  A stable atom naming this operator. Used for fixture skip-logic and
  error messages. Must match the string used in fixture `required_operators`.
  """
  @spec operator_tag(t()) :: atom()
  def operator_tag(formula)
end
