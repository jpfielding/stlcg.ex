defmodule STLCG.Test.NoopOperator do
  @moduledoc false
  # A minimal Formula implementation used to exercise the protocol and
  # the plain-Elixir tree walker without depending on any real operator.
  #
  # :leaf?  — when true, pulls a named tensor from opts[:inputs] and
  #           returns it as the trace (identity).
  # :wrap   — when set to a child formula, wraps it and *negates* its trace.

  defstruct [:leaf?, :wrap, :name]

  defimpl STLCG.Formula do
    def robustness_trace(%{leaf?: true, name: name}, [], opts) do
      inputs = Keyword.fetch!(opts, :inputs)

      case Map.fetch!(inputs, name) do
        %STLCG.Expression{value: v} -> v
        %Nx.Tensor{} = t -> t
      end
    end

    def robustness_trace(%{leaf?: false, wrap: _}, [child_trace], _opts) do
      Nx.negate(child_trace)
    end

    def subformulas(%{leaf?: true}), do: []
    def subformulas(%{leaf?: false, wrap: wrap}), do: [wrap]

    def operator_tag(%{leaf?: true}), do: :noop_leaf
    def operator_tag(%{leaf?: false}), do: :noop_wrap
  end
end
