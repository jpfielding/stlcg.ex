defmodule STLCG.FormulaTest do
  use ExUnit.Case, async: true

  alias STLCG.Expression
  alias STLCG.Test.NoopOperator

  setup do
    x = Expression.new("x", Nx.tensor([[[1.0], [-2.0], [3.0]]]))
    leaf = %NoopOperator{leaf?: true, name: "x"}
    wrapped = %NoopOperator{leaf?: false, wrap: leaf}
    {:ok, x: x, leaf: leaf, wrapped: wrapped}
  end

  test "walker dispatches through Formula protocol for leaf", ctx do
    trace = STLCG.robustness_trace(ctx.leaf, %{"x" => ctx.x})
    assert Nx.to_flat_list(trace) == [1.0, -2.0, 3.0]
  end

  test "walker recurses through subformulas, Noop wrap negates", ctx do
    trace = STLCG.robustness_trace(ctx.wrapped, %{"x" => ctx.x})
    assert Nx.to_flat_list(trace) == [-1.0, 2.0, -3.0]
  end

  test "robustness/3 returns the terminal-time slice", ctx do
    r = STLCG.robustness(ctx.wrapped, %{"x" => ctx.x})
    # Terminal slice preserves {batch, features} shape per semantics contract.
    assert Nx.shape(r) == {1, 1}
    # wrapped negates, so final entry is -3.0
    assert Nx.to_flat_list(r) == [-3.0]
  end

  test "compile/1 returns a reusable closure", ctx do
    fun = STLCG.compile(ctx.wrapped)
    t1 = fun.(%{"x" => ctx.x})
    t2 = fun.(%{"x" => Expression.new("x", Nx.tensor([[[5.0]]]))})
    assert Nx.to_flat_list(t1) == [-1.0, 2.0, -3.0]
    assert Nx.to_flat_list(t2) == [-5.0]
  end

  test "operator_tags/1 walks the tree", ctx do
    tags = STLCG.operator_tags(ctx.wrapped)
    assert MapSet.equal?(tags, MapSet.new([:noop_wrap, :noop_leaf]))
  end

  test "robustness_trace accepts raw Nx tensors as inputs (not just Expressions)" do
    leaf = %NoopOperator{leaf?: true, name: "y"}
    # Exactly-representable floats to avoid fp32 precision surprises in equality.
    y = Nx.tensor([[[0.5], [0.25]]])
    trace = STLCG.robustness_trace(leaf, %{"y" => y})
    assert Nx.to_flat_list(trace) == [0.5, 0.25]
  end

  test "STLCG.version/0 returns the package version" do
    assert STLCG.version() =~ ~r/^\d+\.\d+\.\d+/
  end
end
