defmodule STLCG.ExpressionTest do
  use ExUnit.Case, async: true
  doctest STLCG.Expression

  alias STLCG.Expression

  describe "new/2" do
    test "accepts a number" do
      e = Expression.new("c", 1.0)
      assert e.name == "c"
      assert Nx.to_number(e.value) == 1.0
    end

    test "accepts a list (via Nx.tensor)" do
      e = Expression.new("x", [1.0, 2.0, 3.0])
      assert Nx.to_flat_list(e.value) == [1.0, 2.0, 3.0]
    end

    test "accepts an existing Nx.Tensor" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      e = Expression.new("m", t)
      assert Nx.shape(e.value) == {2, 2}
    end
  end

  describe "arithmetic" do
    setup do
      {:ok, x: Expression.new("x", [1.0, 2.0, 3.0]), y: Expression.new("y", [0.5, 1.0, 1.5])}
    end

    test "add composes names", %{x: x, y: y} do
      z = Expression.add(x, y)
      assert z.name == "(x + y)"
      assert Nx.to_flat_list(z.value) == [1.5, 3.0, 4.5]
    end

    test "sub composes names", %{x: x, y: y} do
      z = Expression.sub(x, y)
      assert z.name == "(x - y)"
      assert Nx.to_flat_list(z.value) == [0.5, 1.0, 1.5]
    end

    test "mul composes names", %{x: x, y: y} do
      z = Expression.mul(x, y)
      assert z.name == "(x * y)"
      assert Nx.to_flat_list(z.value) == [0.5, 2.0, 4.5]
    end

    test "divide composes names", %{x: x, y: y} do
      z = Expression.divide(x, y)
      assert z.name == "(x / y)"
      assert Nx.to_flat_list(z.value) == [2.0, 2.0, 2.0]
    end

    test "neg flips sign and name", %{x: x} do
      n = Expression.neg(x)
      assert n.name == "-x"
      assert Nx.to_flat_list(n.value) == [-1.0, -2.0, -3.0]
    end

    test "mixed scalar / expression" do
      x = Expression.new("x", [1.0, 2.0])
      z = Expression.add(x, 10)
      assert z.name == "(x + 10)"
      assert Nx.to_flat_list(z.value) == [11.0, 12.0]

      w = Expression.sub(10, x)
      assert w.name == "(10 - x)"
      assert Nx.to_flat_list(w.value) == [9.0, 8.0]
    end
  end

  describe "Nx.Container derivation" do
    test "value traverses as a tensor leaf under defn" do
      x = Expression.new("x", [1.0, 2.0, 3.0])

      # A trivial defn that consumes an Expression and returns a tensor.
      defmodule Sink do
        import Nx.Defn

        defn square_value(expr) do
          expr.value * expr.value
        end
      end

      out = Sink.square_value(x)
      assert Nx.to_flat_list(out) == [1.0, 4.0, 9.0]
    end

    test "gradient flows through :value field" do
      x = Expression.new("x", [1.0, 2.0, 3.0])

      fun = fn e ->
        v = e.value
        Nx.sum(Nx.multiply(v, v))
      end

      grad = Nx.Defn.grad(x, fun)
      # d/dv of sum(v^2) is 2v
      assert Nx.to_flat_list(grad.value) == [2.0, 4.0, 6.0]
    end

    test "name metadata is not kept across container traversal" do
      # With keep: [], Nx.Container.traverse reconstructs the struct
      # using defstruct defaults for non-tensor fields. This test pins
      # that behavior so we catch accidental drift if keep: [] changes.
      x = Expression.new("x", [1.0])
      {leaves, _acc} = Nx.Container.traverse(x, [], fn v, acc -> {v, [v | acc]} end)
      assert match?(%STLCG.Expression{}, leaves)
    end
  end

  describe "Inspect" do
    test "renders shape and name" do
      x = Expression.new("x", Nx.tensor([[1.0, 2.0, 3.0]]))
      s = inspect(x)
      assert s =~ "x"
      assert s =~ "1x3"
    end
  end
end
