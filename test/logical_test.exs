defmodule STLCG.LogicalTest do
  use ExUnit.Case, async: true

  alias STLCG.{And, Equal, Expression, GreaterThan, Implies, LessThan, Not, Or}

  setup do
    x = Expression.new("x", Nx.tensor([[[0.2], [0.5], [0.8]]]))
    {:ok, x: x, inputs: %{"x" => x}}
  end

  describe "And" do
    test "min of two predicate traces", ctx do
      # LessThan(x, 1.0) → c - x = [0.8, 0.5, 0.2]
      # GreaterThan(x, 0.0) → x - c = [0.2, 0.5, 0.8]
      # And → min = [0.2, 0.5, 0.2]
      f = %And{lhs: %LessThan{lhs: "x", val: 1.0}, rhs: %GreaterThan{lhs: "x", val: 0.0}}
      trace = STLCG.robustness_trace(f, ctx.inputs)

      assert Nx.to_flat_list(trace)
             |> Enum.zip([0.2, 0.5, 0.2])
             |> Enum.all?(fn {a, b} -> abs(a - b) < 1.0e-6 end)
    end

    test "operator_tags walks the tree", ctx do
      _ = ctx
      f = %And{lhs: %LessThan{lhs: "x", val: 1.0}, rhs: %GreaterThan{lhs: "x", val: 0.0}}
      assert STLCG.operator_tags(f) == MapSet.new([:and, :less_than, :greater_than])
    end
  end

  describe "Or" do
    test "max of two predicate traces", ctx do
      f = %Or{lhs: %LessThan{lhs: "x", val: 0.3}, rhs: %GreaterThan{lhs: "x", val: 0.7}}
      # lt = 0.3 - x = [0.1, -0.2, -0.5]
      # gt = x - 0.7 = [-0.5, -0.2, 0.1]
      # max = [0.1, -0.2, 0.1]
      trace = STLCG.robustness_trace(f, ctx.inputs)

      assert Nx.to_flat_list(trace)
             |> Enum.zip([0.1, -0.2, 0.1])
             |> Enum.all?(fn {a, b} -> abs(a - b) < 1.0e-6 end)
    end
  end

  describe "Not" do
    test "negates the sub-trace", ctx do
      f = %Not{subformula: %LessThan{lhs: "x", val: 1.0}}
      trace = STLCG.robustness_trace(f, ctx.inputs)
      # -(1 - x) = [-0.8, -0.5, -0.2]
      assert Nx.to_flat_list(trace)
             |> Enum.zip([-0.8, -0.5, -0.2])
             |> Enum.all?(fn {a, b} -> abs(a - b) < 1.0e-6 end)
    end
  end

  describe "Implies" do
    test "max(-a, b) — robustness of material implication", ctx do
      # a: GreaterThan(x, 0.5) → x - 0.5 = [-0.3, 0.0, 0.3]
      # b: LessThan(x, 0.9)    → 0.9 - x = [0.7, 0.4, 0.1]
      # implies = max(-a, b) = max([0.3, 0, -0.3], [0.7, 0.4, 0.1]) = [0.7, 0.4, 0.1]
      f = %Implies{lhs: %GreaterThan{lhs: "x", val: 0.5}, rhs: %LessThan{lhs: "x", val: 0.9}}
      trace = STLCG.robustness_trace(f, ctx.inputs)

      assert Nx.to_flat_list(trace)
             |> Enum.zip([0.7, 0.4, 0.1])
             |> Enum.all?(fn {a, b} -> abs(a - b) < 1.0e-6 end)
    end
  end

  describe "Equal combined with Not" do
    test "robustness of Not(Equal) is positive when x ≠ val", ctx do
      f = %Not{subformula: %Equal{lhs: "x", val: 0.5}}
      # Equal = -|x - 0.5| = [-0.3, 0.0, -0.3]
      # Not = |x - 0.5| = [0.3, 0.0, 0.3]
      trace = STLCG.robustness_trace(f, ctx.inputs)

      assert Nx.to_flat_list(trace)
             |> Enum.zip([0.3, 0.0, 0.3])
             |> Enum.all?(fn {a, b} -> abs(a - b) < 1.0e-6 end)
    end
  end

  describe "gradient" do
    @tag :autodiff
    test "flows through And into both sides" do
      x = Expression.new("x", Nx.tensor([[[0.2], [0.5], [0.8]]], type: :f64))
      c = Nx.tensor(1.0, type: :f64)

      f = %And{
        lhs: %LessThan{lhs: "x", val: "c"},
        rhs: %GreaterThan{lhs: "x", val: 0.0}
      }

      fun = fn c_tensor ->
        inp = %{"x" => x, "c" => c_tensor}
        STLCG.robustness_trace(f, inp) |> Nx.sum()
      end

      g = Nx.Defn.grad(c, fun) |> Nx.to_number()
      # x = [0.2, 0.5, 0.8], c = 1
      # a = c - x = [0.8, 0.5, 0.2]   (∂/∂c = 1 everywhere)
      # b = x     = [0.2, 0.5, 0.8]   (∂/∂c = 0 everywhere)
      # min(a, b): t=0 picks b (dc=0), t=1 is a tie (Nx.min splits dc=0.5),
      # t=2 picks a (dc=1). Total = 1.5.
      assert_in_delta(g, 1.5, 1.0e-6)
    end
  end
end
