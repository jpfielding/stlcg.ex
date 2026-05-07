defmodule STLCG.DSLTest do
  use ExUnit.Case, async: true

  describe "predicate operators" do
    test "<~ builds a LessThan with lhs name and threshold" do
      use STLCG.DSL
      x = expr("x", 1.0)
      assert x <~ 1.5 == %STLCG.LessThan{lhs: "x", val: 1.5}
    end

    test "~> builds a GreaterThan" do
      use STLCG.DSL
      x = expr("x", 1.0)
      assert x ~> 0.5 == %STLCG.GreaterThan{lhs: "x", val: 0.5}
    end

    test "operator accepts a string name on the rhs (for threshold variables)" do
      use STLCG.DSL
      x = expr("x", 1.0)
      c = expr("c", 0.5)
      assert x <~ c == %STLCG.LessThan{lhs: "x", val: "c"}
    end
  end

  describe "logical operators" do
    test "&&& builds an And" do
      use STLCG.DSL
      a = %STLCG.LessThan{lhs: "x", val: 1.0}
      b = %STLCG.GreaterThan{lhs: "x", val: 0.0}
      assert (a &&& b) == %STLCG.And{lhs: a, rhs: b}
    end

    test "||| builds an Or" do
      use STLCG.DSL
      a = %STLCG.LessThan{lhs: "x", val: 1.0}
      b = %STLCG.GreaterThan{lhs: "x", val: 0.0}
      assert (a ||| b) == %STLCG.Or{lhs: a, rhs: b}
    end

    test "not_/1 and implies/2 are word-form" do
      use STLCG.DSL
      a = %STLCG.LessThan{lhs: "x", val: 1.0}
      assert not_(a) == %STLCG.Not{subformula: a}
      b = %STLCG.GreaterThan{lhs: "x", val: 0.0}
      assert implies(a, b) == %STLCG.Implies{lhs: a, rhs: b}
    end
  end

  describe "temporal helpers" do
    test "always/1 and always/2" do
      use STLCG.DSL
      a = %STLCG.LessThan{lhs: "x", val: 1.0}
      assert always(a) == %STLCG.Always{subformula: a, interval: nil}
      assert always({1, 3}, a) == %STLCG.Always{subformula: a, interval: {1, 3}}
    end

    test "eventually/1 and eventually/2" do
      use STLCG.DSL
      a = %STLCG.LessThan{lhs: "x", val: 1.0}
      assert eventually(a) == %STLCG.Eventually{subformula: a, interval: nil}

      assert eventually({0, 5}, a) ==
               %STLCG.Eventually{subformula: a, interval: {0, 5}}
    end

    test "until/2 and until/3" do
      use STLCG.DSL
      a = %STLCG.LessThan{lhs: "x", val: 1.0}
      b = %STLCG.GreaterThan{lhs: "y", val: 0.0}
      assert until(a, b) == %STLCG.Until{lhs: a, rhs: b, interval: nil}
      assert until({1, 2}, a, b) == %STLCG.Until{lhs: a, rhs: b, interval: {1, 2}}
    end

    test "then_/2 and then_/3 (word form: then is reserved)" do
      use STLCG.DSL
      a = %STLCG.LessThan{lhs: "x", val: 1.0}
      b = %STLCG.GreaterThan{lhs: "y", val: 0.0}
      assert then_(a, b) == %STLCG.Then{lhs: a, rhs: b, interval: nil}
    end

    test "integral/1 and integral/2" do
      use STLCG.DSL
      a = %STLCG.LessThan{lhs: "x", val: 1.0}
      assert integral(a) == %STLCG.Integral1d{subformula: a, interval: nil}

      assert integral({1, 2}, a) ==
               %STLCG.Integral1d{subformula: a, interval: {1, 2}}
    end
  end

  describe "end-to-end: a DSL-built formula computes robustness" do
    test "Always(x <= c) AND Eventually(x > 0)" do
      use STLCG.DSL

      x =
        expr("x", Nx.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5]]]))

      c = expr("c", Nx.tensor(0.6))

      f = always(x <~ c) &&& eventually(x ~> 0.0)

      r = STLCG.robustness(f, %{"x" => x, "c" => c})
      assert Nx.shape(r) == {1, 1, 1}
    end
  end

  test "DSL output matches direct struct construction exactly" do
    use STLCG.DSL

    x = expr("x", 1.0)
    c = expr("c", 0.5)

    dsl_form = always({0, 3}, x <~ c) &&& eventually({1, 2}, x ~> 0.0)

    direct =
      %STLCG.And{
        lhs: %STLCG.Always{
          subformula: %STLCG.LessThan{lhs: "x", val: "c"},
          interval: {0, 3}
        },
        rhs: %STLCG.Eventually{
          subformula: %STLCG.GreaterThan{lhs: "x", val: 0.0},
          interval: {1, 2}
        }
      }

    assert dsl_form == direct
  end
end
