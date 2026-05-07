defmodule STLCG.AutodiffTest do
  @moduledoc """
  Consolidated gradient-validation suite. One test per operator, each
  compares `Nx.Defn.grad` to a central finite difference.

  Run with: `mix test --only autodiff`
  """
  use ExUnit.Case, async: true

  alias STLCG.{
    Always,
    And,
    Eventually,
    Expression,
    GreaterThan,
    Implies,
    Integral1d,
    LessThan,
    Not,
    Or,
    Then,
    Until
  }

  @moduletag :autodiff

  @h 1.0e-5
  @tol 1.0e-4

  defp finite_diff(fun, c_val) do
    plus = fun.(Nx.tensor(c_val + @h, type: :f64)) |> Nx.to_number()
    minus = fun.(Nx.tensor(c_val - @h, type: :f64)) |> Nx.to_number()
    (plus - minus) / (2 * @h)
  end

  defp check_grad(formula, inputs_fn, c_val) do
    fun = fn c ->
      STLCG.robustness(formula, inputs_fn.(c)) |> Nx.sum()
    end

    fd = finite_diff(fun, c_val)
    ad = Nx.Defn.grad(Nx.tensor(c_val, type: :f64), fun) |> Nx.to_number()
    {ad, fd}
  end

  # Signal common to all tests: 8-step monotone-increasing trace.
  defp x do
    Expression.new(
      "x",
      Nx.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8]]], type: :f64)
    )
  end

  defp y do
    Expression.new(
      "y",
      Nx.tensor([[[-0.3], [0.1], [-0.2], [0.4], [-0.1], [0.2], [-0.3], [0.0]]], type: :f64)
    )
  end

  defp inputs_x(c), do: %{"x" => x(), "c" => c}
  defp inputs_xy(c), do: %{"x" => x(), "y" => y(), "c" => c}

  # ---------------------------------------------------------------------

  test "LessThan grad matches finite-diff" do
    f = %LessThan{lhs: "x", val: "c"}
    {ad, fd} = check_grad(f, &inputs_x/1, 0.5)
    assert_in_delta(ad, fd, @tol)
  end

  test "GreaterThan grad matches finite-diff" do
    f = %GreaterThan{lhs: "x", val: "c"}
    {ad, fd} = check_grad(f, &inputs_x/1, 0.5)
    assert_in_delta(ad, fd, @tol)
  end

  test "And grad matches finite-diff" do
    f = %And{
      lhs: %LessThan{lhs: "x", val: "c"},
      rhs: %GreaterThan{lhs: "x", val: 0.0}
    }

    {ad, fd} = check_grad(f, &inputs_x/1, 0.9)
    assert_in_delta(ad, fd, @tol)
  end

  test "Or grad matches finite-diff" do
    f = %Or{
      lhs: %LessThan{lhs: "x", val: "c"},
      rhs: %GreaterThan{lhs: "x", val: 0.0}
    }

    {ad, fd} = check_grad(f, &inputs_x/1, 0.3)
    assert_in_delta(ad, fd, @tol)
  end

  test "Not grad matches finite-diff" do
    f = %Not{subformula: %LessThan{lhs: "x", val: "c"}}
    {ad, fd} = check_grad(f, &inputs_x/1, 0.5)
    assert_in_delta(ad, fd, @tol)
  end

  test "Implies grad matches finite-diff" do
    f = %Implies{
      lhs: %GreaterThan{lhs: "x", val: 0.5},
      rhs: %LessThan{lhs: "x", val: "c"}
    }

    {ad, fd} = check_grad(f, &inputs_x/1, 0.9)
    assert_in_delta(ad, fd, @tol)
  end

  test "Always(nil) grad matches finite-diff" do
    f = %Always{subformula: %LessThan{lhs: "x", val: "c"}, interval: nil}
    {ad, fd} = check_grad(f, &inputs_x/1, 1.5)
    assert_in_delta(ad, fd, @tol)
  end

  test "Always({1, 2}) grad matches finite-diff" do
    f = %Always{subformula: %LessThan{lhs: "x", val: "c"}, interval: {1, 2}}
    {ad, fd} = check_grad(f, &inputs_x/1, 1.5)
    assert_in_delta(ad, fd, @tol)
  end

  test "Always({1, :infinity}) grad matches finite-diff" do
    f = %Always{subformula: %LessThan{lhs: "x", val: "c"}, interval: {1, :infinity}}
    {ad, fd} = check_grad(f, &inputs_x/1, 1.5)
    assert_in_delta(ad, fd, @tol)
  end

  test "Eventually(nil) grad matches finite-diff" do
    f = %Eventually{subformula: %LessThan{lhs: "x", val: "c"}, interval: nil}
    {ad, fd} = check_grad(f, &inputs_x/1, 0.5)
    assert_in_delta(ad, fd, @tol)
  end

  test "Eventually({1, 2}) grad matches finite-diff" do
    f = %Eventually{subformula: %LessThan{lhs: "x", val: "c"}, interval: {1, 2}}
    {ad, fd} = check_grad(f, &inputs_x/1, 0.5)
    assert_in_delta(ad, fd, @tol)
  end

  test "Until(nil) grad matches finite-diff" do
    f = %Until{
      lhs: %LessThan{lhs: "x", val: "c"},
      rhs: %GreaterThan{lhs: "y", val: 0.0}
    }

    {ad, fd} = check_grad(f, &inputs_xy/1, 0.5)
    assert_in_delta(ad, fd, @tol)
  end

  test "Then(nil) grad matches finite-diff" do
    # Use a c_val off any predicate boundary (x elements are 0.1..0.8) so
    # neither AD nor FD crosses an argmax discontinuity.
    f = %Then{
      lhs: %LessThan{lhs: "x", val: "c"},
      rhs: %GreaterThan{lhs: "y", val: 0.0}
    }

    {ad, fd} = check_grad(f, &inputs_xy/1, 0.15)
    assert_in_delta(ad, fd, @tol)
  end

  test "Integral1d grad matches finite-diff" do
    f = %Integral1d{subformula: %LessThan{lhs: "x", val: "c"}, interval: nil}
    {ad, fd} = check_grad(f, &inputs_x/1, 0.5)
    assert_in_delta(ad, fd, @tol)
  end

  test "deeply nested formula grad matches finite-diff" do
    # Always(Implies(GreaterThan(x, 0.3), Eventually([1,3], LessThan(x, c))))
    f = %Always{
      subformula: %Implies{
        lhs: %GreaterThan{lhs: "x", val: 0.3},
        rhs: %Eventually{
          subformula: %LessThan{lhs: "x", val: "c"},
          interval: {1, 3}
        }
      },
      interval: nil
    }

    {ad, fd} = check_grad(f, &inputs_x/1, 0.9)
    assert_in_delta(ad, fd, @tol)
  end
end
