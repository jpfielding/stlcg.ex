defmodule STLCG.SpikeAutodiffTest do
  @moduledoc """
  Phase 0.5 spike — validates that per-operator defn kernels dispatched by
  a plain-Elixir tree walker compose into a single differentiable Nx graph.

  If this test fails, the architecture is wrong. Every subsequent operator
  assumes this composition works.
  """
  use ExUnit.Case, async: true

  alias STLCG.Spike.{Always, LessThan}

  @tag :autodiff
  test "Always(x < c) composes through walker into a differentiable graph" do
    # Trace: [0.1, 0.3, 0.5, 0.7, 0.2] → max = 0.7, so robustness of
    # Always(x < c) evaluated at t=0 is (c - 0.7). At the terminal
    # time slice that Always returns, we see min over all t≥0 which is
    # still (c - 0.7) at t=0 but (c - 0.2) at t=4. The `robustness/3`
    # facade returns the terminal-time slice, which is the trailing
    # value — for a singleton suffix, just (c - 0.2).
    #
    # To make the test meaningful, we evaluate at t=0 instead by taking
    # the full robustness_trace and reading index 0.

    x_tensor = Nx.tensor([[[0.1], [0.3], [0.5], [0.7], [0.2]]], type: :f64)

    formula = %Always{
      subformula: %LessThan{lhs: "x", val: "c"}
    }

    # --- Forward pass ---------------------------------------------------
    c_value = 1.0
    c_tensor = Nx.tensor(c_value, type: :f64)
    inputs = %{"x" => x_tensor, "c" => c_tensor}

    trace = STLCG.robustness_trace(formula, inputs)
    rob_at_t0 = trace |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.to_flat_list() |> hd()

    # Always(x < c) at t=0 = min over suffix of (c - x) = c - max(x) = 1.0 - 0.7 = 0.3
    assert_in_delta(rob_at_t0, 0.3, 1.0e-10)

    # --- Gradient via Nx.Defn.grad -------------------------------------
    grad_fn = fn c ->
      inp = %{"x" => x_tensor, "c" => c}
      # Read robustness at t=0 as a scalar so grad is well-defined.
      STLCG.robustness_trace(formula, inp)
      |> Nx.slice_along_axis(0, 1, axis: 1)
      |> Nx.sum()
    end

    grad_c = Nx.Defn.grad(c_tensor, grad_fn)
    grad_val = Nx.to_number(grad_c)

    # d/dc of (c - max(x)) = 1
    assert_in_delta(grad_val, 1.0, 1.0e-8)

    # --- Finite-difference check ---------------------------------------
    h = 1.0e-5

    f_plus =
      grad_fn.(Nx.tensor(c_value + h, type: :f64)) |> Nx.to_number()

    f_minus =
      grad_fn.(Nx.tensor(c_value - h, type: :f64)) |> Nx.to_number()

    fd = (f_plus - f_minus) / (2 * h)
    assert_in_delta(fd, grad_val, 1.0e-5)
  end

  @tag :autodiff
  test "gradient of terminal-time robustness (singleton suffix) wrt c is 1.0" do
    # At t = time - 1 the suffix is length 1, so Always reduces to
    # the single-step predicate: (c - x[time-1]). Gradient wrt c is 1.
    x_tensor = Nx.tensor([[[0.1], [0.3], [0.9]]], type: :f64)

    formula = %Always{subformula: %LessThan{lhs: "x", val: "c"}}

    c_tensor = Nx.tensor(2.0, type: :f64)

    fun = fn c ->
      inp = %{"x" => x_tensor, "c" => c}
      STLCG.robustness(formula, inp) |> Nx.sum()
    end

    assert_in_delta(fun.(c_tensor) |> Nx.to_number(), 2.0 - 0.9, 1.0e-10)

    grad = Nx.Defn.grad(c_tensor, fun) |> Nx.to_number()
    assert_in_delta(grad, 1.0, 1.0e-8)
  end

  @tag :autodiff
  test "gradient wrt x tensor also flows through" do
    # d/dx_k of Always(x < c) at t=0 is -1 on the argmax of x, 0 elsewhere.
    x_tensor = Nx.tensor([[[0.1], [0.3], [0.5], [0.7], [0.2]]], type: :f64)
    c_tensor = Nx.tensor(1.0, type: :f64)

    formula = %Always{subformula: %LessThan{lhs: "x", val: "c"}}

    fun = fn x ->
      inp = %{"x" => x, "c" => c_tensor}

      STLCG.robustness_trace(formula, inp)
      |> Nx.slice_along_axis(0, 1, axis: 1)
      |> Nx.sum()
    end

    grad_x = Nx.Defn.grad(x_tensor, fun)
    flat = Nx.to_flat_list(grad_x)

    # Non-smooth min → argmin gradient is concentrated on the maximum-x index
    # (since robustness = c - x, the min is at max x = 0.7 at index 3).
    assert_in_delta(Enum.at(flat, 3), -1.0, 1.0e-8)

    # The other entries should be 0.
    others = List.delete_at(flat, 3)

    assert Enum.all?(others, fn v -> abs(v) < 1.0e-8 end),
           "expected zero gradient on non-argmin indices, got #{inspect(others)}"
  end
end
