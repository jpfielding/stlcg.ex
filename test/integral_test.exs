defmodule STLCG.IntegralTest do
  @moduledoc """
  Analytic tests for `STLCG.Integral1d`.

  Upstream's `Integral1d` is broken on modern torch (torch>=2 rejects
  float window sizes in its conv-based implementation), so we validate
  against hand-derived results instead of Python-generated fixtures.
  The semantics are locked by `docs/semantics.md` §5 (same past-window
  pattern as Always, with `sum` replacing `min`).
  """
  use ExUnit.Case, async: true

  alias STLCG.{Expression, Integral1d, LessThan}

  # Oracle: O5_pred = 0.0 - O5 = [-0.3, 0.1, -0.2, 0.4, -0.1]
  # Using signal O5 and predicate LessThan(x, 0):
  setup do
    x = Expression.new("x", Nx.tensor([[[0.3], [-0.1], [0.2], [-0.4], [0.1]]]))
    {:ok, x: x, inputs: %{"x" => x}}
  end

  test "interval=nil → forward cumulative sum", ctx do
    f = %Integral1d{subformula: %LessThan{lhs: "x", val: 0.0}, interval: nil}
    trace = STLCG.robustness_trace(f, ctx.inputs)

    # pred = [-0.3, 0.1, -0.2, 0.4, -0.1]
    # cumulative sum forward:
    #   [-0.3, -0.2, -0.4, 0.0, -0.1]
    expected = [-0.3, -0.2, -0.4, 0.0, -0.1]

    assert_trace_close(trace, expected)
  end

  test "interval={a, b} → sliding-window sum over past", ctx do
    f = %Integral1d{subformula: %LessThan{lhs: "x", val: 0.0}, interval: {1, 2}}
    trace = STLCG.robustness_trace(f, ctx.inputs)

    # pred = [-0.3, 0.1, -0.2, 0.4, -0.1]
    # Window size 2, looking 1-2 steps back with left-pad = pred[0] = -0.3.
    # padded = [-0.3, -0.3, -0.3, 0.1, -0.2, 0.4, -0.1]  (length 7)
    # window_sum size 2:
    #   0..1: -0.6
    #   1..2: -0.6
    #   2..3: -0.2
    #   3..4: -0.1
    #   4..5:  0.2
    #   5..6:  0.3
    # first 5: [-0.6, -0.6, -0.2, -0.1, 0.2]
    expected = [-0.6, -0.6, -0.2, -0.1, 0.2]

    assert_trace_close(trace, expected)
  end

  test "interval={a, :infinity} → pad + cumulative sum + slice", ctx do
    f = %Integral1d{subformula: %LessThan{lhs: "x", val: 0.0}, interval: {1, :infinity}}
    trace = STLCG.robustness_trace(f, ctx.inputs)

    # pred = [-0.3, 0.1, -0.2, 0.4, -0.1]
    # Pad 1 copy of -0.3 → padded = [-0.3, -0.3, 0.1, -0.2, 0.4, -0.1]
    # cumulative sum = [-0.3, -0.6, -0.5, -0.7, -0.3, -0.4]
    # first 5 = [-0.3, -0.6, -0.5, -0.7, -0.3]
    expected = [-0.3, -0.6, -0.5, -0.7, -0.3]

    assert_trace_close(trace, expected)
  end

  test "gradient flows through Integral1d", _ctx do
    x = Expression.new("x", Nx.tensor([[[0.3], [-0.1], [0.2]]], type: :f64))

    f = %Integral1d{subformula: %LessThan{lhs: "x", val: "c"}, interval: nil}

    fun = fn c ->
      inp = %{"x" => x, "c" => c}
      STLCG.robustness(f, inp) |> Nx.sum()
    end

    c = Nx.tensor(0.5, type: :f64)
    g = Nx.Defn.grad(c, fun) |> Nx.to_number()

    # robustness at final time = cumulative sum of (c - x[0..2]) = 3*c - sum(x[0..2])
    # derivative wrt c = 3
    assert_in_delta(g, 3.0, 1.0e-10)
  end

  defp assert_trace_close(trace, expected, tol \\ 1.0e-5) do
    actual = Nx.to_flat_list(trace)

    assert length(actual) == length(expected),
           "length mismatch: got #{length(actual)}, expected #{length(expected)}"

    pairs = Enum.zip(actual, expected)

    bad =
      pairs
      |> Enum.with_index()
      |> Enum.filter(fn {{a, e}, _i} -> abs(a - e) > tol end)

    assert bad == [], """
    mismatched entries (tol=#{tol}):
    actual   = #{inspect(actual)}
    expected = #{inspect(expected)}
    bad      = #{inspect(bad)}
    """
  end
end
