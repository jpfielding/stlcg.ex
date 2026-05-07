defmodule STLCG.ParityTest do
  @moduledoc """
  Numerical parity against fixtures produced by the pinned upstream stlcg.

  Each fixture advertises a `required_operators` list and a `regime` tag.
  We skip fixtures whose required operators aren't all implemented yet
  (so the suite grows naturally as each operator ticket lands), and we
  apply the tolerance-matrix tolerance for each fixture's regime.

  Run with: `mix test --only parity`
  """
  use ExUnit.Case, async: true

  alias STLCG.Fixtures

  @moduletag :parity

  # --- Operators currently implemented in this port --------------------
  # Keep in sync with ticket closures. A fixture whose required_operators
  # is a subset of this set runs; otherwise it is skipped with a message.

  @implemented MapSet.new([
                 # Filled in by each operator ticket as it lands.
                 # wave-1:
                 :less_than,
                 :greater_than,
                 :equal,
                 :identity,
                 # wave-2:
                 :and,
                 :or,
                 :not,
                 :implies,
                 # wave-3:
                 :always,
                 :eventually,
                 :until,
                 :then,
                 :integral1d
               ])

  setup_all do
    {:ok, ids: Fixtures.list()}
  end

  test "at least one fixture exists on disk", %{ids: ids} do
    assert ids != [],
           "no fixtures found — run `scripts/bootstrap_fixtures.sh` to generate them"
  end

  describe "fixture parity" do
    for id <- Fixtures.list() do
      @tag fixture: id
      test "parity: #{id}" do
        fx = Fixtures.load(unquote(id))

        if Enum.all?(fx.required_operators, &MapSet.member?(@implemented, &1)) do
          actual_trace = STLCG.robustness_trace(fx.formula, fx.inputs, fx.opts)
          actual_rob = STLCG.robustness(fx.formula, fx.inputs, fx.opts)

          {rtol, atol} = Fixtures.tolerance(fx.regime)

          assert_close(actual_trace, fx.expected_trace, rtol, atol, "trace")
          assert_close(actual_rob, fx.expected_robustness, rtol, atol, "robustness")
        else
          missing =
            fx.required_operators
            |> Enum.reject(&MapSet.member?(@implemented, &1))
            |> Enum.map(&to_string/1)
            |> Enum.join(", ")

          ExUnit.configure(capture_log: false)

          IO.puts(
            "  [skip] #{fx.id} requires unimplemented operators: #{missing} — " <>
              "will run once those operators land."
          )
        end
      end
    end
  end

  defp assert_close(actual, expected, rtol, atol, label) do
    unless Nx.shape(actual) == Nx.shape(expected) do
      flunk("""
      shape mismatch on #{label}:
        actual   = #{inspect(Nx.shape(actual))}
        expected = #{inspect(Nx.shape(expected))}
      """)
    end

    diff =
      Nx.subtract(actual, expected)
      |> Nx.abs()
      |> Nx.to_flat_list()
      |> Enum.max()

    expected_flat = Nx.to_flat_list(expected)
    max_expected = Enum.max(Enum.map(expected_flat, &abs/1))
    allowed = atol + rtol * max_expected

    if diff > allowed do
      flunk("""
      parity FAIL on #{label}
        max absolute diff  = #{diff}
        tolerance allowed  = #{allowed}  (rtol=#{rtol}, atol=#{atol})
        actual (flat)      = #{inspect(Nx.to_flat_list(actual))}
        expected (flat)    = #{inspect(expected_flat)}
      """)
    end
  end
end
