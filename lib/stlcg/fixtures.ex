defmodule STLCG.Fixtures do
  @moduledoc """
  Load and validate Python-generated parity fixtures.

  Each fixture is a JSON file produced by `fixtures/gen_fixtures.py` with
  the schema documented in that script. This module

    1. reads a fixture from disk,
    2. reconstructs the expected `Nx.Tensor`s,
    3. lowers the `formula_ast` field into an `STLCG.Formula` struct tree
       (so the Elixir side can be asked to compute the same robustness),
    4. maps each fixture's `regime` to a `{rtol, atol}` tolerance pair.

  The schema validator never evaluates numerics — it catches oracle-side
  bugs without depending on any operator being implemented yet.
  """

  alias STLCG.Expression

  @fixture_dir "fixtures"

  # --- Tolerance matrix (PLAN.md §Parity fixtures) ---------------------

  @tolerances %{
    "hard_f32" => {1.0e-6, 1.0e-7},
    "hard_f64" => {1.0e-10, 1.0e-12},
    "soft_shallow_f32" => {1.0e-5, 1.0e-6},
    "soft_deep_f32" => {1.0e-4, 1.0e-5},
    "agm_f32" => {1.0e-5, 1.0e-6},
    "distributed_f32" => {1.0e-6, 1.0e-7}
  }

  @doc "Return `{rtol, atol}` for a fixture's regime tag."
  @spec tolerance(String.t()) :: {float(), float()}
  def tolerance(regime) when is_map_key(@tolerances, regime), do: Map.fetch!(@tolerances, regime)

  def tolerance(regime),
    do: raise(ArgumentError, "unknown fixture regime #{inspect(regime)}")

  @doc """
  Resolve the fixture directory (relative to the project root, overridable
  via the `STLCG_FIXTURES_DIR` env var for out-of-tree testing).
  """
  @spec fixture_dir() :: String.t()
  def fixture_dir do
    System.get_env("STLCG_FIXTURES_DIR", @fixture_dir)
  end

  @doc "List all fixture ids (reads `index.json`)."
  @spec list() :: [String.t()]
  def list do
    path = Path.join(fixture_dir(), "index.json")

    if File.exists?(path) do
      path
      |> File.read!()
      |> Jason.decode!()
      |> Map.fetch!("fixtures")
    else
      []
    end
  end

  @doc """
  Load a single fixture by id. Raises if the file is malformed.

  The returned map has these pre-processed keys:

    * `:id` — the fixture id (same as `"formula_id"`).
    * `:formula` — the reconstructed `STLCG.Formula` struct.
    * `:inputs` — a map from variable name to `STLCG.Expression`.
    * `:opts` — Elixir keyword options ready for `STLCG.robustness/3`.
    * `:expected_trace` / `:expected_robustness` — `Nx.Tensor`s.
    * `:regime` / `:required_operators` / `:meta` — pass-throughs.
  """
  @spec load(String.t()) :: map()
  def load(id) do
    path = Path.join(fixture_dir(), "#{id}.json")

    raw =
      path
      |> File.read!()
      |> Jason.decode!()

    dtype = dtype_of(raw["dtype"])

    inputs =
      Map.new(raw["inputs"], fn {name, tj} -> {name, Expression.new(name, tensor(tj, dtype))} end)

    %{
      id: raw["formula_id"],
      formula: build_formula(raw["formula_ast"]),
      inputs: inputs,
      opts: opts_of(raw["opts"]),
      dtype: dtype,
      expected_trace: tensor(raw["expected_trace"], dtype),
      expected_robustness: tensor(raw["expected_robustness"], dtype),
      regime: raw["meta"]["regime"],
      required_operators: Enum.map(raw["required_operators"], &String.to_atom/1),
      meta: raw["meta"]
    }
  end

  @doc """
  Given an already-implemented set of operator tags, return the list of
  fixture ids whose `required_operators` are all covered — the set that
  the parity harness should actually run.
  """
  @spec runnable(MapSet.t(atom())) :: [String.t()]
  def runnable(implemented_tags) do
    for id <- list(),
        fx = load_requirements(id),
        MapSet.subset?(fx.required, implemented_tags) do
      id
    end
  end

  @doc """
  Schema-only validation: reconstruct the formula tree and input tensors
  without running any operator. Raises if any required key is missing or
  any tensor's flat length disagrees with its declared shape.
  """
  @spec validate!(String.t()) :: :ok
  def validate!(id) do
    fx = load(id)

    verify_tensor!(fx.expected_trace, "expected_trace")
    verify_tensor!(fx.expected_robustness, "expected_robustness")

    Enum.each(fx.inputs, fn {name, %Expression{value: v}} ->
      verify_tensor!(v, "inputs.#{name}")
    end)

    :ok
  end

  # --- Helpers ---------------------------------------------------------

  defp load_requirements(id) do
    path = Path.join(fixture_dir(), "#{id}.json")
    raw = path |> File.read!() |> Jason.decode!()

    %{
      id: id,
      required: raw["required_operators"] |> Enum.map(&String.to_atom/1) |> MapSet.new()
    }
  end

  defp dtype_of("f32"), do: {:f, 32}
  defp dtype_of("f64"), do: {:f, 64}
  defp dtype_of(t), do: raise(ArgumentError, "unknown fixture dtype #{inspect(t)}")

  defp tensor(%{"shape" => shape, "data" => data}, dtype) do
    data
    |> Nx.tensor(type: dtype)
    |> Nx.reshape(List.to_tuple(shape))
  end

  defp opts_of(opts) do
    [
      pscale: opts["pscale"] || 1.0,
      scale: opts["scale"] || -1.0,
      agm?: opts["agm"] || false,
      distributed?: opts["distributed"] || false
    ]
  end

  defp verify_tensor!(tensor, label) do
    shape_size = tensor |> Nx.shape() |> Tuple.product()
    actual = tensor |> Nx.flatten() |> Nx.size()

    unless shape_size == actual do
      raise "fixture #{label}: shape #{inspect(Nx.shape(tensor))} implies #{shape_size} elements but tensor has #{actual}"
    end
  end

  # --- Formula AST reconstruction --------------------------------------
  # Dispatches on the "op" field. Operators not yet implemented raise
  # `STLCG.Fixtures.UnsupportedOperator` — the fixture harness catches
  # this and skips the fixture rather than failing the build.

  defmodule UnsupportedOperator do
    defexception [:op]

    def message(%{op: op}), do: "fixture requires unsupported operator: #{op}"
  end

  @doc false
  @spec build_formula(map()) :: struct()
  def build_formula(%{"op" => "LessThan", "lhs" => lhs, "val" => val}) do
    %STLCG.LessThan{lhs: lhs, val: val}
  end

  def build_formula(%{"op" => "GreaterThan", "lhs" => lhs, "val" => val}) do
    %STLCG.GreaterThan{lhs: lhs, val: val}
  end

  def build_formula(%{"op" => "Equal", "lhs" => lhs, "val" => val}) do
    %STLCG.Equal{lhs: lhs, val: val}
  end

  def build_formula(%{"op" => "Identity", "name" => name}) do
    %STLCG.Identity{name: name}
  end

  def build_formula(%{"op" => "Always", "interval" => i, "subformula" => sub}) do
    %STLCG.Always{interval: interval(i), subformula: build_formula(sub)}
  end

  def build_formula(%{"op" => "Eventually", "interval" => i, "subformula" => sub}) do
    %STLCG.Eventually{interval: interval(i), subformula: build_formula(sub)}
  end

  def build_formula(%{"op" => "Until", "interval" => i, "lhs" => lhs, "rhs" => rhs}) do
    %STLCG.Until{
      interval: interval(i),
      lhs: build_formula(lhs),
      rhs: build_formula(rhs)
    }
  end

  def build_formula(%{"op" => "Then", "interval" => i, "lhs" => lhs, "rhs" => rhs}) do
    %STLCG.Then{
      interval: interval(i),
      lhs: build_formula(lhs),
      rhs: build_formula(rhs)
    }
  end

  def build_formula(%{"op" => "Integral1d", "interval" => i, "subformula" => sub}) do
    %STLCG.Integral1d{interval: interval(i), subformula: build_formula(sub)}
  end

  def build_formula(%{"op" => op}) do
    raise UnsupportedOperator, op: op
  end

  defp interval(nil), do: nil
  defp interval(%{"lo" => lo, "hi" => "infinity"}), do: {lo, :infinity}
  defp interval(%{"lo" => lo, "hi" => hi}) when is_integer(hi), do: {lo, hi}
end
