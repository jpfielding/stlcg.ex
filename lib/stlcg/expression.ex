defmodule STLCG.Expression do
  @moduledoc """
  A named signal expression.

  `%STLCG.Expression{name: String.t(), value: Nx.Tensor.t()}` pairs a
  symbolic name (used by `Inspect` and the DSL) with the actual numerical
  tensor that flows through `Nx.Defn` kernels.

  ## `Nx.Container` semantics

  `Expression` **derives `Nx.Container` with `containers: [:value], keep: []`**.
  That means:

  - Only `:value` is treated as a tensor leaf — gradients flow through it.
  - `:name` is intentionally *not* kept across `defn` calls; it is formula
    metadata consumed by plain Elixir (Inspect, DSL, error messages).
    Keeping the name would churn `defn`'s compile cache unnecessarily
    when users construct structurally identical formulas with different
    variable names.

  The upshot: before calling into `defn`, the formula walker extracts
  `.value` tensors. The names are preserved on the Elixir side for
  debugging/printing.

  ## Functional API

  Arithmetic and comparison helpers build new `Expression` structs with
  composed names. They are *not* the DSL — see `STLCG.DSL` for operator
  sugar (`<~`, `~>`, `&&&`, etc.) that produces formula structs.

      iex> x = STLCG.Expression.new("x", Nx.tensor([1.0, 2.0, 3.0]))
      iex> y = STLCG.Expression.new("y", Nx.tensor([0.5, 1.0, 1.5]))
      iex> z = STLCG.Expression.add(x, y)
      iex> z.name
      "(x + y)"
      iex> Nx.to_flat_list(z.value)
      [1.5, 3.0, 4.5]
  """

  @derive {Nx.Container, containers: [:value], keep: []}
  defstruct name: "_", value: nil

  @type t :: %__MODULE__{name: String.t(), value: Nx.Tensor.t()}

  @doc """
  Builds an `Expression` from a name and an `Nx.Tensor` (or something
  convertible by `Nx.tensor/1`).

      iex> e = STLCG.Expression.new("x", [1.0, 2.0])
      iex> e.name
      "x"
      iex> Nx.to_flat_list(e.value)
      [1.0, 2.0]
  """
  @spec new(String.t(), Nx.Tensor.t() | number() | list()) :: t()
  def new(name, value) when is_binary(name) do
    %__MODULE__{name: name, value: Nx.tensor(value)}
  end

  @doc "Returns the tensor value of `expr`."
  @spec value(t()) :: Nx.Tensor.t()
  def value(%__MODULE__{value: v}), do: v

  @doc "Returns the name of `expr`."
  @spec name(t()) :: String.t()
  def name(%__MODULE__{name: n}), do: n

  # --- Arithmetic (compose names, delegate numerics to Nx) -------------

  @doc "Element-wise addition. `a + b`."
  @spec add(t() | number(), t() | number()) :: t()
  def add(a, b), do: binop(a, b, "+", &Nx.add/2)

  @doc "Element-wise subtraction. `a - b`."
  @spec sub(t() | number(), t() | number()) :: t()
  def sub(a, b), do: binop(a, b, "-", &Nx.subtract/2)

  @doc "Element-wise multiplication. `a * b`."
  @spec mul(t() | number(), t() | number()) :: t()
  def mul(a, b), do: binop(a, b, "*", &Nx.multiply/2)

  @doc "Element-wise division. `a / b`."
  @spec divide(t() | number(), t() | number()) :: t()
  def divide(a, b), do: binop(a, b, "/", &Nx.divide/2)

  @doc "Unary negation."
  @spec neg(t()) :: t()
  def neg(%__MODULE__{name: n, value: v}) do
    %__MODULE__{name: "-#{n}", value: Nx.negate(v)}
  end

  # --- Comparison helpers ------------------------------------------------
  #
  # These return `Expression`s whose `value` is a numeric difference (not
  # a boolean). They are *convenience* — the canonical way to build a
  # formula predicate is `STLCG.Predicates.less_than/2` etc., which will
  # land in ticket #5.

  @doc """
  Returns the numeric residue `b - a` as an `Expression` — positive when
  `a < b`. Suitable for feeding into a `LessThan` predicate.
  """
  @spec lt_residue(t() | number(), t() | number()) :: t()
  def lt_residue(a, b), do: binop(b, a, "-", &Nx.subtract/2)

  @doc "Returns `a - b` as an `Expression` — positive when `a > b`."
  @spec gt_residue(t() | number(), t() | number()) :: t()
  def gt_residue(a, b), do: binop(a, b, "-", &Nx.subtract/2)

  # --- Helpers -----------------------------------------------------------

  defp binop(%__MODULE__{} = a, %__MODULE__{} = b, op, fun) do
    %__MODULE__{
      name: "(#{a.name} #{op} #{b.name})",
      value: fun.(a.value, b.value)
    }
  end

  defp binop(%__MODULE__{} = a, b, op, fun) when is_number(b) do
    %__MODULE__{
      name: "(#{a.name} #{op} #{b})",
      value: fun.(a.value, b)
    }
  end

  defp binop(a, %__MODULE__{} = b, op, fun) when is_number(a) do
    %__MODULE__{
      name: "(#{a} #{op} #{b.name})",
      value: fun.(a, b.value)
    }
  end

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(%STLCG.Expression{name: name, value: v}, opts) do
      shape = Nx.shape(v) |> Tuple.to_list() |> Enum.join("x")

      concat([
        "#Expr<",
        string(name),
        ", shape=",
        string(shape),
        ", dtype=",
        to_doc(Nx.type(v), opts),
        ">"
      ])
    end
  end
end
