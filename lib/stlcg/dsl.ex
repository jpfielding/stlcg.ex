defmodule STLCG.DSL do
  @moduledoc """
  Optional syntax sugar for building `STLCG` formulas.

      use STLCG.DSL

      x = expr "x", [[[0.1], [0.3], [0.5]]]
      c = expr "c", 1.0

      formula = always(x <~ c) &&& eventually({0, 2}, x ~> 0.0)

  This module brings the following bindings into the calling scope:

  ## Formula constructors

    * `expr/2` — `STLCG.Expression.new/2` under a short name.
    * `always/1`, `always/2` — `%STLCG.Always{}`.
    * `eventually/1`, `eventually/2` — `%STLCG.Eventually{}`.
    * `until/2`, `until/3` — `%STLCG.Until{}`.
    * `then_/2`, `then_/3` — `%STLCG.Then{}`. (`then` is a reserved name.)
    * `integral/1`, `integral/2` — `%STLCG.Integral1d{}`.

  ## Operators

    * `<~ 2`   — `%STLCG.LessThan{}`
    * `~> 2`   — `%STLCG.GreaterThan{}`
    * `&&& 2`  — `%STLCG.And{}`
    * `||| 2`  — `%STLCG.Or{}`

  ## Limits

  Elixir's operator table is fixed — we can't overload `<`, `>`, `==`, or
  unary `~` — so `Not` and `Implies` stay as word-form functions
  (`not_/1`, `implies/2`). Kernel's comparison operators retain their
  usual meaning (boolean, not formula-building).

  The macro produces *plain structs*; no protocol dispatch happens at
  macro expansion time. A fixture-backed test asserts every DSL form is
  structurally identical to calling the constructor directly.
  """

  defmacro __using__(_opts) do
    quote do
      import STLCG.DSL
      import Kernel, except: [<~: 2, ~>: 2, &&&: 2, |||: 2]
    end
  end

  # --- Predicate operators ---------------------------------------------

  @doc "LessThan — `x <~ c`."
  defmacro lhs <~ rhs do
    quote do
      %STLCG.LessThan{
        lhs: STLCG.DSL.__extract_name__(unquote(lhs)),
        val: STLCG.DSL.__extract_name_or_value__(unquote(rhs))
      }
    end
  end

  @doc "GreaterThan — `x ~> c`."
  defmacro lhs ~> rhs do
    quote do
      %STLCG.GreaterThan{
        lhs: STLCG.DSL.__extract_name__(unquote(lhs)),
        val: STLCG.DSL.__extract_name_or_value__(unquote(rhs))
      }
    end
  end

  # --- Logical operators -----------------------------------------------

  @doc "And — `φ &&& ψ`."
  defmacro a &&& b do
    quote do: %STLCG.And{lhs: unquote(a), rhs: unquote(b)}
  end

  @doc "Or — `φ ||| ψ`."
  defmacro a ||| b do
    quote do: %STLCG.Or{lhs: unquote(a), rhs: unquote(b)}
  end

  @doc "Not — word form (no overridable prefix operator)."
  def not_(f), do: %STLCG.Not{subformula: f}

  @doc "Implies — word form (no clean glyph)."
  def implies(a, b), do: %STLCG.Implies{lhs: a, rhs: b}

  # --- Temporal operators ----------------------------------------------

  @doc "Expression constructor — `expr \"x\", [[1.0]]`."
  def expr(name, value), do: STLCG.Expression.new(name, value)

  @doc "Always with default interval `nil`."
  def always(formula), do: %STLCG.Always{subformula: formula, interval: nil}

  @doc "Always with explicit interval (`nil | {a, :infinity} | {a, b}`)."
  def always(interval, formula), do: %STLCG.Always{subformula: formula, interval: interval}

  @doc "Eventually with default interval `nil`."
  def eventually(formula), do: %STLCG.Eventually{subformula: formula, interval: nil}

  @doc "Eventually with explicit interval."
  def eventually(interval, formula),
    do: %STLCG.Eventually{subformula: formula, interval: interval}

  @doc "Until with default interval `nil`."
  def until(a, b), do: %STLCG.Until{lhs: a, rhs: b, interval: nil}

  @doc "Until with explicit interval."
  def until(interval, a, b), do: %STLCG.Until{lhs: a, rhs: b, interval: interval}

  @doc "Then with default interval `nil` (word form — `then` is reserved)."
  def then_(a, b), do: %STLCG.Then{lhs: a, rhs: b, interval: nil}

  @doc "Then with explicit interval."
  def then_(interval, a, b), do: %STLCG.Then{lhs: a, rhs: b, interval: interval}

  @doc "Integral1d with default interval `nil`."
  def integral(formula), do: %STLCG.Integral1d{subformula: formula, interval: nil}

  @doc "Integral1d with explicit interval."
  def integral(interval, formula),
    do: %STLCG.Integral1d{subformula: formula, interval: interval}

  # --- Name extraction helpers ------------------------------------------
  #
  # Predicates want their `lhs` to be a variable *name* (String). The
  # DSL accepts either a bare string, an Expression (we use its name),
  # or a number (treated as a literal threshold).

  @doc false
  def __extract_name__(%STLCG.Expression{name: n}), do: n
  def __extract_name__(n) when is_binary(n), do: n

  def __extract_name__(other) do
    raise ArgumentError, "expected STLCG.Expression or string name, got: #{inspect(other)}"
  end

  @doc false
  def __extract_name_or_value__(%STLCG.Expression{name: n}), do: n
  def __extract_name_or_value__(n) when is_binary(n), do: n
  def __extract_name_or_value__(n) when is_number(n), do: n
  def __extract_name_or_value__(%Nx.Tensor{} = t), do: t
end
