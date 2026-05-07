defmodule STLCG.Logical do
  @moduledoc """
  Binary logical operators — `And`, `Or`, `Not`, `Implies`.

  Each operator stacks its two sub-traces along a new leading axis and
  aggregates with `STLCG.Aggregation.minish/maxish` (for And/Or) or
  composes from existing primitives (Not/Implies). See
  `docs/semantics.md` §3.

  | Operator           | Robustness trace              |
  |--------------------|-------------------------------|
  | `And(φ, ψ)`        | `min(ρφ, ρψ)`                 |
  | `Or(φ, ψ)`         | `max(ρφ, ρψ)`                 |
  | `Not(φ)`           | `-ρφ`                         |
  | `Implies(φ, ψ)`    | `max(-ρφ, ρψ)`                |
  """

  import Nx.Defn

  @doc """
  Stack two traces on a new axis-0 then reduce with Minish.
  (And uses the mode from opts.)
  """
  defn and_kernel(a, b) do
    Nx.min(a, b)
  end

  @doc "Or = Nx.max; element-wise."
  defn or_kernel(a, b) do
    Nx.max(a, b)
  end

  @doc "Not = negate."
  defn not_kernel(a) do
    Nx.negate(a)
  end

  @doc "Implies(a, b) = max(-a, b) — element-wise."
  defn implies_kernel(a, b) do
    Nx.max(Nx.negate(a), b)
  end
end

defmodule STLCG.And do
  @moduledoc "Logical And: `ρ_{a ∧ b}(t) = min(ρ_a(t), ρ_b(t))`."
  defstruct [:lhs, :rhs]

  defimpl STLCG.Formula do
    def robustness_trace(_op, [a, b], _opts), do: STLCG.Logical.and_kernel(a, b)
    def subformulas(%{lhs: a, rhs: b}), do: [a, b]
    def operator_tag(_), do: :and
  end
end

defmodule STLCG.Or do
  @moduledoc "Logical Or: `ρ_{a ∨ b}(t) = max(ρ_a(t), ρ_b(t))`."
  defstruct [:lhs, :rhs]

  defimpl STLCG.Formula do
    def robustness_trace(_op, [a, b], _opts), do: STLCG.Logical.or_kernel(a, b)
    def subformulas(%{lhs: a, rhs: b}), do: [a, b]
    def operator_tag(_), do: :or
  end
end

defmodule STLCG.Not do
  @moduledoc "Logical Not: `ρ_{¬a}(t) = -ρ_a(t)`."
  defstruct [:subformula]

  defimpl STLCG.Formula do
    def robustness_trace(_op, [a], _opts), do: STLCG.Logical.not_kernel(a)
    def subformulas(%{subformula: s}), do: [s]
    def operator_tag(_), do: :not
  end
end

defmodule STLCG.Implies do
  @moduledoc "Logical Implies: `ρ_{a ⇒ b}(t) = max(-ρ_a(t), ρ_b(t))`."
  defstruct [:lhs, :rhs]

  defimpl STLCG.Formula do
    def robustness_trace(_op, [a, b], _opts), do: STLCG.Logical.implies_kernel(a, b)
    def subformulas(%{lhs: a, rhs: b}), do: [a, b]
    def operator_tag(_), do: :implies
  end
end
