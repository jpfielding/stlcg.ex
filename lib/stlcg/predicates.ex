defmodule STLCG.Predicates do
  @moduledoc """
  Predicate operators — the leaves of every STL formula tree.

  Each predicate reads a named signal from the `inputs` map, compares it
  to a threshold, and returns a robustness trace of the same shape as
  the signal.

  | Operator                 | Robustness trace                    |
  |--------------------------|-------------------------------------|
  | `LessThan(lhs, val)`     | `(val - trace) * pscale`            |
  | `GreaterThan(lhs, val)`  | `(trace - val) * pscale`            |
  | `Equal(lhs, val)`        | `-abs(trace - val) * pscale`        |
  | `Identity(name)`         | `trace * pscale`                    |

  See `docs/semantics.md` §4 for the semantic contract.
  """

  import Nx.Defn

  # --- defn kernels (shape-preserving, differentiable) ------------------

  @doc "Kernel: `(val - trace) * pscale`."
  defn less_than_kernel(trace, val, opts \\ []) do
    opts = keyword!(opts, pscale: 1.0)
    (val - trace) * opts[:pscale]
  end

  @doc "Kernel: `(trace - val) * pscale`."
  defn greater_than_kernel(trace, val, opts \\ []) do
    opts = keyword!(opts, pscale: 1.0)
    (trace - val) * opts[:pscale]
  end

  @doc "Kernel: `-abs(trace - val) * pscale`."
  defn equal_kernel(trace, val, opts \\ []) do
    opts = keyword!(opts, pscale: 1.0)
    -Nx.abs(trace - val) * opts[:pscale]
  end

  @doc "Kernel: `trace * pscale` (pass-through)."
  defn identity_kernel(trace, opts \\ []) do
    opts = keyword!(opts, pscale: 1.0)
    trace * opts[:pscale]
  end

  # --- Helpers for Formula implementations -----------------------------

  @doc false
  def fetch_tensor(_inputs, %STLCG.Expression{value: v}), do: v

  def fetch_tensor(inputs, name) when is_binary(name) do
    case Map.fetch!(inputs, name) do
      %STLCG.Expression{value: v} -> v
      %Nx.Tensor{} = t -> t
    end
  end

  @doc false
  def fetch_scalar(_inputs, %Nx.Tensor{} = t), do: t
  def fetch_scalar(_inputs, n) when is_number(n), do: Nx.tensor(n)
  def fetch_scalar(_inputs, %STLCG.Expression{value: v}), do: v

  def fetch_scalar(inputs, name) when is_binary(name) do
    case Map.fetch!(inputs, name) do
      %STLCG.Expression{value: v} -> v
      %Nx.Tensor{} = t -> t
      n when is_number(n) -> Nx.tensor(n)
    end
  end
end

defmodule STLCG.LessThan do
  @moduledoc "Predicate `x < val`. Robustness `(val - x) * pscale`."
  defstruct [:lhs, :val, pscale: 1.0]

  defimpl STLCG.Formula do
    alias STLCG.Predicates

    def robustness_trace(%{lhs: lhs, val: val}, [], opts) do
      inputs = Keyword.fetch!(opts, :inputs)
      pscale = Keyword.fetch!(opts, :pscale)
      x = Predicates.fetch_tensor(inputs, lhs)
      c = Predicates.fetch_scalar(inputs, val) |> Nx.as_type(Nx.type(x))
      Predicates.less_than_kernel(x, c, pscale: pscale)
    end

    def subformulas(_), do: []
    def operator_tag(_), do: :less_than
  end
end

defmodule STLCG.GreaterThan do
  @moduledoc "Predicate `x > val`. Robustness `(x - val) * pscale`."
  defstruct [:lhs, :val, pscale: 1.0]

  defimpl STLCG.Formula do
    alias STLCG.Predicates

    def robustness_trace(%{lhs: lhs, val: val}, [], opts) do
      inputs = Keyword.fetch!(opts, :inputs)
      pscale = Keyword.fetch!(opts, :pscale)
      x = Predicates.fetch_tensor(inputs, lhs)
      c = Predicates.fetch_scalar(inputs, val) |> Nx.as_type(Nx.type(x))
      Predicates.greater_than_kernel(x, c, pscale: pscale)
    end

    def subformulas(_), do: []
    def operator_tag(_), do: :greater_than
  end
end

defmodule STLCG.Equal do
  @moduledoc "Predicate `x == val`. Robustness `-abs(x - val) * pscale` (≤ 0)."
  defstruct [:lhs, :val, pscale: 1.0]

  defimpl STLCG.Formula do
    alias STLCG.Predicates

    def robustness_trace(%{lhs: lhs, val: val}, [], opts) do
      inputs = Keyword.fetch!(opts, :inputs)
      pscale = Keyword.fetch!(opts, :pscale)
      x = Predicates.fetch_tensor(inputs, lhs)
      c = Predicates.fetch_scalar(inputs, val) |> Nx.as_type(Nx.type(x))
      Predicates.equal_kernel(x, c, pscale: pscale)
    end

    def subformulas(_), do: []
    def operator_tag(_), do: :equal
  end
end

defmodule STLCG.Identity do
  @moduledoc "Identity predicate. Robustness `x * pscale`."
  defstruct [:name, pscale: 1.0]

  defimpl STLCG.Formula do
    alias STLCG.Predicates

    def robustness_trace(%{name: name}, [], opts) do
      inputs = Keyword.fetch!(opts, :inputs)
      pscale = Keyword.fetch!(opts, :pscale)
      x = Predicates.fetch_tensor(inputs, name)
      Predicates.identity_kernel(x, pscale: pscale)
    end

    def subformulas(_), do: []
    def operator_tag(_), do: :identity
  end
end
