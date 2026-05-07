defmodule STLCG.Integral do
  @moduledoc """
  `Integral1d` — sum of the predicate trace over an interval window.

  ## Semantics (locked to upstream)

  Same past-window + first-value-padding pattern as `Always`, with
  `sum` replacing `min`:

  ```
  ρ_{∫ϕ}(t) = sum over  i ∈ [t - b, t - a]  of  ρ'ϕ(i)
  ```

  where `ρ'ϕ(i) = ρϕ(i)` for `i ≥ 0` and `ρ'ϕ(i) = ρϕ(0)` for `i < 0`.

  - `interval = nil` ≡ cumulative sum forward on the raw trace:
    `ρ_{∫ϕ}(t) = sum(ρϕ[0..t])`.
  - `interval = {a, :infinity}`: pad left by `a` copies of `ρϕ(0)`,
    cumulative sum, slice to original length.
  - `interval = {a, b}`: pad left by `b` copies of `ρϕ(0)`,
    `Nx.window_sum` of size `b - a + 1`, slice to original length.
  """

  import Nx.Defn

  defn nil_kernel(trace) do
    Nx.cumulative_sum(trace, axis: 1)
  end

  defn a_inf_kernel(trace, opts \\ []) do
    opts = keyword!(opts, a: 0)
    a = opts[:a]

    padded = STLCG.Temporal.left_pad_with_first(trace, n: a)
    cum = Nx.cumulative_sum(padded, axis: 1)
    Nx.slice_along_axis(cum, 0, Nx.axis_size(trace, 1), axis: 1)
  end

  defn ab_kernel(trace, opts \\ []) do
    opts = keyword!(opts, a: 0, b: 1)
    a = opts[:a]
    b = opts[:b]
    steps = b - a + 1

    padded = STLCG.Temporal.left_pad_with_first(trace, n: b)
    windowed = Nx.window_sum(padded, {1, steps, 1}, strides: [1, 1, 1])
    Nx.slice_along_axis(windowed, 0, Nx.axis_size(trace, 1), axis: 1)
  end
end

defmodule STLCG.Integral1d do
  @moduledoc """
  `∫ ϕ` over a past window. See `STLCG.Integral` for the contract.
  """
  defstruct [:subformula, :interval]

  defimpl STLCG.Formula do
    alias STLCG.Integral

    def robustness_trace(%{interval: nil}, [sub], _opts) do
      Integral.nil_kernel(sub)
    end

    def robustness_trace(%{interval: {a, :infinity}}, [sub], _opts) do
      Integral.a_inf_kernel(sub, a: a)
    end

    def robustness_trace(%{interval: {a, b}}, [sub], _opts) do
      Integral.ab_kernel(sub, a: a, b: b)
    end

    def subformulas(%{subformula: s}), do: [s]
    def operator_tag(_), do: :integral1d
  end
end
