defmodule STLCG.Temporal do
  @moduledoc """
  Temporal operators — `Always` (□) and `Eventually` (◇) — with three
  interval kinds: `nil`, `{a, :infinity}`, `{a, b}`.

  ## Semantics (locked to upstream fixtures)

  Upstream `stlcg` implements temporal operators as RNN cells that
  iterate over the signal in forward time order. At each step, the RNN
  maintains a hidden buffer that it **initializes with the first
  predicate value**, not with zeros. The output at position `t` is the
  min (Always) or max (Eventually) over a *past* window of indices
  `t - b, t - b + 1, …, t - a` — with any index `< 0` replaced by the
  first predicate value `pred[0]`.

  Concretely:

  - **`Always(nil)`** = forward cumulative min. `output[t] = min(pred[0..t])`.
  - **`Eventually(nil)`** = forward cumulative max.
  - **`Always({a, :infinity})`** — pad left by `a` copies of `pred[0]`,
    forward cumulative min, slice to original length.
  - **`Always({a, b})`** — pad left by `b` copies of `pred[0]`,
    `Nx.window_min` with window size `steps = b - a + 1`, slice to
    original length.
  - **Eventually** variants mirror Always with min→max and
    cumulative_min→cumulative_max.

  ## Implementation notes

  The padding value is **data-dependent** (it's a slice of the input
  trace, not a constant), so `Nx.pad` is unsuitable. We synthesize the
  padding via `Nx.concatenate` of a tiled first-element slice.

  `Nx.cumulative_min/max` and `Nx.window_min/max` are both differentiable
  under `Nx.Defn`; the `while + put_slice` pattern the Phase 0.5 spike
  ruled out is not used here.
  """

  import Nx.Defn

  # --- Helpers ---------------------------------------------------------

  @doc """
  Left-pad `trace` along the time axis with `n` copies of its first
  element. `n` is a compile-time integer.
  """
  defn left_pad_with_first(trace, opts \\ []) do
    opts = keyword!(opts, n: 0)
    n = opts[:n]
    first = Nx.slice_along_axis(trace, 0, 1, axis: 1)
    padding = tile_time(first, n)
    Nx.concatenate([padding, trace], axis: 1)
  end

  deftransformp tile_time(first, n) do
    Nx.tile(first, [1, n, 1])
  end

  # --- Always kernels --------------------------------------------------

  @doc "Always(nil): forward cumulative min on the time axis."
  defn always_nil_kernel(trace) do
    Nx.cumulative_min(trace, axis: 1)
  end

  @doc """
  Always({a, :infinity}): pad left by `a` copies of `trace[0]`, forward
  cumulative min, slice to original time length.
  """
  defn always_a_inf_kernel(trace, opts \\ []) do
    opts = keyword!(opts, a: 0)
    a = opts[:a]

    padded = left_pad_with_first(trace, n: a)
    cum = Nx.cumulative_min(padded, axis: 1)
    Nx.slice_along_axis(cum, 0, Nx.axis_size(trace, 1), axis: 1)
  end

  @doc """
  Always({a, b}): pad left by `b` copies of `trace[0]`, `Nx.window_min`
  with window size `steps = b - a + 1`, slice to original time length.
  """
  defn always_ab_kernel(trace, opts \\ []) do
    opts = keyword!(opts, a: 0, b: 1)
    a = opts[:a]
    b = opts[:b]
    steps = b - a + 1

    padded = left_pad_with_first(trace, n: b)
    windowed = Nx.window_min(padded, {1, steps, 1}, strides: [1, 1, 1])
    Nx.slice_along_axis(windowed, 0, Nx.axis_size(trace, 1), axis: 1)
  end

  # --- Eventually kernels ----------------------------------------------

  @doc "Eventually(nil): forward cumulative max."
  defn eventually_nil_kernel(trace) do
    Nx.cumulative_max(trace, axis: 1)
  end

  @doc "Eventually({a, :infinity}): pad+cumulative-max+slice."
  defn eventually_a_inf_kernel(trace, opts \\ []) do
    opts = keyword!(opts, a: 0)
    a = opts[:a]

    padded = left_pad_with_first(trace, n: a)
    cum = Nx.cumulative_max(padded, axis: 1)
    Nx.slice_along_axis(cum, 0, Nx.axis_size(trace, 1), axis: 1)
  end

  @doc "Eventually({a, b}): pad+window_max+slice."
  defn eventually_ab_kernel(trace, opts \\ []) do
    opts = keyword!(opts, a: 0, b: 1)
    a = opts[:a]
    b = opts[:b]
    steps = b - a + 1

    padded = left_pad_with_first(trace, n: b)
    windowed = Nx.window_max(padded, {1, steps, 1}, strides: [1, 1, 1])
    Nx.slice_along_axis(windowed, 0, Nx.axis_size(trace, 1), axis: 1)
  end
end

defmodule STLCG.Always do
  @moduledoc """
  Always (□) — per upstream: min over past window with first-value
  left-padding. See `STLCG.Temporal` module docs for the full contract.

  `interval` is one of `nil | {a, :infinity} | {a, b}`.
  """
  defstruct [:subformula, :interval]

  defimpl STLCG.Formula do
    alias STLCG.Temporal

    def robustness_trace(%{interval: nil}, [sub], _opts) do
      Temporal.always_nil_kernel(sub)
    end

    def robustness_trace(%{interval: {a, :infinity}}, [sub], _opts) do
      Temporal.always_a_inf_kernel(sub, a: a)
    end

    def robustness_trace(%{interval: {a, b}}, [sub], _opts) do
      Temporal.always_ab_kernel(sub, a: a, b: b)
    end

    def subformulas(%{subformula: s}), do: [s]
    def operator_tag(_), do: :always
  end
end

defmodule STLCG.Eventually do
  @moduledoc """
  Eventually (◇) — mirror of `STLCG.Always` with min → max.
  """
  defstruct [:subformula, :interval]

  defimpl STLCG.Formula do
    alias STLCG.Temporal

    def robustness_trace(%{interval: nil}, [sub], _opts) do
      Temporal.eventually_nil_kernel(sub)
    end

    def robustness_trace(%{interval: {a, :infinity}}, [sub], _opts) do
      Temporal.eventually_a_inf_kernel(sub, a: a)
    end

    def robustness_trace(%{interval: {a, b}}, [sub], _opts) do
      Temporal.eventually_ab_kernel(sub, a: a, b: b)
    end

    def subformulas(%{subformula: s}), do: [s]
    def operator_tag(_), do: :eventually
  end
end
