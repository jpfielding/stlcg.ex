defmodule STLCG.Spike.LessThan do
  @moduledoc false
  # Minimal LessThan spike — used only by the Phase 0.5 autodiff-composition
  # spike test. The real predicate implementation lands under ticket #5.

  import Nx.Defn

  defstruct [:lhs, :val, pscale: 1.0]

  @doc """
  Kernel: robustness of `x < c` is `(c - x) * pscale`.
  Shape-preserving: input `x` of shape `{batch, time, features}` returns
  same shape. `c` is a scalar or broadcast-compatible tensor.
  """
  defn kernel(x, c, opts \\ []) do
    opts = keyword!(opts, pscale: 1.0)
    (c - x) * opts[:pscale]
  end

  defimpl STLCG.Formula do
    alias STLCG.Spike.LessThan

    def robustness_trace(%{lhs: lhs, val: val}, [], opts) do
      inputs = Keyword.fetch!(opts, :inputs)
      pscale = Keyword.fetch!(opts, :pscale)

      x = fetch_tensor(inputs, lhs)
      c = fetch_tensor(inputs, val)

      LessThan.kernel(x, c, pscale: pscale)
    end

    def subformulas(_), do: []
    def operator_tag(_), do: :spike_less_than

    defp fetch_tensor(_inputs, %STLCG.Expression{} = e), do: e.value
    defp fetch_tensor(inputs, name) when is_binary(name), do: get_value(inputs, name)

    defp get_value(inputs, name) do
      case Map.fetch!(inputs, name) do
        %STLCG.Expression{value: v} -> v
        %Nx.Tensor{} = t -> t
      end
    end
  end
end

defmodule STLCG.Spike.Always do
  @moduledoc false
  # Minimal Always(subformula) spike for interval=nil — reverse-time
  # cumulative min. Uses Nx.cumulative_min (trivially differentiable)
  # so this spike purely tests the walker's kernel-composition, not the
  # while-loop/put_slice combo that ticket #8 will need for bounded
  # intervals.

  import Nx.Defn

  defstruct [:subformula]

  @doc """
  For a trace of shape {batch, time, features}, compute at each t the
  minimum over trace[:, t..time-1, :] via a reverse cumulative min on
  the time axis.
  """
  defn kernel(trace) do
    Nx.cumulative_min(trace, axis: 1, reverse: true)
  end

  defimpl STLCG.Formula do
    alias STLCG.Spike.Always

    def robustness_trace(%{}, [sub_trace], _opts), do: Always.kernel(sub_trace)
    def subformulas(%{subformula: s}), do: [s]
    def operator_tag(_), do: :spike_always
  end
end
