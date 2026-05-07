defmodule STLCG.UntilThen do
  @moduledoc """
  `Until` (φ U ψ) and `Then` (φ T ψ) — past-directed per upstream.

  ## Semantics (locked to upstream fixtures)

  ```
  ρ_{φ U ψ}(t) = max over  t' ≤ t  of  min(ρψ(t'),  min over t'' ∈ [t', t] of ρφ(t''))
  ρ_{φ T ψ}(t) = max over  t' ≤ t  of  min(ρψ(t'),  max over t'' ∈ [t', t] of ρφ(t''))
  ```

  "There is a past witness time `t'` at which ψ fired, and between `t'`
  and `t` inclusive, φ held (Until) or eventually held (Then)."

  For empty-interval cases (bounded intervals where no valid `t'` exists),
  upstream fills with a large negative sentinel. We use the same.

  ## Interval support

  - `interval = nil` — unrestricted past witness (`t' ∈ [0, t]`).
  - `interval = {a, b}` — witness restricted to `t' ∈ [t - b, t - a]`
    (bounded lookback). Entries with an empty window get a large negative
    sentinel.
  - `interval = {a, :infinity}` — witness in `[0, t - a]`; same
    construction as `nil` but cap the witness range.

  ## Implementation

  Outer loop over `t'` in plain Elixir; each iteration is a `defn`
  expression. All per-t' tensors are stacked and reduced with
  `Nx.reduce_max` — a single differentiable graph.
  """

  @neg_large -1.0e6

  # --- Public dispatch --------------------------------------------------

  def until_trace(phi_trace, psi_trace, interval \\ nil) do
    build_trace(phi_trace, psi_trace, interval, :min)
  end

  def then_trace(phi_trace, psi_trace, interval \\ nil) do
    build_trace(phi_trace, psi_trace, interval, :max)
  end

  # --- Internals --------------------------------------------------------

  # `mode` is :min for Until (φ held throughout) or :max for Then
  # (φ eventually held).
  defp build_trace(phi_trace, psi_trace, interval, mode) do
    {_b, t, _f} = Nx.shape(phi_trace)
    shape = Nx.shape(phi_trace)

    cols =
      for t_prime <- 0..(t - 1) do
        column(phi_trace, psi_trace, t_prime, t, shape, mode, interval)
      end

    # Stack on a new trailing axis and reduce max over it.
    stacked = Nx.stack(cols, axis: -1)
    Nx.reduce_max(stacked, axes: [tuple_size(Nx.shape(stacked)) - 1])
  end

  defp column(phi_trace, psi_trace, t_prime, t, shape, mode, interval) do
    # 1. φ aggregate over [t', t] for each t (or -∞ when t < t').
    agg = phi_aggregate(phi_trace, t_prime, t, mode)

    # 2. ψ at t', broadcast to shape {B, T, F}.
    psi_at_tprime = Nx.slice_along_axis(psi_trace, t_prime, 1, axis: 1)
    psi_broadcast = Nx.broadcast(psi_at_tprime, shape)

    # 3. inner(t', t) = min(ψ(t'), agg(t', t)).
    inner = Nx.min(psi_broadcast, agg)

    # 4. Apply interval mask: if interval constrains the witness time,
    # positions where t' is outside the allowed window get -∞ so they
    # lose in the outer max.
    apply_interval_mask(inner, t_prime, t, interval, shape)
  end

  # For a given t', compute a tensor `agg` of shape {B, T, F} where
  # agg[b, t, f] = (min or max) over φ[b, t_prime..t, f] if t >= t_prime
  # else @neg_large.
  defp phi_aggregate(phi_trace, t_prime, t, mode) do
    phi_suffix = Nx.slice_along_axis(phi_trace, t_prime, t - t_prime, axis: 1)

    cum =
      case mode do
        :min -> Nx.cumulative_min(phi_suffix, axis: 1)
        :max -> Nx.cumulative_max(phi_suffix, axis: 1)
      end

    if t_prime > 0 do
      pad_shape = put_elem(Nx.shape(phi_trace), 1, t_prime)
      pad_tensor = Nx.broadcast(Nx.tensor(@neg_large, type: Nx.type(phi_trace)), pad_shape)
      Nx.concatenate([pad_tensor, cum], axis: 1)
    else
      cum
    end
  end

  # Interval = nil: no masking (all t' ≤ t eligible).
  defp apply_interval_mask(inner, _t_prime, _t, nil, _shape), do: inner

  # Interval = {a, :infinity}: witness t' ≥ 0 with t - t' ≥ a.
  # (No upper bound constraint, so the `t_out >= a_max` strict-bound
  # requirement does not apply — any t_out ≥ a is allowed.)
  defp apply_interval_mask(inner, t_prime, t_total, {a, :infinity}, shape) do
    mask_rows(inner, t_prime, t_total, a, :infinity, shape)
  end

  # Interval = {a, b}: witness t' in [t - b, t - a], and t ≥ b
  # (upstream requires the earliest valid witness index to be ≥ 0).
  defp apply_interval_mask(inner, t_prime, t_total, {a, b}, shape) do
    mask_rows(inner, t_prime, t_total, a, b, shape)
  end

  # Zero out (mask to -∞) rows where t_out - t_prime is outside [a_min, a_max].
  # Upstream additionally requires `t_out >= a_max` so the witness window's
  # earliest position is non-negative. If t_out < a_max, no witness is
  # valid at this row.
  defp mask_rows(inner, t_prime, t_total, a_min, a_max, shape) do
    a_max_cap = if a_max == :infinity, do: 0, else: a_max

    mask_list =
      for t_out <- 0..(t_total - 1) do
        offset = t_out - t_prime

        within_interval =
          offset >= a_min and (a_max == :infinity or offset <= a_max)

        window_in_bounds = t_out >= a_max_cap

        if within_interval and window_in_bounds, do: 1.0, else: 0.0
      end

    # Broadcast the mask to full shape.
    {b, t, f} = shape
    mask_1d = Nx.tensor(mask_list, type: Nx.type(inner)) |> Nx.reshape({1, t, 1})
    mask = Nx.broadcast(mask_1d, {b, t, f})

    # Where mask = 1 keep inner, where 0 use @neg_large.
    Nx.select(Nx.equal(mask, 1.0), inner, Nx.tensor(@neg_large, type: Nx.type(inner)))
  end
end

defmodule STLCG.Until do
  @moduledoc "Until — `φ U_I ψ`. See `STLCG.UntilThen` for semantics."
  defstruct [:lhs, :rhs, interval: nil, overlap: true]

  defimpl STLCG.Formula do
    alias STLCG.UntilThen

    def robustness_trace(%{interval: interval}, [phi, psi], _opts) do
      UntilThen.until_trace(phi, psi, interval)
    end

    def subformulas(%{lhs: a, rhs: b}), do: [a, b]
    def operator_tag(_), do: :until
  end
end

defmodule STLCG.Then do
  @moduledoc "Then — `φ T_I ψ`. See `STLCG.UntilThen`."
  defstruct [:lhs, :rhs, interval: nil, overlap: true]

  defimpl STLCG.Formula do
    alias STLCG.UntilThen

    def robustness_trace(%{interval: interval}, [phi, psi], _opts) do
      UntilThen.then_trace(phi, psi, interval)
    end

    def subformulas(%{lhs: a, rhs: b}), do: [a, b]
    def operator_tag(_), do: :then
  end
end
