defmodule STLCG.Aggregation do
  @moduledoc """
  `Maxish` and `Minish` reductions — the smooth / non-smooth aggregation
  primitives used by every temporal and logical operator.

  Four modes (see `docs/semantics.md` §6):

    * `:true_` — exact `Nx.reduce_max/min`.
    * `:distributed` — tied-argmax average via `stop_grad(mask)/sum(mask)`.
      Gradient distributes uniformly across tied entries.
    * `:soft` — log-sum-exp with a positive `scale`; `scale → ∞` recovers
      true max. Minish negates input and output.
    * `:agm` — sign-partitioned arithmetic-geometric mean per upstream.

  The mode is **selected in plain Elixir before the `defn` call**; each
  mode is its own `defn` kernel so the tensor shapes stay statically
  known. This avoids runtime branching inside `defn` (which is allowed
  for `cond` on scalars but surprisingly expensive for reductions).

  The public entry points are `maxish/3` and `minish/3`. Temporal and
  logical operators choose a mode and call these from plain Elixir,
  which dispatches into the corresponding defn kernel.

  Axes:

  All kernels accept `opts[:axes]` — the axes to reduce over
  (same convention as `Nx.reduce_max`). `opts[:keep_axes]` mirrors Nx's
  keyword of the same name. For a formula whose robustness trace has
  shape `{batch, time, features}`, typical reductions are:

    * temporal operators reduce over the time axis (`axes: [1]`).
    * logical operators stack sub-formulas on a *new* axis first, then
      reduce over it.
  """

  import Nx.Defn

  @type mode :: :true_ | :distributed | :soft | :agm

  # --- Public dispatch (plain Elixir) -----------------------------------

  @doc """
  Compute the max-like aggregation of `x` over `opts[:axes]`.

  `mode` selects the semantic. `opts` is passed through to the chosen
  kernel (see kernel doc for accepted keys).
  """
  @spec maxish(Nx.Tensor.t(), mode(), keyword()) :: Nx.Tensor.t()
  def maxish(x, mode, opts \\ [])
  def maxish(x, :true_, opts), do: maxish_true(x, opts)
  def maxish(x, :distributed, opts), do: maxish_distributed(x, opts)
  def maxish(x, :soft, opts), do: maxish_soft(x, opts)
  def maxish(x, :agm, opts), do: maxish_agm(x, opts)

  @doc "Min-like aggregation (see `maxish/3`)."
  @spec minish(Nx.Tensor.t(), mode(), keyword()) :: Nx.Tensor.t()
  def minish(x, mode, opts \\ [])

  def minish(x, mode, opts) do
    # Minish = negate -> maxish -> negate. Works for every mode.
    Nx.negate(maxish(Nx.negate(x), mode, opts))
  end

  # --- defn kernels -----------------------------------------------------

  @doc "Maxish :true_. `Nx.reduce_max` with caller-supplied axes/keep_axes."
  defn maxish_true(x, opts \\ []) do
    opts = keyword!(opts, axes: [1], keep_axes: true)
    Nx.reduce_max(x, axes: opts[:axes], keep_axes: opts[:keep_axes])
  end

  @doc """
  Maxish :distributed. Computes the true max but averages across tied
  argmax entries so gradient is distributed uniformly.

  `mask = stop_grad(Nx.equal(x, reduce_max(x)))`
  `sum(x * mask) / sum(mask)`
  """
  defn maxish_distributed(x, opts \\ []) do
    opts = keyword!(opts, axes: [1], keep_axes: true)
    axes = opts[:axes]
    keep = opts[:keep_axes]

    m = Nx.reduce_max(x, axes: axes, keep_axes: true)
    mask = stop_grad(Nx.equal(x, m))
    num = Nx.sum(x * mask, axes: axes, keep_axes: true)
    den = Nx.sum(mask, axes: axes, keep_axes: true)
    out = num / den

    if keep, do: out, else: squeeze_axes(out, axes)
  end

  @doc """
  Maxish :soft — `logsumexp(x * scale) / scale`, differentiable smooth max.
  Requires `opts[:scale] > 0`.
  """
  defn maxish_soft(x, opts \\ []) do
    opts = keyword!(opts, axes: [1], keep_axes: true, scale: 1.0)
    axes = opts[:axes]
    keep = opts[:keep_axes]
    scale = opts[:scale]

    # logsumexp(z) = log(sum(exp(z - m))) + m, with m = max(z), for numerical stability.
    z = x * scale
    m = Nx.reduce_max(z, axes: axes, keep_axes: true)
    shifted = z - m
    lse = Nx.log(Nx.sum(Nx.exp(shifted), axes: axes, keep_axes: true)) + m
    out = lse / scale

    if keep, do: out, else: squeeze_axes(out, axes)
  end

  @doc """
  Maxish :agm — sign-partitioned arithmetic-geometric mean (upstream's
  experimental smooth-max). Per-window reduction on caller axes.

  `pos_mask = x > 0`
  `pos_mean = sum(x*pos_mask) / max(sum(pos_mask), 1)`
  `neg_branch = 1 - exp(mean(log(max(1-x, tiny_eps))))`
  `out = select(any_positive, pos_mean, neg_branch)`

  Caller precondition: values must be `< 1` in windows where the neg
  branch fires. Not enforced at runtime (defn forbids data-dependent
  host branching); see `docs/semantics.md` §6.4.
  """
  defn maxish_agm(x, opts \\ []) do
    opts = keyword!(opts, axes: [1], keep_axes: true)
    axes = opts[:axes]
    keep = opts[:keep_axes]

    tiny_eps = 1.0e-12
    one = 1.0

    pos_mask = Nx.greater(x, 0)
    pos_count = Nx.sum(pos_mask, axes: axes, keep_axes: true)
    pos_sum = Nx.sum(x * pos_mask, axes: axes, keep_axes: true)
    pos_mean = pos_sum / Nx.max(pos_count, 1)

    # log(max(1-x, tiny_eps)) — guards against log(0) when x == 1.
    neg_inner = Nx.log(Nx.max(one - x, tiny_eps))
    neg_branch = one - Nx.exp(Nx.mean(neg_inner, axes: axes, keep_axes: true))

    any_pos = Nx.sum(pos_mask, axes: axes, keep_axes: true) |> Nx.greater(0)
    out = Nx.select(any_pos, pos_mean, neg_branch)

    if keep, do: out, else: squeeze_axes(out, axes)
  end

  # --- Helpers ----------------------------------------------------------

  deftransformp squeeze_axes(t, axes) do
    Nx.squeeze(t, axes: axes)
  end
end
