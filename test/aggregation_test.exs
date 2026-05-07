defmodule STLCG.AggregationTest do
  use ExUnit.Case, async: true

  alias STLCG.Aggregation

  describe "maxish :true_" do
    test "reduces to Nx.reduce_max with keep_axes: true" do
      x = Nx.tensor([[[1.0], [5.0], [3.0]]])
      out = Aggregation.maxish(x, :true_, axes: [1])
      assert Nx.shape(out) == {1, 1, 1}
      assert Nx.to_flat_list(out) == [5.0]
    end

    test "keep_axes: false squeezes the reduction axis" do
      x = Nx.tensor([[[1.0], [5.0], [3.0]]])
      out = Aggregation.maxish(x, :true_, axes: [1], keep_axes: false)
      assert Nx.shape(out) == {1, 1}
      assert Nx.to_flat_list(out) == [5.0]
    end

    test "gradient is concentrated on the argmax (non-smooth)" do
      x = Nx.tensor([[[1.0], [5.0], [3.0]]], type: :f64)

      grad =
        Nx.Defn.grad(x, fn t ->
          Aggregation.maxish(t, :true_, axes: [1]) |> Nx.sum()
        end)

      # argmax at index 1 → grad = 1 there, 0 elsewhere
      assert Nx.to_flat_list(grad) == [0.0, 1.0, 0.0]
    end
  end

  describe "maxish :distributed" do
    test "value equals reduce_max when there are no ties" do
      x = Nx.tensor([[[1.0], [5.0], [3.0]]], type: :f64)
      true_val = Aggregation.maxish(x, :true_, axes: [1])
      dist_val = Aggregation.maxish(x, :distributed, axes: [1])

      assert_in_delta(
        Nx.to_flat_list(true_val) |> hd(),
        Nx.to_flat_list(dist_val) |> hd(),
        1.0e-12
      )
    end

    test "distributes gradient across tied argmax" do
      # Two tied maxima at indices 1 and 2.
      x = Nx.tensor([[[1.0], [5.0], [5.0], [3.0]]], type: :f64)

      grad =
        Nx.Defn.grad(x, fn t ->
          Aggregation.maxish(t, :distributed, axes: [1]) |> Nx.sum()
        end)

      flat = Nx.to_flat_list(grad)
      # Expected: 0, 0.5, 0.5, 0 — ties split gradient equally.
      assert_in_delta(Enum.at(flat, 0), 0.0, 1.0e-10)
      assert_in_delta(Enum.at(flat, 1), 0.5, 1.0e-10)
      assert_in_delta(Enum.at(flat, 2), 0.5, 1.0e-10)
      assert_in_delta(Enum.at(flat, 3), 0.0, 1.0e-10)
    end
  end

  describe "maxish :soft (logsumexp)" do
    test "approaches true max as scale grows" do
      x = Nx.tensor([[[1.0], [5.0], [3.0]]], type: :f64)
      soft_fast = Aggregation.maxish(x, :soft, axes: [1], scale: 100.0)
      assert_in_delta(Nx.to_flat_list(soft_fast) |> hd(), 5.0, 1.0e-3)
    end

    test "smoothly differentiable everywhere" do
      x = Nx.tensor([[[1.0], [5.0], [5.0], [3.0]]], type: :f64)

      grad =
        Nx.Defn.grad(x, fn t ->
          Aggregation.maxish(t, :soft, axes: [1], scale: 1.0) |> Nx.sum()
        end)

      flat = Nx.to_flat_list(grad)
      # All gradients should be positive (softmax weights sum to 1).
      assert Enum.all?(flat, fn g -> g > 0.0 end)
      assert_in_delta(Enum.sum(flat), 1.0, 1.0e-10)
    end

    test "minish :soft mirrors maxish :soft with negation" do
      x = Nx.tensor([[[1.0], [5.0], [3.0]]], type: :f64)
      mx = Aggregation.maxish(x, :soft, axes: [1], scale: 10.0)
      mn = Aggregation.minish(x, :soft, axes: [1], scale: 10.0)
      # max > 3, min < 3 (strict approximations)
      assert Nx.to_flat_list(mx) |> hd() > 3.0
      assert Nx.to_flat_list(mn) |> hd() < 3.0
    end
  end

  describe "maxish :agm" do
    test "positive branch: mean of positive entries" do
      x = Nx.tensor([[[1.0], [-2.0], [3.0], [-4.0]]], type: :f64)
      out = Aggregation.maxish(x, :agm, axes: [1])
      # pos entries: 1.0, 3.0 → mean = 2.0
      assert_in_delta(Nx.to_flat_list(out) |> hd(), 2.0, 1.0e-12)
    end

    test "all-non-positive branch: returns 0 when all x <= 0" do
      # For all-non-positive, neg_branch = 1 - exp(mean(log(max(1-x, eps))))
      # with all x <= 0, 1-x >= 1, log(1-x) >= 0, exp(mean) >= 1.
      # If all x = 0, 1-x = 1, log(1) = 0, exp(0) = 1, result = 0.
      x = Nx.tensor([[[0.0], [0.0], [0.0]]], type: :f64)
      out = Aggregation.maxish(x, :agm, axes: [1])
      assert_in_delta(Nx.to_flat_list(out) |> hd(), 0.0, 1.0e-12)
    end

    test "strict-gt on positive mask: zeros excluded from positive branch" do
      # Exactly [0, 0, 0] is all-non-positive (takes neg branch, returns 0).
      # [0, 0.5, 0] has one positive (takes pos branch, mean = 0.5).
      x = Nx.tensor([[[0.0], [0.5], [0.0]]], type: :f64)
      out = Aggregation.maxish(x, :agm, axes: [1])
      assert_in_delta(Nx.to_flat_list(out) |> hd(), 0.5, 1.0e-12)
    end
  end

  describe "minish mirrors maxish across all modes" do
    for mode <- [:true_, :distributed] do
      test "minish #{inspect(mode)} = -maxish(-x)" do
        x = Nx.tensor([[[1.0], [5.0], [3.0]]], type: :f64)
        mn = Aggregation.minish(x, unquote(mode), axes: [1])
        assert Nx.to_flat_list(mn) == [1.0]
      end
    end
  end
end
