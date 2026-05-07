#!/usr/bin/env python3
"""Generate canonical robustness fixtures from the pinned upstream stlcg.

For each formula × trace × aggregation-mode combination we run upstream
stlcg, capture the robustness trace and terminal-time robustness, then
dump a JSON document that the Elixir `parity_test.exs` harness can load.

Schema (see PLAN.md):

    {
      "formula_id": "always_lt_nil_true",
      "formula_ast": {...},               # JSON-ified struct tree
      "required_operators": [...],        # atoms the Elixir side needs
      "inputs": {                         # named tensors fed to the formula
        "x": {"shape": [1, 5, 1], "data": [...]}
      },
      "opts": {
        "pscale": 1.0, "scale": -1.0, "agm": false, "distributed": false
      },
      "dtype": "f32",
      "backend": "pytorch-cpu-fp32",
      "expected_trace": {"shape": [1, 5, 1], "data": [...]},
      "expected_robustness": {"shape": [1, 1], "data": [...]},
      "meta": {
        "depth": 1, "aggregation_modes": ["hard"], "regime": "hard_f32"
      }
    }

Only a handful of fixtures land in this initial cut — enough to gate
ticket #5 (Predicates) and ticket #6 (Maxish/Minish). More fixtures are
added alongside each operator ticket.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

import stlcg
from stlcg import Expression


# --- Oracle traces from docs/semantics.md §9 --------------------------
# Shape convention: {batch=1, time=T, features=1} — matches the port.

ORACLES: Dict[str, np.ndarray] = {
    "O1": np.array([[[0.0]]], dtype=np.float32),
    "O2": np.array([[[0.5], [-0.3], [0.1]]], dtype=np.float32),
    "O3": np.array([[[1.0], [1.0], [1.0], [1.0], [1.0]]], dtype=np.float32),
    "O4": np.array(
        [[[-0.2], [-0.1], [0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]],
        dtype=np.float32,
    ),
    "O5": np.array([[[0.3], [-0.1], [0.2], [-0.4], [0.1]]], dtype=np.float32),
}


def _tensor_json(t: torch.Tensor) -> Dict[str, Any]:
    """Serialize a torch tensor as {shape, data} for JSON."""
    arr = t.detach().cpu().numpy()
    return {
        "shape": list(arr.shape),
        "data": arr.reshape(-1).tolist(),
    }


def _np_json(a: np.ndarray) -> Dict[str, Any]:
    return {"shape": list(a.shape), "data": a.reshape(-1).tolist()}


def _dtype_string(t: torch.Tensor) -> str:
    """Map torch dtype to the short code used in fixtures."""
    return {
        torch.float32: "f32",
        torch.float64: "f64",
    }.get(t.dtype, str(t.dtype))


def _evaluate(formula, inputs, scale=-1.0, pscale=1.0, agm=False, distributed=False):
    """Run a stlcg formula and return (trace, robustness) numpy arrays."""
    trace = formula.robustness_trace(
        inputs, scale=scale, pscale=pscale, agm=agm, distributed=distributed
    )
    rob = formula.robustness(
        inputs, scale=scale, pscale=pscale, agm=agm, distributed=distributed
    )
    return trace, rob


# --- Fixture builders per operator -------------------------------------


def fx_predicate_less_than(oracle_id: str, x: np.ndarray, c: float) -> Dict[str, Any]:
    """LessThan(x, c) on a named oracle trace."""
    xt = torch.tensor(x, requires_grad=False)
    ct = torch.tensor(c, dtype=torch.float32, requires_grad=False)
    formula = stlcg.LessThan(lhs="x", val=ct)
    trace, rob = _evaluate(formula, xt)

    return {
        "formula_id": f"less_than_{oracle_id}_c{c}",
        "formula_ast": {
            "op": "LessThan",
            "lhs": "x",
            "val": float(c),
        },
        "required_operators": ["less_than"],
        "inputs": {"x": _np_json(x)},
        "opts": {"pscale": 1.0, "scale": -1.0, "agm": False, "distributed": False},
        "dtype": "f32",
        "backend": "pytorch-cpu-fp32",
        "expected_trace": _tensor_json(trace),
        "expected_robustness": _tensor_json(rob),
        "meta": {
            "depth": 1,
            "aggregation_modes": ["hard"],
            "regime": "hard_f32",
            "oracle": oracle_id,
        },
    }


def fx_predicate_greater_than(oracle_id: str, x: np.ndarray, c: float) -> Dict[str, Any]:
    xt = torch.tensor(x, requires_grad=False)
    ct = torch.tensor(c, dtype=torch.float32, requires_grad=False)
    formula = stlcg.GreaterThan(lhs="x", val=ct)
    trace, rob = _evaluate(formula, xt)

    return {
        "formula_id": f"greater_than_{oracle_id}_c{c}",
        "formula_ast": {"op": "GreaterThan", "lhs": "x", "val": float(c)},
        "required_operators": ["greater_than"],
        "inputs": {"x": _np_json(x)},
        "opts": {"pscale": 1.0, "scale": -1.0, "agm": False, "distributed": False},
        "dtype": "f32",
        "backend": "pytorch-cpu-fp32",
        "expected_trace": _tensor_json(trace),
        "expected_robustness": _tensor_json(rob),
        "meta": {
            "depth": 1,
            "aggregation_modes": ["hard"],
            "regime": "hard_f32",
            "oracle": oracle_id,
        },
    }


def fx_predicate_equal(oracle_id: str, x: np.ndarray, c: float) -> Dict[str, Any]:
    xt = torch.tensor(x, requires_grad=False)
    ct = torch.tensor(c, dtype=torch.float32, requires_grad=False)
    formula = stlcg.Equal(lhs="x", val=ct)
    trace, rob = _evaluate(formula, xt)

    return {
        "formula_id": f"equal_{oracle_id}_c{c}",
        "formula_ast": {"op": "Equal", "lhs": "x", "val": float(c)},
        "required_operators": ["equal"],
        "inputs": {"x": _np_json(x)},
        "opts": {"pscale": 1.0, "scale": -1.0, "agm": False, "distributed": False},
        "dtype": "f32",
        "backend": "pytorch-cpu-fp32",
        "expected_trace": _tensor_json(trace),
        "expected_robustness": _tensor_json(rob),
        "meta": {
            "depth": 1,
            "aggregation_modes": ["hard"],
            "regime": "hard_f32",
            "oracle": oracle_id,
        },
    }


def _ast_less_than(c: float) -> Dict[str, Any]:
    return {"op": "LessThan", "lhs": "x", "val": float(c)}


def fx_always(oracle_id: str, x: np.ndarray, c: float, interval) -> Dict[str, Any]:
    """Always(LessThan(x, c)) with the given interval."""
    xt = torch.tensor(x, requires_grad=False)
    ct = torch.tensor(c, dtype=torch.float32, requires_grad=False)
    sub = stlcg.LessThan(lhs="x", val=ct)
    formula = stlcg.Always(subformula=sub, interval=interval)
    trace, rob = _evaluate(formula, xt)

    interval_ast, interval_id = _interval_ast(interval)

    return {
        "formula_id": f"always_{interval_id}_lt_{oracle_id}_c{c}",
        "formula_ast": {
            "op": "Always",
            "interval": interval_ast,
            "subformula": _ast_less_than(c),
        },
        "required_operators": ["always", "less_than"],
        "inputs": {"x": _np_json(x)},
        "opts": {"pscale": 1.0, "scale": -1.0, "agm": False, "distributed": False},
        "dtype": "f32",
        "backend": "pytorch-cpu-fp32",
        "expected_trace": _tensor_json(trace),
        "expected_robustness": _tensor_json(rob),
        "meta": {
            "depth": 2,
            "aggregation_modes": ["hard"],
            "regime": "hard_f32",
            "oracle": oracle_id,
            "interval": str(interval),
        },
    }


def fx_eventually(oracle_id: str, x: np.ndarray, c: float, interval) -> Dict[str, Any]:
    """Eventually(LessThan(x, c)) with the given interval."""
    xt = torch.tensor(x, requires_grad=False)
    ct = torch.tensor(c, dtype=torch.float32, requires_grad=False)
    sub = stlcg.LessThan(lhs="x", val=ct)
    formula = stlcg.Eventually(subformula=sub, interval=interval)
    trace, rob = _evaluate(formula, xt)

    interval_ast, interval_id = _interval_ast(interval)

    return {
        "formula_id": f"eventually_{interval_id}_lt_{oracle_id}_c{c}",
        "formula_ast": {
            "op": "Eventually",
            "interval": interval_ast,
            "subformula": _ast_less_than(c),
        },
        "required_operators": ["eventually", "less_than"],
        "inputs": {"x": _np_json(x)},
        "opts": {"pscale": 1.0, "scale": -1.0, "agm": False, "distributed": False},
        "dtype": "f32",
        "backend": "pytorch-cpu-fp32",
        "expected_trace": _tensor_json(trace),
        "expected_robustness": _tensor_json(rob),
        "meta": {
            "depth": 2,
            "aggregation_modes": ["hard"],
            "regime": "hard_f32",
            "oracle": oracle_id,
            "interval": str(interval),
        },
    }


def _interval_ast(interval):
    """Return (ast-json, short-id) for an interval spec."""
    if interval is None:
        return None, "nil"

    lo, hi = interval
    if hi == float("inf") or (isinstance(hi, float) and hi != hi):  # inf or nan guard
        return {"lo": int(lo), "hi": "infinity"}, f"{int(lo)}inf"
    return {"lo": int(lo), "hi": int(hi)}, f"{int(lo)}_{int(hi)}"


def build_all_fixtures() -> List[Dict[str, Any]]:
    """Return a list of fixture dicts covering the initial operator set."""
    fixtures: List[Dict[str, Any]] = []

    # Predicates across the full oracle set at threshold c=0.0.
    for oracle_id, trace in ORACLES.items():
        fixtures.append(fx_predicate_less_than(oracle_id, trace, 0.0))
        fixtures.append(fx_predicate_greater_than(oracle_id, trace, 0.0))
        fixtures.append(fx_predicate_equal(oracle_id, trace, 0.0))

    # Extra predicate thresholds to stress sign/magnitude.
    fixtures.append(fx_predicate_less_than("O3", ORACLES["O3"], 1.0))
    fixtures.append(fx_predicate_less_than("O4", ORACLES["O4"], 0.5))
    fixtures.append(fx_predicate_greater_than("O3", ORACLES["O3"], 0.5))

    # Always / Eventually across each interval kind on O2/O4/O5.
    interval_cases = [
        None,
        [1, 2],
        [2, 3],
        [1, float("inf")],
        [2, float("inf")],
    ]
    for oracle_id in ("O2", "O4", "O5"):
        for interval in interval_cases:
            fixtures.append(fx_always(oracle_id, ORACLES[oracle_id], 0.0, interval))
            fixtures.append(fx_eventually(oracle_id, ORACLES[oracle_id], 0.0, interval))

    # Until / Then on same-length oracle pairs. Integral1d is validated
    # via analytic tests in Elixir — upstream's torch-conv-based impl
    # crashes on modern torch and is not a reliable oracle.
    fixtures.extend(build_until_then_fixtures())

    return fixtures


def build_until_then_fixtures() -> List[Dict[str, Any]]:
    """Until(phi, psi) and Then(phi, psi) on a pair of oracle traces."""
    # (phi_signal, psi_signal, pair_id) — both must be same length.
    # O5 is length 5; build a second length-5 signal to pair with it.
    O5b = np.array([[[0.4], [-0.2], [0.1], [0.3], [-0.1]]], dtype=np.float32)
    pairs = [
        (ORACLES["O5"], O5b, "O5O5b"),
        (O5b, ORACLES["O5"], "O5bO5"),
    ]

    fixtures = []
    for x, y, pair_id in pairs:
        xt = torch.tensor(x, requires_grad=False)
        yt = torch.tensor(y, requires_grad=False)
        c0 = torch.tensor(0.0)

        phi = stlcg.LessThan(lhs="x", val=c0)
        psi = stlcg.GreaterThan(lhs="y", val=c0)

        for interval in [None, [1, 2], [1, float("inf")]]:
            interval_ast, interval_id = _interval_ast(interval)

            for op_name, OpClass, tag in (
                ("Until", stlcg.Until, "until"),
                ("Then", stlcg.Then, "then"),
            ):
                f = OpClass(subformula1=phi, subformula2=psi, interval=interval)
                trace, rob = _evaluate(f, (xt, yt))

                fixtures.append({
                    "formula_id": f"{op_name.lower()}_{interval_id}_{pair_id}",
                    "formula_ast": {
                        "op": op_name,
                        "interval": interval_ast,
                        "lhs": _ast_less_than(0.0),
                        "rhs": {"op": "GreaterThan", "lhs": "y", "val": 0.0},
                    },
                    "required_operators": [tag, "less_than", "greater_than"],
                    "inputs": {"x": _np_json(x), "y": _np_json(y)},
                    "opts": {"pscale": 1.0, "scale": -1.0, "agm": False, "distributed": False},
                    "dtype": "f32",
                    "backend": "pytorch-cpu-fp32",
                    "expected_trace": _tensor_json(trace),
                    "expected_robustness": _tensor_json(rob),
                    "meta": {
                        "depth": 3,
                        "aggregation_modes": ["hard"],
                        "regime": "hard_f32",
                        "pair": pair_id,
                        "interval": str(interval),
                    },
                })

    return fixtures


def build_integral_fixtures() -> List[Dict[str, Any]]:
    fixtures = []

    for oracle_id in ("O4", "O5"):
        x = ORACLES[oracle_id]
        xt = torch.tensor(x, requires_grad=False)
        c0 = torch.tensor(0.0)
        phi = stlcg.LessThan(lhs="x", val=c0)

        for interval in [None, [1, 2], [1, float("inf")]]:
            interval_ast, interval_id = _interval_ast(interval)
            f = stlcg.Integral1d(subformula=phi, interval=interval)
            trace, rob = _evaluate(f, xt)

            fixtures.append({
                "formula_id": f"integral_{interval_id}_{oracle_id}",
                "formula_ast": {
                    "op": "Integral1d",
                    "interval": interval_ast,
                    "subformula": _ast_less_than(0.0),
                },
                "required_operators": ["integral1d", "less_than"],
                "inputs": {"x": _np_json(x)},
                "opts": {"pscale": 1.0, "scale": -1.0, "agm": False, "distributed": False},
                "dtype": "f32",
                "backend": "pytorch-cpu-fp32",
                "expected_trace": _tensor_json(trace),
                "expected_robustness": _tensor_json(rob),
                "meta": {
                    "depth": 2,
                    "aggregation_modes": ["hard"],
                    "regime": "hard_f32",
                    "oracle": oracle_id,
                    "interval": str(interval),
                },
            })

    return fixtures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory for *.json fixtures")
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    fixtures = build_all_fixtures()

    # Write one file per fixture for easy grep + diff.
    for fx in fixtures:
        fpath = outdir / f"{fx['formula_id']}.json"
        with open(fpath, "w") as fh:
            json.dump(fx, fh, indent=2, sort_keys=True)

    # Also write an index so the Elixir harness doesn't need to glob.
    index = {
        "count": len(fixtures),
        "fixtures": sorted(fx["formula_id"] for fx in fixtures),
        "generated_from": "StanfordASL/stlcg@abd16c92108f1b57a72d66c58492c949b6c5a8ea",
    }
    with open(outdir / "index.json", "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)

    print(f"  wrote {len(fixtures)} fixtures to {outdir}")


if __name__ == "__main__":
    main()
