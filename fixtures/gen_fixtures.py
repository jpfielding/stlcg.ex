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


def build_all_fixtures() -> List[Dict[str, Any]]:
    """Return a list of fixture dicts covering the initial operator set."""
    fixtures: List[Dict[str, Any]] = []

    # Predicates across the full oracle set at threshold c=0.0.
    for oracle_id, trace in ORACLES.items():
        fixtures.append(fx_predicate_less_than(oracle_id, trace, 0.0))
        fixtures.append(fx_predicate_greater_than(oracle_id, trace, 0.0))
        fixtures.append(fx_predicate_equal(oracle_id, trace, 0.0))

    # Extra predicate thresholds to stress sign/magnitude.
    fixtures.append(fx_predicate_less_than("O3", ORACLES["O3"], 1.0))   # tight: trace == c
    fixtures.append(fx_predicate_less_than("O4", ORACLES["O4"], 0.5))   # equal at endpoint
    fixtures.append(fx_predicate_greater_than("O3", ORACLES["O3"], 0.5))

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
