"""
Ground-truth reference solver: directly calls load_case + cvxpy (no agent).
Returns the provably optimal dispatch for each test scenario.
"""

import sys
import os
import math
import json
import copy

sys.path.insert(0, os.path.dirname(__file__))

import pandapower.networks as pn
import pandas as pd
from tool_cvxpy import solve_ed_from_json


def load_case_as_json(case_name: str) -> dict:
    """Load an IEEE case and return structured JSON for the solver."""
    case_map = {
        "IEEE14": pn.case14,
        "IEEE30": pn.case30,
        "IEEE57": pn.case57,
        "IEEE118": pn.case118,
        "IEEE200": pn.case_illinois200,
        "IEEE300": pn.case300,
    }
    net = case_map[case_name]()

    gen_df = net.gen[["bus", "max_p_mw", "min_p_mw"]].copy()
    # Use same join logic as load_case.py (index-based join, mirrors agent behavior)
    cost_df = net.poly_cost[net.poly_cost.et == "gen"].copy()
    merged = gen_df.join(cost_df[["cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2"]])
    total_load = float(net.load.p_mw.sum())

    generators = []
    for i, row in merged.iterrows():
        a = float(row.cp2_eur_per_mw2)
        b = float(row.cp1_eur_per_mw)
        c = float(row.cp0_eur)
        status = 0 if (math.isnan(a) or math.isnan(b)) else 1
        generators.append({
            "id": i,
            "status": status,
            "p_min": float(row.min_p_mw),
            "p_max": float(row.max_p_mw),
            "cost": {"a": a, "b": b, "c": c if not math.isnan(c) else 0.0},
        })

    return {"target_load_mw": total_load, "generators": generators}


def solve_reference(case_name: str, offline_unit: int = None, load_override: float = None) -> dict:
    """
    Compute the reference optimal dispatch.
    - offline_unit: generator ID to force offline
    - load_override: override target load (MW)
    """
    ed_json = load_case_as_json(case_name)

    if offline_unit is not None:
        for g in ed_json["generators"]:
            if g["id"] == offline_unit:
                g["status"] = 0

    if load_override is not None:
        ed_json["target_load_mw"] = load_override

    result = solve_ed_from_json(ed_json)
    result["case"] = case_name
    result["offline_unit"] = offline_unit
    result["target_load"] = ed_json["target_load_mw"]
    result["n_active_generators"] = sum(
        1 for g in ed_json["generators"]
        if g["status"] == 1 and not math.isnan(g["cost"]["a"])
    )
    return result


if __name__ == "__main__":
    cases = ["IEEE14", "IEEE30", "IEEE57", "IEEE118", "IEEE200", "IEEE300"]
    for case in cases:
        r = solve_reference(case)
        print(f"{case}: status={r['status']}, cost=${r.get('total_cost', 'N/A'):.2f}/hr, "
              f"dispatch={r.get('dispatch_results', {})}")
