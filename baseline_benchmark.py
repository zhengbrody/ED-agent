"""
Baseline (LLM-only) Benchmark Evaluation
Runs the BaselineAgent (no tools, single-shot LLM prompting) on the same 20 test cases
used for EDAgent, then compares against CVXPY ground truth.

This directly mirrors benchmark.py but uses BaselineAgent instead of the agentic EDAgent.

Usage:
  /opt/anaconda3/bin/python baseline_benchmark.py
"""

import re
import sys
import json
import time
import math
import numpy as np

sys.path.insert(0, ".")

from benchmark_reference import solve_reference, load_case_as_json
from benchmark import TEST_CASES, parse_agent_response
from EDAgentLLM import EDAgentLLM
from BaselineAgent import BaselineAgent


# ─────────────────────────────────────────────
# Prompt Builder for Baseline (LLM-only, no tools)
# Follows the reference paper methodology:
# Mohammadi et al., "LLM for Economic Dispatch" (arXiv:2505.21931)
# ─────────────────────────────────────────────

def build_baseline_prompt(case_name: str, offline_unit: int = None, load_override: float = None):
    """
    Build a prompt for the LLM-only baseline.
    Provides the system data as text and asks the LLM to directly compute
    the optimal dispatch — NO solver, NO tools, just chain-of-thought prompting.
    """
    ed_json = load_case_as_json(case_name)

    # Apply scenario modifications
    if offline_unit is not None:
        for g in ed_json["generators"]:
            if g["id"] == offline_unit:
                g["status"] = 0

    if load_override is not None:
        ed_json["target_load_mw"] = load_override

    target_load = ed_json["target_load_mw"]
    active_gens = [g for g in ed_json["generators"] if g["status"] == 1 and not math.isnan(g["cost"]["a"])]

    # Build generator table
    gen_table = ""
    for g in active_gens:
        gen_table += (
            f"  Unit {g['id']}: Pmin={g['p_min']:.1f} MW, Pmax={g['p_max']:.1f} MW, "
            f"Cost coefficients: a={g['cost']['a']:.6f}, b={g['cost']['b']:.1f}, c={g['cost'].get('c', 0.0):.1f}\n"
        )

    offline_note = ""
    if offline_unit is not None:
        offline_note = f"\nNote: Generator {offline_unit} is OFFLINE and must NOT be dispatched.\n"

    prompt = f"""You are a power systems expert. Solve the following Economic Dispatch (ED) problem.

Problem: Minimize the total generation cost while meeting the system load demand.

The cost function for each generator i is: Cost_i = a_i * Pg_i^2 + b_i * Pg_i + c_i

Constraints:
1. Power Balance: Sum of all Pg_i must equal the total system load ({target_load:.1f} MW)
2. Generator Limits: Pmin_i <= Pg_i <= Pmax_i for each active generator
{offline_note}
System Data ({case_name}):
Total System Load (Pd) = {target_load:.1f} MW
Number of Active Generators: {len(active_gens)}

Generator Parameters:
{gen_table}

Instructions:
- Think step-by-step about the optimal dispatch strategy
- The optimal solution uses equal incremental cost (lambda-iteration) principle
- Compute the dispatch for each generator
- Calculate the total generation cost
- Report: Total Cost: $<value>
- Report: PG = [list of dispatch values]

Solve this Economic Dispatch problem now.
"""
    return prompt


# ─────────────────────────────────────────────
# Baseline Response Parser
# ─────────────────────────────────────────────

def parse_baseline_response(raw: str, case_name: str, offline_unit: int = None,
                            load_override: float = None) -> dict:
    """
    Parse the baseline LLM response.
    Extract cost and dispatch vector, then verify feasibility.
    """
    result = {"cost": None, "feasible": None, "dispatch": [], "power_balance": False}

    if not raw:
        return result

    # Try standard cost parser first
    std_result = parse_agent_response(raw)
    result["cost"] = std_result["cost"]

    # Also try to extract PG vector
    pg_patterns = [
        r"PG\s*=\s*\[([\d\.,\s]+)\]",
        r"\[([\d]+\.?\d*(?:\s*,\s*\d+\.?\d*)+)\]",
    ]
    for pat in pg_patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            try:
                vals = [float(x.strip()) for x in m.group(1).split(",")]
                result["dispatch"] = vals
                break
            except ValueError:
                pass

    # Verify feasibility if we have dispatch values
    if result["dispatch"]:
        ed_json = load_case_as_json(case_name)
        if offline_unit is not None:
            for g in ed_json["generators"]:
                if g["id"] == offline_unit:
                    g["status"] = 0
        if load_override is not None:
            ed_json["target_load_mw"] = load_override

        target = ed_json["target_load_mw"]
        total_gen = sum(result["dispatch"])
        balance_error = abs(total_gen - target) / target * 100

        result["power_balance"] = balance_error < 1.0  # within 1% tolerance
        result["balance_error_pct"] = balance_error

        # Check generator limits
        active_gens = [g for g in ed_json["generators"]
                       if g["status"] == 1 and not math.isnan(g["cost"]["a"])]
        limits_ok = True
        if len(result["dispatch"]) == len(active_gens):
            for i, (pg_val, g) in enumerate(zip(result["dispatch"], active_gens)):
                if pg_val < g["p_min"] - 0.1 or pg_val > g["p_max"] + 0.1:
                    limits_ok = False
                    break
        else:
            limits_ok = False  # dimension mismatch

        result["limits_ok"] = limits_ok
        result["feasible"] = result["power_balance"] and limits_ok

        # Compute actual cost from dispatch if cost not parsed from text
        if result["feasible"] and len(result["dispatch"]) == len(active_gens):
            actual_cost = 0.0
            for pg_val, g in zip(result["dispatch"], active_gens):
                a = g["cost"]["a"]
                b = g["cost"]["b"]
                c = g["cost"].get("c", 0.0)
                actual_cost += a * pg_val**2 + b * pg_val + c
            result["computed_cost"] = actual_cost
            if result["cost"] is None:
                result["cost"] = actual_cost

    elif result["cost"] is not None:
        result["feasible"] = True  # assume feasible if cost was reported

    # Detect infeasibility keywords
    infeasible_patterns = [
        r"infeasible", r"cannot be solved", r"no feasible", r"insufficient capacity",
    ]
    for pat in infeasible_patterns:
        if re.search(pat, raw.lower()):
            result["feasible"] = False
            return result

    return result


# ─────────────────────────────────────────────
# Main Benchmark Runner
# ─────────────────────────────────────────────

def run_baseline_benchmark():
    print("=" * 70)
    print("  Baseline (LLM-only) Benchmark Evaluation")
    print(f"  {len(TEST_CASES)} test cases across IEEE 14/30/57/118-bus")
    print("  Method: Single-shot prompting, NO tools, NO solver")
    print("=" * 70)

    llm_client = EDAgentLLM()
    baseline = BaselineAgent(llm_client, temperature=0)

    all_metrics = []
    for i, tc in enumerate(TEST_CASES):
        print(f"\n[{i+1:02d}/{len(TEST_CASES)}] {tc['id']} | {tc['scenario']}")

        # Compute reference solution
        ref = solve_reference(
            tc["case"],
            offline_unit=tc["offline_unit"],
            load_override=tc["load_override"],
        )
        ref_cost = ref.get("total_cost")
        ref_feasible = ref.get("status") == "Success"
        print(f"  Reference: {ref['status']}, cost=${ref_cost or 'N/A'}")

        # Build baseline prompt and call LLM
        prompt = build_baseline_prompt(
            tc["case"],
            offline_unit=tc["offline_unit"],
            load_override=tc["load_override"],
        )
        print("  Calling baseline LLM... ", end="", flush=True)
        t0 = time.time()
        try:
            raw_response = baseline.run(prompt)
            elapsed = time.time() - t0
            success = True
        except Exception as e:
            raw_response = ""
            elapsed = time.time() - t0
            success = False
            print(f"ERROR: {e}")

        print(f"done ({elapsed:.1f}s)")

        # Parse response
        parsed = parse_baseline_response(
            raw_response or "",
            tc["case"],
            offline_unit=tc["offline_unit"],
            load_override=tc["load_override"],
        )

        # Build metrics
        m = {
            "test_id": tc["id"],
            "case": tc["case"],
            "scenario": tc["scenario"],
            "expected_feasible": tc["expected_feasible"],
            "baseline_success": success,
            "baseline_feasible": parsed["feasible"],
            "baseline_cost": parsed["cost"],
            "baseline_computed_cost": parsed.get("computed_cost"),
            "ref_feasible": ref_feasible,
            "ref_cost": ref_cost,
            "cost_error_pct": None,
            "power_balance_ok": parsed.get("power_balance", False),
            "balance_error_pct": parsed.get("balance_error_pct"),
            "limits_ok": parsed.get("limits_ok", False),
            "dispatch_vector": parsed.get("dispatch", []),
            "elapsed": elapsed,
            "failure_mode": None,
            "raw_response": (raw_response or "")[:3000],
        }

        # Compute cost error
        if parsed["cost"] is not None and ref_cost is not None and ref_feasible:
            cost_to_compare = parsed.get("computed_cost", parsed["cost"])
            m["cost_error_pct"] = abs(cost_to_compare - ref_cost) / ref_cost * 100

        # Determine failure mode
        if not success:
            m["failure_mode"] = "API error"
        elif parsed["feasible"] is False:
            if not tc["expected_feasible"]:
                m["failure_mode"] = None  # correctly identified infeasible
            else:
                if not parsed.get("power_balance"):
                    m["failure_mode"] = "Power balance violation"
                elif not parsed.get("limits_ok"):
                    m["failure_mode"] = "Generator limit violation"
                else:
                    m["failure_mode"] = "Infeasible reported"
        elif parsed["cost"] is None and not parsed["dispatch"]:
            m["failure_mode"] = "Parse error (no output extracted)"
        elif parsed["feasible"] is None:
            m["failure_mode"] = "Unknown (no feasibility determined)"

        all_metrics.append(m)

        # Print status
        if m["failure_mode"]:
            print(f"  ✗ {m['failure_mode']}")
            if m.get("balance_error_pct") is not None:
                print(f"    Balance error: {m['balance_error_pct']:.2f}%")
        elif m["cost_error_pct"] is not None:
            err = m["cost_error_pct"]
            icon = "✓" if err < 1.0 else ("~" if err < 10.0 else "✗")
            print(f"  {icon} Cost: ${m['baseline_cost']:.2f} vs ref ${ref_cost:.2f} (err={err:.2f}%)")
        else:
            print(f"  ? Unknown result")

        time.sleep(2)

    # ─── Summary ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BASELINE BENCHMARK SUMMARY")
    print("=" * 70)

    total = len(all_metrics)
    success_count = sum(1 for m in all_metrics if m["baseline_success"])
    feasible_count = sum(1 for m in all_metrics if m["baseline_feasible"] is True)
    balance_ok = sum(1 for m in all_metrics if m["power_balance_ok"])
    limits_ok = sum(1 for m in all_metrics if m["limits_ok"])

    expected_feasible_count = sum(1 for tc in TEST_CASES if tc["expected_feasible"])
    feasible_correct = sum(
        1 for m in all_metrics
        if m["expected_feasible"] and m["baseline_feasible"] is True
    )
    infeasible_correct = sum(
        1 for m in all_metrics
        if not m["expected_feasible"] and m["baseline_feasible"] is False
    )

    cost_errors = [m["cost_error_pct"] for m in all_metrics if m["cost_error_pct"] is not None]
    avg_cost_error = sum(cost_errors) / len(cost_errors) if cost_errors else None
    max_cost_error = max(cost_errors) if cost_errors else None

    failure_modes = {}
    for m in all_metrics:
        if m["failure_mode"]:
            fm = m["failure_mode"]
            failure_modes[fm] = failure_modes.get(fm, 0) + 1

    print(f"\n  Total queries:              {total}")
    print(f"  API success:                {success_count}/{total}")
    print(f"  Feasible solutions:         {feasible_count}/{total}")
    print(f"  Power balance satisfied:    {balance_ok}/{total}")
    print(f"  Generator limits respected: {limits_ok}/{total}")
    print(f"  Feasibility accuracy:       {feasible_correct}/{expected_feasible_count}")
    print(f"  Infeasibility detected:     {infeasible_correct}/{total - expected_feasible_count}")

    if avg_cost_error is not None:
        print(f"  Avg cost error:             {avg_cost_error:.2f}%")
        print(f"  Max cost error:             {max_cost_error:.2f}%")

    print(f"\n  Failure mode breakdown:")
    if failure_modes:
        for fm, cnt in sorted(failure_modes.items(), key=lambda x: -x[1]):
            print(f"    - {fm}: {cnt}")
    else:
        print("    None")

    # Per-case breakdown
    print(f"\n  Per-case breakdown:")
    for case in ["IEEE14", "IEEE30", "IEEE57", "IEEE118"]:
        case_m = [m for m in all_metrics if m["case"] == case]
        case_feasible = sum(1 for m in case_m if m["baseline_feasible"] is True)
        case_errors = [m["cost_error_pct"] for m in case_m if m.get("cost_error_pct") is not None]
        avg_err = f"{sum(case_errors)/len(case_errors):.2f}%" if case_errors else "N/A"
        print(f"    {case}: {case_feasible}/{len(case_m)} feasible, avg cost error={avg_err}")

    # Save results
    output_path = "baseline_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "success_count": success_count,
                "feasible_count": feasible_count,
                "balance_ok": balance_ok,
                "limits_ok": limits_ok,
                "feasible_correct": feasible_correct,
                "infeasible_correct": infeasible_correct,
                "avg_cost_error_pct": round(avg_cost_error, 4) if avg_cost_error else None,
                "max_cost_error_pct": round(max_cost_error, 4) if max_cost_error else None,
                "failure_modes": failure_modes,
            },
            "results": all_metrics,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to {output_path}")
    print("=" * 70)

    return all_metrics


if __name__ == "__main__":
    run_baseline_benchmark()

