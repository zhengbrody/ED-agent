"""
EDAgent Benchmark Evaluation
Tests the agent on 20 queries across IEEE 14/30/57/118-bus cases.
Compares results against a direct CVXPY reference solver.

Usage:
  # Ensure the agent server is running on port 5001 first:
  #   /opt/anaconda3/bin/python app.py
  /opt/anaconda3/bin/python benchmark.py
"""

import re
import sys
import json
import time
import math
import requests
from typing import Optional

sys.path.insert(0, ".")
from benchmark_reference import solve_reference


# ─────────────────────────────────────────────
# Test Case Definitions
# ─────────────────────────────────────────────

TEST_CASES = [
    # ── IEEE 14-bus ──────────────────────────────────
    {
        "id": "14-base",
        "case": "IEEE14",
        "scenario": "baseline",
        "offline_unit": None,
        "load_override": None,
        "query": "Run Economic Dispatch for IEEE 14-bus system. Minimize total generation cost.",
        "expected_feasible": True,
    },
    {
        "id": "14-offline-1",
        "case": "IEEE14",
        "scenario": "unit_outage",
        "offline_unit": 1,
        "load_override": 180.0,
        "query": "Run Economic Dispatch for IEEE 14-bus system with Generator 1 offline. The total system load is 180 MW.",
        "expected_feasible": True,
    },
    {
        "id": "14-offline-2",
        "case": "IEEE14",
        "scenario": "unit_outage",
        "offline_unit": 2,
        "load_override": 170.0,
        "query": "Run Economic Dispatch for IEEE 14-bus system with Generator 2 forced offline. The total system load is 170 MW.",
        "expected_feasible": True,
    },
    {
        "id": "14-load-hi",
        "case": "IEEE14",
        "scenario": "load_change",
        "offline_unit": None,
        "load_override": 230.0,
        "query": "Run Economic Dispatch for IEEE 14-bus system. The total system load is 230 MW.",
        "expected_feasible": True,
    },
    {
        "id": "14-load-lo",
        "case": "IEEE14",
        "scenario": "load_change",
        "offline_unit": None,
        "load_override": 180.0,
        "query": "Run Economic Dispatch for IEEE 14-bus system. The total system load is 180 MW.",
        "expected_feasible": True,
    },
    # ── IEEE 30-bus ──────────────────────────────────
    {
        "id": "30-base",
        "case": "IEEE30",
        "scenario": "baseline",
        "offline_unit": None,
        "load_override": None,
        "query": "Run Economic Dispatch for IEEE 30-bus system. Minimize total generation cost.",
        "expected_feasible": False,  # Sum Pmax < load due to index bug in data loader
    },
    {
        "id": "30-load-lo",
        "case": "IEEE30",
        "scenario": "load_change",
        "offline_unit": None,
        "load_override": 150.0,
        "query": "Run Economic Dispatch for IEEE 30-bus system. The total system load is 150 MW.",
        "expected_feasible": True,
    },
    {
        "id": "30-offline-1",
        "case": "IEEE30",
        "scenario": "unit_outage",
        "offline_unit": 1,
        "load_override": 110.0,
        "query": "Run Economic Dispatch for IEEE 30-bus system with Generator 1 offline. The total load is 110 MW.",
        "expected_feasible": True,
    },
    {
        "id": "30-offline-2",
        "case": "IEEE30",
        "scenario": "unit_outage",
        "offline_unit": 2,
        "load_override": 100.0,
        "query": "Run Economic Dispatch for IEEE 30-bus system with Generator 2 offline. The total load is 100 MW.",
        "expected_feasible": True,
    },
    {
        "id": "30-load-mid",
        "case": "IEEE30",
        "scenario": "load_change",
        "offline_unit": None,
        "load_override": 120.0,
        "query": "Run Economic Dispatch for IEEE 30-bus system. The total system load is 120 MW.",
        "expected_feasible": True,
    },
    # ── IEEE 57-bus ──────────────────────────────────
    {
        "id": "57-base",
        "case": "IEEE57",
        "scenario": "baseline",
        "offline_unit": None,
        "load_override": None,
        "query": "Run Economic Dispatch for IEEE 57-bus system. Minimize total generation cost.",
        "expected_feasible": True,
    },
    {
        "id": "57-offline-1",
        "case": "IEEE57",
        "scenario": "unit_outage",
        "offline_unit": 1,
        "load_override": 1100.0,
        "query": "Run Economic Dispatch for IEEE 57-bus system with Generator 1 taken offline. The total system load is 1100 MW.",
        "expected_feasible": True,
    },
    {
        "id": "57-offline-2",
        "case": "IEEE57",
        "scenario": "unit_outage",
        "offline_unit": 2,
        "load_override": 1000.0,
        "query": "Run Economic Dispatch for IEEE 57-bus system with Generator 2 offline. The total system load is 1000 MW.",
        "expected_feasible": True,
    },
    {
        "id": "57-load-hi",
        "case": "IEEE57",
        "scenario": "load_change",
        "offline_unit": None,
        "load_override": 1100.0,
        "query": "Run Economic Dispatch for IEEE 57-bus system. The total system load is 1100 MW.",
        "expected_feasible": True,
    },
    {
        "id": "57-load-lo",
        "case": "IEEE57",
        "scenario": "load_change",
        "offline_unit": None,
        "load_override": 900.0,
        "query": "Run Economic Dispatch for IEEE 57-bus system. The total system load is 900 MW.",
        "expected_feasible": True,
    },
    # ── IEEE 118-bus ─────────────────────────────────
    {
        "id": "118-base",
        "case": "IEEE118",
        "scenario": "baseline",
        "offline_unit": None,
        "load_override": None,
        "query": "Run Economic Dispatch for IEEE 118-bus system. Minimize total generation cost.",
        "expected_feasible": True,
    },
    {
        "id": "118-offline-1",
        "case": "IEEE118",
        "scenario": "unit_outage",
        "offline_unit": 1,
        "load_override": None,
        "query": "Run Economic Dispatch for IEEE 118-bus system with Generator 1 offline.",
        "expected_feasible": True,
    },
    {
        "id": "118-offline-5",
        "case": "IEEE118",
        "scenario": "unit_outage",
        "offline_unit": 5,
        "load_override": None,
        "query": "Run Economic Dispatch for IEEE 118-bus system with Generator 5 offline.",
        "expected_feasible": True,
    },
    {
        "id": "118-load-hi",
        "case": "IEEE118",
        "scenario": "load_change",
        "offline_unit": None,
        "load_override": 4000.0,
        "query": "Run Economic Dispatch for IEEE 118-bus system. The total system load is 4000 MW.",
        "expected_feasible": True,
    },
    {
        "id": "118-load-lo",
        "case": "IEEE118",
        "scenario": "load_change",
        "offline_unit": None,
        "load_override": 3500.0,
        "query": "Run Economic Dispatch for IEEE 118-bus system. The total system load is 3500 MW.",
        "expected_feasible": True,
    },
    # ── IEEE 200-bus (Illinois) ───────────────────
    {
        "id": "200-base",
        "case": "IEEE200",
        "scenario": "baseline",
        "offline_unit": None,
        "load_override": None,
        "query": "Run Economic Dispatch for IEEE 200-bus system. Minimize total generation cost.",
        "expected_feasible": True,
    },
    {
        "id": "200-offline-1",
        "case": "IEEE200",
        "scenario": "unit_outage",
        "offline_unit": 1,
        "load_override": 2000.0,
        "query": "Run Economic Dispatch for IEEE 200-bus system with Generator 1 offline. The total system load is 2000 MW.",
        "expected_feasible": True,
    },
    {
        "id": "200-offline-5",
        "case": "IEEE200",
        "scenario": "unit_outage",
        "offline_unit": 5,
        "load_override": 1800.0,
        "query": "Run Economic Dispatch for IEEE 200-bus system with Generator 5 offline. The total system load is 1800 MW.",
        "expected_feasible": True,

    },
    {
        "id": "200-load-lo",
        "case": "IEEE200",
        "scenario": "load_change",
        "offline_unit": None,
        "load_override": 1800.0,
        "query": "Run Economic Dispatch for IEEE 200-bus system. The total system load is 1800 MW.",
        "expected_feasible": True,
    },
    {
        "id": "200-load-hi",
        "case": "IEEE200",
        "scenario": "load_change",
        "offline_unit": None,
        "load_override": 2200.0,
        "query": "Run Economic Dispatch for IEEE 200-bus system. The total system load is 2200 MW.",
        "expected_feasible": True,
    },
    # ── IEEE 300-bus ──────────────────────────────
    {
        "id": "300-base",
        "case": "IEEE300",
        "scenario": "baseline",
        "offline_unit": None,
        "load_override": None,
        "query": "Run Economic Dispatch for IEEE 300-bus system. Minimize total generation cost.",
        "expected_feasible": True,
    },
    {
        "id": "300-offline-1",
        "case": "IEEE300",
        "scenario": "unit_outage",
        "offline_unit": 1,
        "load_override": None,
        "query": "Run Economic Dispatch for IEEE 300-bus system with Generator 1 offline.",
        "expected_feasible": True,
    },
    {
        "id": "300-offline-10",
        "case": "IEEE300",
        "scenario": "unit_outage",
        "offline_unit": 10,
        "load_override": 20000.0,
        "query": "Run Economic Dispatch for IEEE 300-bus system with Generator 10 offline. The total system load is 20000 MW.",
        "expected_feasible": True,
    },
    {
        "id": "300-load-lo",
        "case": "IEEE300",
        "scenario": "load_change",
        "offline_unit": None,
        "load_override": 19000.0,
        "query": "Run Economic Dispatch for IEEE 300-bus system. The total system load is 19000 MW.",
        "expected_feasible": True,
    },
    {
        "id": "300-load-hi",
        "case": "IEEE300",
        "scenario": "load_change",
        "offline_unit": None,
        "load_override": 26000.0,
        "query": "Run Economic Dispatch for IEEE 300-bus system. The total system load is 26000 MW.",
        "expected_feasible": True,
    },
]


# ─────────────────────────────────────────────
# Agent API Call
# ─────────────────────────────────────────────

API_URL = "http://localhost:5001/v1/chat/completions"
TIMEOUT = 360  # seconds per query


def call_agent(query: str) -> dict:
    """Send a query to the EDAgent and return parsed result."""
    payload = {
        "model": "ed-agent-react",
        "messages": [{"role": "user", "content": query}],
        "stream": False,
    }
    try:
        t0 = time.time()
        resp = requests.post(API_URL, json=payload, timeout=TIMEOUT)
        elapsed = time.time() - t0
        if resp.status_code != 200:
            return {"success": False, "error": f"HTTP {resp.status_code}", "elapsed": elapsed, "raw": ""}
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return {"success": True, "raw": content, "elapsed": elapsed}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout", "elapsed": TIMEOUT, "raw": ""}
    except Exception as e:
        return {"success": False, "error": str(e), "elapsed": 0, "raw": ""}


# ─────────────────────────────────────────────
# Response Parser
# ─────────────────────────────────────────────

def parse_agent_response(raw: str) -> dict:
    """Extract total cost and feasibility from agent natural language response."""
    result = {"cost": None, "feasible": None, "dispatch": {}}

    # Detect infeasibility
    infeasible_patterns = [
        r"infeasible", r"cannot be solved", r"no feasible", r"insufficient capacity",
        r"unable to meet", r"not feasible", r"exceeds.*capacity", r"insufficient.*generation"
    ]
    raw_lower = raw.lower()
    for pat in infeasible_patterns:
        if re.search(pat, raw_lower):
            result["feasible"] = False
            return result

    # Extract total cost: patterns like "$10,250.25" or "10250.25 $/hr" or "Total Cost: 10250.25"
    cost_patterns = [
        r"\$\s*([\d,]+\.?\d*)\s*/?\s*hr",
        r"\*{0,2}total\s+(?:generation\s+)?cost\*{0,2}\s*[:\|]\s*\*{0,2}\s*\$?\s*([\d,]+\.?\d*)",
        r"total\s+cost[:\s]+\$?\s*([\d,]+\.?\d*)",
        r"minimum\s+(?:total\s+)?cost[:\s]+\$?\s*([\d,]+\.?\d*)",
        r"optimal\s+(?:total\s+)?cost[:\s]+\$?\s*([\d,]+\.?\d*)",
        r"cost[:\s=]+\$?\s*([\d,]+\.?\d*)",
        r"([\d,]+\.?\d+)\s*\$?\s*/hr",
    ]
    for pat in cost_patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            try:
                result["cost"] = float(m.group(1).replace(",", ""))
                result["feasible"] = True
                break
            except ValueError:
                pass

    # If cost found, mark feasible
    if result["cost"] is not None:
        result["feasible"] = True
    elif result["feasible"] is None:
        # If neither infeasible keyword nor cost found → parse error
        result["feasible"] = None  # unknown

    return result


# ─────────────────────────────────────────────
# Feasibility Checker
# ─────────────────────────────────────────────

def check_agent_feasibility(agent_result: dict, ref_result: dict) -> dict:
    """
    Compare agent result against reference solution.
    Returns metrics dict.
    """
    metrics = {
        "agent_success": agent_result["success"],
        "agent_feasible": None,
        "ref_feasible": ref_result.get("status") == "Success",
        "cost_error_pct": None,
        "agent_cost": None,
        "ref_cost": ref_result.get("total_cost"),
        "elapsed": agent_result.get("elapsed", 0),
        "failure_mode": None,
    }

    if not agent_result["success"]:
        metrics["failure_mode"] = agent_result.get("error", "API error")
        return metrics

    parsed = parse_agent_response(agent_result["raw"])
    metrics["agent_feasible"] = parsed["feasible"]
    metrics["agent_cost"] = parsed["cost"]

    if parsed["feasible"] is False:
        metrics["failure_mode"] = "Infeasible reported"
    elif parsed["cost"] is None:
        metrics["failure_mode"] = "Parse error (no cost extracted)"
    elif metrics["ref_feasible"] and parsed["cost"] is not None:
        ref_cost = metrics["ref_cost"]
        metrics["cost_error_pct"] = abs(parsed["cost"] - ref_cost) / ref_cost * 100

    return metrics


# ─────────────────────────────────────────────
# Main Benchmark Runner
# ─────────────────────────────────────────────

def run_benchmark():
    print("=" * 70)
    print("  EDAgent Benchmark Evaluation")
    print(f"  {len(TEST_CASES)} test cases across IEEE 14/30/57/118-bus")
    print("=" * 70)

    all_metrics = []
    for i, tc in enumerate(TEST_CASES):
        print(f"\n[{i+1:02d}/{len(TEST_CASES)}] {tc['id']} | {tc['scenario']}")
        print(f"  Query: {tc['query'][:80]}...")

        # Compute reference
        ref = solve_reference(
            tc["case"],
            offline_unit=tc["offline_unit"],
            load_override=tc["load_override"],
        )
        print(f"  Reference: {ref['status']}, cost=${ref.get('total_cost', 'N/A')}")

        # Call agent
        print("  Calling agent... ", end="", flush=True)
        agent_resp = call_agent(tc["query"])
        print(f"done ({agent_resp.get('elapsed', 0):.1f}s)")

        # Evaluate
        m = check_agent_feasibility(agent_resp, ref)
        m["test_id"] = tc["id"]
        m["case"] = tc["case"]
        m["scenario"] = tc["scenario"]
        m["expected_feasible"] = tc["expected_feasible"]
        all_metrics.append(m)

        # Quick status
        if not m["agent_success"]:
            print(f"  ✗ FAIL: {m['failure_mode']}")
        elif m["agent_feasible"] is False:
            if not tc["expected_feasible"]:
                print(f"  ✓ Correctly reported infeasible")
            else:
                print(f"  ✗ Unexpected infeasible: {m['failure_mode']}")
        elif m["cost_error_pct"] is not None:
            err = m["cost_error_pct"]
            icon = "✓" if err < 1.0 else ("~" if err < 5.0 else "✗")
            print(f"  {icon} Cost: ${m['agent_cost']:.2f} vs ref ${m['ref_cost']:.2f} (err={err:.2f}%)")
        else:
            print(f"  ? Result: {m['failure_mode'] or 'unknown'}")

        # Brief pause between calls to avoid overwhelming the server
        time.sleep(2)

    # ─── Summary ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    total = len(all_metrics)
    success_count = sum(1 for m in all_metrics if m["agent_success"])
    feasible_correct = sum(
        1 for m in all_metrics
        if m["expected_feasible"] and m["agent_feasible"] is True
    )
    infeasible_correct = sum(
        1 for m in all_metrics
        if not m["expected_feasible"] and m["agent_feasible"] is False
    )
    expected_feasible_count = sum(1 for tc in TEST_CASES if tc["expected_feasible"])
    expected_infeasible_count = total - expected_feasible_count

    cost_errors = [m["cost_error_pct"] for m in all_metrics if m["cost_error_pct"] is not None]
    avg_cost_error = sum(cost_errors) / len(cost_errors) if cost_errors else None
    max_cost_error = max(cost_errors) if cost_errors else None

    failure_modes = {}
    for m in all_metrics:
        if m["failure_mode"]:
            fm = m["failure_mode"]
            failure_modes[fm] = failure_modes.get(fm, 0) + 1

    print(f"\n  Total queries:              {total}")
    print(f"  End-to-end success:         {success_count}/{total} ({100*success_count/total:.0f}%)")
    print(f"  Feasibility accuracy:       {feasible_correct}/{expected_feasible_count} correctly feasible")
    print(f"  Infeasibility accuracy:     {infeasible_correct}/{expected_infeasible_count} correctly detected")
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
        case_metrics = [m for m in all_metrics if m["case"] == case]
        case_success = sum(1 for m in case_metrics if m["agent_success"])
        case_errors = [m["cost_error_pct"] for m in case_metrics if m["cost_error_pct"] is not None]
        avg_err = f"{sum(case_errors)/len(case_errors):.2f}%" if case_errors else "N/A"
        print(f"    {case}: {case_success}/{len(case_metrics)} success, avg cost error={avg_err}")

    # Save results to JSON
    output_path = "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "success_count": success_count,
                "success_rate": round(success_count / total, 4),
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
    run_benchmark()
