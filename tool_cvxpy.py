import json
import re
import math
import cvxpy as cp
import numpy as np


def extract_json(response_text):
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(response_text)
    except Exception as e:
        print(f"JSON 解析失败: {e}")
        return None


def solve_ed_from_json(ed_json):
    try:
        P_d = ed_json['target_load_mw']

        # about nan
        all_generators = ed_json['generators']
        generators = []
        excluded = []

        for g in all_generators:
            if g['status'] != 1:
                continue
            a = g['cost']['a']
            b = g['cost']['b']
            c = g['cost'].get('c', 0.0)
            if any(math.isnan(v) for v in [a, b, c]):
                excluded.append(g['id'])
                print(f"⚠️ Unit {g['id']} 因 cost=nan 被排除出经济调度")
            else:
                generators.append(g)

        n = len(generators)
        if n == 0:
            return {"status": "Error", "error": "No active generators with valid cost coefficients"}

        pg = cp.Variable(n)

        # 约束
        constraints = []
        constraints.append(cp.sum(pg) == P_d)
        p_min = np.array([g['p_min'] for g in generators])
        p_max = np.array([g['p_max'] for g in generators])
        constraints.append(pg >= p_min)
        constraints.append(pg <= p_max)

        costs = []
        for i in range(n):
            a = generators[i]['cost']['a']
            b = generators[i]['cost']['b']
            c = generators[i]['cost'].get('c', 0.0)
            costs.append(a * cp.square(pg[i]) + b * pg[i] + c)

        objective = cp.Minimize(cp.sum(costs))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        if prob.status == cp.OPTIMAL:
            result = {
                "status": "Success",
                "total_cost": prob.value,
                "dispatch_results": {generators[i]['id']: pg.value[i] for i in range(n)}
            }
            if excluded:
                result["excluded_units"] = excluded
                result["note"] = f"Unit(s) {excluded} excluded due to missing cost data (nan)"
            return result
        else:
            return {"status": "Infeasible", "error": prob.status}

    except Exception as e:
        return {"status": "Error", "error": str(e)}


def cvxpy(command):
    json_data = extract_json(command)
    return solve_ed_from_json(json_data)
