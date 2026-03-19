import pandapower.networks as pn
import pandas as pd


def load_case_data(command):

    case_map = {
        "IEEE14": pn.case14,
        "IEEE30": pn.case30,
        "IEEE57": pn.case57,
        "IEEE118": pn.case118,
        "IEEE200": pn.case_illinois200,
        "IEEE300": pn.case300,
    }

    # 1. 算例匹配 (match longest key first to avoid IEEE30 matching IEEE300)
    selected_case = None
    cmd_upper = command.upper()
    for key in sorted(case_map.keys(), key=len, reverse=True):
        if key in cmd_upper:
            selected_case = key
            break

    if not selected_case:
        return f"Error: Unsupported IEEE case requested in command: '{command}'."

    try:
        net = case_map[selected_case]()

        gen_df = net.gen[['bus', 'max_p_mw', 'min_p_mw']].copy()


        cost_df = net.poly_cost[net.poly_cost.et == 'gen'].copy()


        merged = gen_df.join(cost_df[['cp0_eur', 'cp1_eur_per_mw', 'cp2_eur_per_mw2']])


        total_load = net.load.p_mw.sum()


        output = f"--- System Data for {selected_case} ---\n"
        output += f"Target Total System Load (Pd): {total_load:.2f} MW\n\n"
        output += "Generator Parameters (Status is active by default):\n"

        for i, row in merged.iterrows():
            output += (f"Unit ID {i}: Bus={int(row.bus)}, Pmin={row.min_p_mw}MW, Pmax={row.max_p_mw}MW, "
                       f"Cost(a={row.cp2_eur_per_mw2}, b={row.cp1_eur_per_mw}, c={row.cp0_eur})\n")


        output += ("\nSystem Message: Data retrieved successfully. Please extract these parameters into the required JSON format and proceed to 'solve_ed'"+
                   "⚠️ WARNING: Units with Cost=nan MUST be set to status=0 (offline) and excluded from the cvxpy call. Do NOT replace nan with 0.0. Please extract valid units only and proceed to 'solve_ed'.")

        return output

    except Exception as e:
        return f"Error loading data for {selected_case}: {str(e)}"
