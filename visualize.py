"""
Economic Dispatch Result Visualization
Chart 1: Generator dispatch bar chart
Chart 2: Load distribution pie chart + system summary
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive mode, required for FastAPI environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
import base64
import io


def parse_cvxpy_result(result):
    """Parse cvxpy tool output, compatible with np.float64."""
    if isinstance(result, str):
        result = json.loads(result)
    dispatch = {int(k): float(v) for k, v in result['dispatch_results'].items()}
    total_cost = float(result['total_cost'])
    status = result['status']
    return dispatch, total_cost, status


def plot_dispatch_bar(dispatch: dict, generators: list, case_name: str = "IEEE") -> plt.Figure:
    """Chart 1: Generator power output bar chart (dispatch vs Pmax)."""
    gen_map = {g['id']: g for g in generators if g.get('status', 1) == 1}
    ids = sorted(dispatch.keys())
    p_dispatch = [dispatch[i] for i in ids]
    p_max = [gen_map[i]['p_max'] if i in gen_map else 0 for i in ids]

    x = np.arange(len(ids))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(12, len(ids) * 0.4), 6))
    ax.bar(x, p_max, width, color='#d0e8f1', zorder=1)
    colors = ['#e74c3c' if abs(p_dispatch[i] - p_max[i]) < 0.1 else '#2ecc71'
              for i in range(len(ids))]
    ax.bar(x, p_dispatch, width * 0.7, color=colors, zorder=2)

    ax.set_xlabel('Generator ID', fontsize=12)
    ax.set_ylabel('Power Output (MW)', fontsize=12)
    ax.set_title(f'{case_name} - Generator Dispatch Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{i}' for i in ids], rotation=90 if len(ids) > 20 else 45)
    ax.grid(axis='y', alpha=0.3)

    at_max = mpatches.Patch(color='#e74c3c', label='At Pmax (saturated)')
    partial = mpatches.Patch(color='#2ecc71', label='Partial dispatch')
    bg = mpatches.Patch(color='#d0e8f1', label='Pmax capacity')
    ax.legend(handles=[at_max, partial, bg], loc='upper right')

    plt.tight_layout()
    return fig


def plot_load_pie(dispatch: dict, generators: list, total_load: float,
                  case_name: str = "IEEE") -> plt.Figure:
    """Chart 2: Load distribution pie chart grouped by cost coefficient + system summary."""
    groups = {}
    for g in generators:
        if g.get('status', 1) == 1 and g['id'] in dispatch:
            b_key = f"b={g['cost']['b']:.0f}"
            groups.setdefault(b_key, 0)
            groups[b_key] += dispatch[g['id']]

    offline = sum(1 for g in generators if g.get('status', 1) == 0)
    labels = list(groups.keys())
    sizes = list(groups.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax1.set_title(f'{case_name} - Load Distribution by Cost Group', fontsize=12, fontweight='bold')

    total_gen = sum(sizes)
    ax2.axis('off')
    summary = (
        f"System Summary\n"
        f"{'─' * 30}\n"
        f"Total Load:       {total_load:.2f} MW\n"
        f"Total Generation: {total_gen:.2f} MW\n"
        f"Balance Error:    {abs(total_load - total_gen):.4f} MW\n"
        f"Active Units:     {len(dispatch)}\n"
        f"Offline Units:    {offline}\n"
        f"{'─' * 30}\n"
    )
    for label, mw in sorted(groups.items(), key=lambda x: -x[1]):
        summary += f"{label:>8}: {mw:>8.2f} MW\n"

    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    return fig


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def visualize_all(cvxpy_result, generators: list, total_load: float,
                  case_name: str = "IEEE118") -> str:
    """Generate Chart 1 and Chart 2, save to output_charts/, return file paths.

    Called by the visualization() tool function.
    """
    dispatch, total_cost, status = parse_cvxpy_result(cvxpy_result)
    print(f"Status: {status} | Total Cost: ${total_cost:,.2f}/hr | Active Units: {len(dispatch)}")

    os.makedirs("output_charts", exist_ok=True)
    path1 = f"output_charts/{case_name}_dispatch.png"
    path2 = f"output_charts/{case_name}_pie.png"

    fig1 = plot_dispatch_bar(dispatch, generators, case_name)
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    fig2 = plot_load_pie(dispatch, generators, total_load, case_name)
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    return (
        f"Visualization complete:\n"
        f"  Chart 1 (dispatch bar) -> {path1}\n"
        f"  Chart 2 (load pie)     -> {path2}"
    )


def visualization(tool_input: str) -> str:
    """Tool function registered to ToolExecutor.

    Input JSON:
    {
        "cvxpy_result": {...},
        "generators":   [...],
        "total_load":   4242.0,
        "case_name":    "IEEE118"   # optional
    }
    """
    try:
        if isinstance(tool_input, str):
            params = json.loads(tool_input)
        else:
            params = tool_input

        cvxpy_result = params.get("cvxpy_result")
        generators   = params.get("generators")
        total_load   = float(params.get("total_load", 0))
        case_name    = params.get("case_name", "IEEE")

        if not cvxpy_result:
            return "❌ Missing parameter: cvxpy_result"
        if not generators:
            return "❌ Missing parameter: generators"
        if total_load == 0:
            return "❌ Missing parameter: total_load"

        return visualize_all(cvxpy_result, generators, total_load, case_name)

    except Exception as e:
        return f"❌ Visualization failed: {e}"
