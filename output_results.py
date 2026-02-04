import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.lines as mlines
import textwrap
45

# ----------------------------------------
# Method name mapping (for tables and plots)
# ----------------------------------------
method_map = {
    "true_ate": "True ATE",
    "naive": "Naive difference in means",
    "adjustment_z": "Adjustment formula (Z)",
    "adjustment_zw": "Adjustment formula (Z, W)",
    "adjustment_w": "Adjustment formula (W)",
    "linreg_causal_zw": r"Linear regression ($Y \sim D + Z + W$)",
    "linreg_causal_z": r"Linear regression ($Y \sim D + Z$)",
    "linreg_causal_w": r"Linear regression ($Y \sim D + W$)",
    "linreg_potentialoutcome": "Linear regression: outcome modeling",
    "ipw": "Inverse Probability Weighting (IPW)",
    "ipw_stabilized": "IPW with stabilized weights",
    "ps_linreg": "Linear regression with propensity score",
    "ps_matching": "Propensity score matching",
    "double_robust": "Doubly robust",
}


# ----------------------------------------
# LaTeX table formatting
# ----------------------------------------
def df_to_latex(
    df,
    method_map=method_map,
    caption="Estimates and confidence intervals",
    label="tab:results",
    number_format=".4f",
):
    """
    Convert a results DataFrame into a formatted LaTeX table.

    Expected columns:
        ['method', 'value', 'ci_low', 'ci_high']
    """

    df_formatted = df.copy()

    # Method names
    df_formatted["Method"] = df_formatted["method"].replace(method_map)

    # Point estimates
    df_formatted["Estimate"] = df_formatted["value"].apply(
        lambda x: f"{x:{number_format}}"
    )

    # Confidence intervals
    df_formatted["Confidence Interval"] = df_formatted.apply(
        lambda r: f"({r['ci_low']:{number_format}}, {r['ci_high']:{number_format}})",
        axis=1,
    )

    df_latex = df_formatted[
        ["Method", "Estimate", "Confidence Interval"]
    ].reset_index(drop=True)

    latex_str = (
        df_latex.style.hide(axis=0)
        .to_latex(
            hrules=True,
            caption=caption,
            label=label,
        )
    )

    # Clean NaN intervals
    latex_str = latex_str.replace("(nan, nan)", "")

    # Center table
    latex_str = latex_str.replace(
        r"\begin{table}",
        r"\begin{table}[htb]" + "\n\\centering",
    )

    return latex_str


# ----------------------------------------
# Plot helpers
# ----------------------------------------
def plot_ci_lines(
    x,
    y,
    low,
    high,
    true_value,
    ax,
    color_if_true="green",
    color_if_false="red",
    lw_if_true=2,
    lw_if_false=2,
    cap_width=0.08,
    zorder=30,
):
    """Draw confidence interval lines with caps."""
    if true_value is None:
        contains_true = np.full(low.shape, True)
    else:
        contains_true = (low <= true_value) & (high >= true_value)

    for xi, lo, hi, ok in zip(x, low, high, contains_true):
        color = color_if_true if ok else color_if_false
        lw = lw_if_true if ok else lw_if_false

        ax.plot([xi, xi], [lo, hi], color=color, linewidth=lw, zorder=zorder)
        ax.plot([xi - cap_width, xi + cap_width], [lo, lo], color=color, linewidth=lw, zorder=zorder)
        ax.plot([xi - cap_width, xi + cap_width], [hi, hi], color=color, linewidth=lw, zorder=zorder)


def add_labels(bars, ax, zorder=35):
    """Add numeric labels to bar plots."""

    for bar in bars:
        height = bar.get_height()
        if pd.isnull(height):
            continue

        xpos = bar.get_x() + bar.get_width() / 2
        ypos = height if height >= 0 else 0.07
        va = "bottom" if height >= 0 else "top"

        ax.text(
            xpos,
            ypos,
            f"{height:.2f}",
            ha="center",
            va=va,
            fontsize=6,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.4,
                boxstyle="round,pad=0.2",
            ),
            zorder=zorder,
        )

# =============================================================
# Common plotting helpers
# =============================================================

def wrapped_method_labels(df):
    return [
        "\n".join(textwrap.wrap(method_map.get(m, m), width=20))
        for m in df["method"]
    ]


def add_true_ate_line(ax, true_ATE):
    ax.axhline(
        true_ATE,
        color='red',
        linestyle='--',
        linewidth=1,
        zorder=4
    )

    ax.text(
        x=ax.get_xlim()[1] * 1.005,
        y=true_ATE,
        s=r"$\blacktriangleleft$ True ATE",
        va='center',
        ha='left',
        fontsize=9,
        color='red',
        clip_on=False
    )


def add_ci_legend(ax, lw_true, lw_false):
    ci_contains = mlines.Line2D([], [], color='black', linewidth=lw_true,
                               label='CI contains true ATE')
    ci_misses = mlines.Line2D([], [], color='black', linewidth=lw_false,
                             label='CI misses true ATE')

    handles, labels = ax.get_legend_handles_labels()
    handles += [ci_contains, ci_misses]
    labels += [ci_contains.get_label(), ci_misses.get_label()]

    ax.legend(handles, labels, loc='upper left', frameon=True)

