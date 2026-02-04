from output_results import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =============================================================
# Load results
# =============================================================

df_results = pd.read_parquet('synthetic_data/results/combined_results.parquet')
true_ATE = 2.0


# =============================================================
# Tables
# =============================================================

print('\nDISCRETE DATA TABLE\n')

latex_discrete = df_to_latex(
    df_results[['method'] + [c for c in df_results.columns if c.endswith('_d')]]
        .rename(columns={c: c.replace('_d', '') for c in df_results.columns if c.endswith('_d')}),
    caption="Estimated ATEs and 95\\% Confidence Intervals for the Synthetic Experiment",
    label='tab:synthetic_results'
)
print(latex_discrete)

print('\nCONTINUOUS DATA TABLE\n')

latex_continuous = df_to_latex(
    df_results[['method'] + [c for c in df_results.columns if c.endswith('_c')]]
        .rename(columns={c: c.replace('_c', '') for c in df_results.columns if c.endswith('_c')}),
    caption="Estimated ATEs and 95\\% Confidence Intervals for the Synthetic Experiment with Continuous Data",
    label='tab:synthetic_resultsContinuous'
)
print(latex_continuous)



# =============================================================
# Plot 1 — Discrete vs Continuous
# =============================================================

df_graph = df_results.loc[df_results['method'] != 'true_ate']

fig, ax = plt.subplots(figsize=(12, 6))

width = 0.30
offset = (width / 2) + 0.01

color_disc = "#94b2c7"
color_cont = "#e7b68b"

hatch_disc = '...'
hatch_cont = '\\\\\\'

lw_true = 1
lw_false = 2.5

# Discrete
ax.bar(
    df_graph.index - offset,
    df_graph['value_d'],
    width=width,
    label='Discrete',
    edgecolor='black',
    lw=0.5,
    color=color_disc,
    hatch=hatch_disc,
    zorder=3
)

plot_ci_lines(
    df_graph.index - offset,
    df_graph['value_d'],
    df_graph['ci_low_d'],
    df_graph['ci_high_d'],
    color_if_true='black',
    color_if_false='black',
    lw_if_true=lw_true,
    lw_if_false=lw_false,
    true_value=2,
    ax=ax
)

# Continuous
ax.bar(
    df_graph.index + offset,
    df_graph['value_c'],
    width=width,
    label='Continuous',
    edgecolor='black',
    lw=0.5,
    color=color_cont,
    hatch=hatch_cont,
    zorder=3
)

plot_ci_lines(
    df_graph.index + offset,
    df_graph['value_c'],
    df_graph['ci_low_c'],
    df_graph['ci_high_c'],
    color_if_true='black',
    color_if_false='black',
    lw_if_true=lw_true,
    lw_if_false=lw_false,
    true_value=2,
    ax=ax
)

add_true_ate_line(ax, true_ATE)

ax.set_xticks(df_graph.index)
ax.set_xticklabels(wrapped_method_labels(df_graph), rotation=90, ha='center')

ax.set_ylabel("Estimated values")
ax.grid(axis='y', linestyle=':', alpha=0.3, zorder=0)

add_ci_legend(ax, lw_true, lw_false)

fig.tight_layout()
plt.show()
fig.savefig(
    'synthetic_data/results/synthetic_results_combined.png', 
    dpi=300, 
    bbox_inches='tight'
    )

# =============================================================
# Plot 2 — ONLY DISCRETE
# =============================================================

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    df_graph.index,
    df_graph['value_d'],
    width=width,
    edgecolor='black',
    lw=0.5,
    color=color_disc,
    hatch=hatch_disc,
    zorder=3
)

plot_ci_lines(
    df_graph.index,
    df_graph['value_d'],
    df_graph['ci_low_d'],
    df_graph['ci_high_d'],
    color_if_true='black',
    color_if_false='black',
    lw_if_true=lw_true,
    lw_if_false=lw_false,
    true_value=2,
    ax=ax
)

add_true_ate_line(ax, true_ATE)

ax.set_xticks(df_graph.index)
ax.set_xticklabels(wrapped_method_labels(df_graph), rotation=90, ha='center')

ax.set_ylabel("Estimated values")
ax.grid(axis='y', linestyle=':', alpha=0.3, zorder=0)

add_ci_legend(ax, lw_true, lw_false)

fig.tight_layout()
plt.show()

fig.savefig(
    'synthetic_data/results/synthetic_results_discrete.png',
    dpi=300,
    bbox_inches='tight'
)


# =============================================================
# Plot 3 — ONLY CONTINUOUS
# =============================================================

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    df_graph.index,
    df_graph['value_c'],
    width=width,
    edgecolor='black',
    lw=0.5,
    color=color_cont,
    hatch=hatch_cont,
    zorder=3
)

plot_ci_lines(
    df_graph.index,
    df_graph['value_c'],
    df_graph['ci_low_c'],
    df_graph['ci_high_c'],
    color_if_true='black',
    color_if_false='black',
    lw_if_true=lw_true,
    lw_if_false=lw_false,
    true_value=2,
    ax=ax
)

add_true_ate_line(ax, true_ATE)

ax.set_xticks(df_graph.index)
ax.set_xticklabels(wrapped_method_labels(df_graph), rotation=90, ha='center')

ax.set_ylabel("Estimated values")
ax.set_ylim(top=3.5)
ax.grid(axis='y', linestyle=':', alpha=0.3, zorder=0)

add_ci_legend(ax, lw_true, lw_false)

fig.tight_layout()
plt.show()

fig.savefig(
    'synthetic_data/results/synthetic_results_continuous.png',
    dpi=300,
    bbox_inches='tight'
)
