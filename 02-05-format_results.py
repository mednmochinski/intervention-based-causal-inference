from output_results import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =============================================================
# Load results
# =============================================================
df_results = pd.read_csv('observational_data/results/causal_results_2018-2022_100r_20260204-165329.csv')

print(df_results.head())

# =============================================================
# Tables
# =============================================================

print('\nOBSERVATIONAL DATA TABLE\n')

latex_observational = df_to_latex(
    df_results,
    caption="Estimated ATEs and 95\\% Confidence Intervals for exposure to extreme heat on very early neonatal mortality",
    label='tab:realWorldResults',
    number_format='.4e'
)
print(latex_observational)


width = 0.30

color = "#b68bc5"

hatch = '...'

lw_true = 2.5
lw_false = 2.5


# =============================================================
# Plot 
# =============================================================

df_graph = df_results

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    df_graph.index,
    df_graph['value'],
    width=width,
    edgecolor='black',
    lw=0.5,
    color=color,
    hatch=hatch,
    zorder=3
)

plot_ci_lines(
    df_graph.index,
    df_graph['value'],
    df_graph['ci_low'],
    df_graph['ci_high'],
    color_if_true='black',
    color_if_false='black',
    lw_if_true=lw_true,
    lw_if_false=lw_false,
    true_value=None,
    ax=ax
)

# add_true_ate_line(ax, true_ATE)

ax.set_xticks(df_graph.index)
ax.set_xticklabels(wrapped_method_labels(df_graph), rotation=90, ha='center')

ax.set_ylabel("Estimated values")
ax.grid(axis='y', linestyle=':', alpha=0.3, zorder=0)

# add_ci_legend(ax, lw_true, lw_false)

fig.tight_layout()
plt.show()

fig.savefig(
    'observational_data/results/observational_results.png',
    dpi=300,
    bbox_inches='tight'
)

