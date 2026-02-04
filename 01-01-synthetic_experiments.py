print("start")

import numpy as np
import pandas as pd

import generate_data as gd
import aux_functions as aux
import causal_estimators as csl


# ------------------------------------------------
# Configuration
# ------------------------------------------------
n = 10_000
seed = 1944

np.random.seed(seed)


# -------------------------------------
# Discrete data experiments
# -------------------------------------
print("\n\nDISCRETE DATA EXPERIMENTS\n")

df_discrete, true_ATE = gd.generate_data_discrete(n=n)
df_discrete.to_parquet(
    "synthetic_data/datasets/discrete_experiment.parquet",
    index=False,
)


print(f"Simulated {len(df_discrete)} observations.\n")
print(f"True ATE = {true_ATE}\n")

results_discrete = {}
results_discrete["true_ate"] = true_ATE


aux.log_step("Naive estimator")
results_discrete["naive"] = aux.bootstrap(
    df_discrete,
    csl.naive_estimator,
)
print("Estimated ATE:", results_discrete["naive"])


aux.log_step("Adjustment formula (confounders: Z)")
results_discrete["adjustment_z"] = aux.bootstrap(
    df_discrete,
    csl.adjustment_formula_estimator,
    adjustment_set=["Z"],
)
print("Estimated ATE:", results_discrete["adjustment_z"])


aux.log_step("Adjustment formula (confounders: Z, W)")
results_discrete["adjustment_zw"] = aux.bootstrap(
    df_discrete,
    csl.adjustment_formula_estimator,
    adjustment_set=["Z", "W"],
)
print("Estimated ATE:", results_discrete["adjustment_zw"])


aux.log_step("Adjustment formula (confounders: W)")
results_discrete["adjustment_w"] = aux.bootstrap(
    df_discrete,
    csl.adjustment_formula_estimator,
    adjustment_set=["W"],
)
print("Estimated ATE:", results_discrete["adjustment_w"])


aux.log_step("Linear regression (causal estimate, confounders: Z, W)")
results_discrete["linreg_causal_zw"] = aux.bootstrap(
    df_discrete,
    csl.linreg_causal_estimator,
    model_exp="D + Z + W",
)
print("Estimated ATE:", results_discrete["linreg_causal_zw"])


aux.log_step("Linear regression (causal estimate, confounders: Z)")
results_discrete["linreg_causal_z"] = aux.bootstrap(
    df_discrete,
    csl.linreg_causal_estimator,
    model_exp="D + Z",
)
print("Estimated ATE:", results_discrete["linreg_causal_z"])


aux.log_step("Linear regression (causal estimate, confounders: W)")
results_discrete["linreg_causal_w"] = aux.bootstrap(
    df_discrete,
    csl.linreg_causal_estimator,
    model_exp="D + W",
)
print("Estimated ATE:", results_discrete["linreg_causal_w"])


aux.log_step("Linear regression (potential outcome model)")
results_discrete["linreg_potentialoutcome"] = aux.bootstrap(
    df_discrete,
    csl.linreg_potentialoutcome_estimator,
)
print("Estimated ATE:", results_discrete["linreg_potentialoutcome"])


aux.log_step("IPW")
results_discrete["ipw"] = aux.bootstrap(
    df_discrete,
    csl.ipw_estimator,
)
print("Estimated ATE:", results_discrete["ipw"])


aux.log_step("IPW stabilized")
results_discrete["ipw_stabilized"] = aux.bootstrap(
    df_discrete,
    csl.ipw_stabilized_estimator,
)
print("Estimated ATE:", results_discrete["ipw_stabilized"])


aux.log_step("Linear regression using propensity score")
results_discrete["ps_linreg"] = aux.bootstrap(
    df_discrete,
    csl.ps_linreg_estimator,
)
print("Estimated ATE:", results_discrete["ps_linreg"])


aux.log_step("Propensity score matching")
results_discrete["ps_matching"] = aux.bootstrap(
    df_discrete,
    csl.ps_matching_estimator,
)
print("Estimated ATE:", results_discrete["ps_matching"])


aux.log_step("Doubly robust estimator")
results_discrete["double_robust"] = aux.bootstrap(
    df_discrete,
    csl.double_robust_estimator,
)
print("Estimated ATE:", results_discrete["double_robust"])


df_rd = aux.results_to_df(results_discrete)


# -------------------------------------
# Continuous data experiments
# -------------------------------------
print("\n\nCONTINUOUS DATA EXPERIMENTS\n")

df_continuous, true_ATE = gd.generate_data_continuous(n=n)
df_continuous.to_parquet(
    "synthetic_data/datasets/continuous_experiment.parquet",
    index=False,
)

print(f"Simulated {len(df_continuous)} observations.\n")
print(f"True ATE = {true_ATE}\n")

results_continuous = {}
results_continuous["true_ate"] = true_ATE


aux.log_step("Naive estimator")
results_continuous["naive"] = aux.bootstrap(
    df_continuous,
    csl.naive_estimator,
)
print("Estimated ATE:", results_continuous["naive"])


aux.log_step("Linear regression (causal estimate, confounders: Z, W)")
results_continuous["linreg_causal_zw"] = aux.bootstrap(
    df_continuous,
    csl.linreg_causal_estimator,
    model_exp="D + Z + W",
)
print("Estimated ATE:", results_continuous["linreg_causal_zw"])


aux.log_step("Linear regression (causal estimate, confounders: Z)")
results_continuous["linreg_causal_z"] = aux.bootstrap(
    df_continuous,
    csl.linreg_causal_estimator,
    model_exp="D + Z",
)
print("Estimated ATE:", results_continuous["linreg_causal_z"])


aux.log_step("Linear regression (causal estimate, confounders: W)")
results_continuous["linreg_causal_w"] = aux.bootstrap(
    df_continuous,
    csl.linreg_causal_estimator,
    model_exp="D + W",
)
print("Estimated ATE:", results_continuous["linreg_causal_w"])


aux.log_step("Linear regression (potential outcome model)")
results_continuous["linreg_potentialoutcome"] = aux.bootstrap(
    df_continuous,
    csl.linreg_potentialoutcome_estimator,
)
print("Estimated ATE:", results_continuous["linreg_potentialoutcome"])


aux.log_step("IPW")
results_continuous["ipw"] = aux.bootstrap(
    df_continuous,
    csl.ipw_estimator,
)
print("Estimated ATE:", results_continuous["ipw"])


aux.log_step("IPW stabilized")
results_continuous["ipw_stabilized"] = aux.bootstrap(
    df_continuous,
    csl.ipw_stabilized_estimator,
)
print("Estimated ATE:", results_continuous["ipw_stabilized"])


aux.log_step("Linear regression using propensity score")
results_continuous["ps_linreg"] = aux.bootstrap(
    df_continuous,
    csl.ps_linreg_estimator,
)
print("Estimated ATE:", results_continuous["ps_linreg"])


aux.log_step("Propensity score matching")
results_continuous["ps_matching"] = aux.bootstrap(
    df_continuous,
    csl.ps_matching_estimator,
)
print("Estimated ATE:", results_continuous["ps_matching"])


aux.log_step("Doubly robust estimator")
results_continuous["double_robust"] = aux.bootstrap(
    df_continuous,
    csl.double_robust_estimator,
)
print("Estimated ATE:", results_continuous["double_robust"])


df_rc = aux.results_to_df(results_continuous)


# ----------------------------------
# Consolidate results
# ----------------------------------
df_full = df_rd.merge(
    df_rc,
    how="outer",
    on="method",
    suffixes=["_d", "_c"],
)

df_full.to_parquet(
    "synthetic_data/results/combined_results.parquet",
    index=False,
)
