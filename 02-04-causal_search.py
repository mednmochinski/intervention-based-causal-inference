import datetime
import pandas as pd

from causal_estimators import *
from aux_functions import *

# ======================================================
# Configuration
# ======================================================
DATA_PATH = (
    "observational_data/processed_data/"
    "climate_births_deaths_2018-2022.parquet"
)

RESULTS_DIR = "observational_data/results/"


ROUNDS = 100
N_JOBS = 8
N_JOBS_KNN = 8  # used only for PS matching (currently disabled)

print("\nSTART")
print(datetime.datetime.now())

# ======================================================
# Load and preprocess data
# ======================================================
df = pd.read_parquet(
    DATA_PATH,
    columns=[
        "risk_score",
        "IDANOMAL",
        "heat_event",
        "early_neonatal_death",
        "DATA",
    ],
)

df["YEAR"] = pd.to_datetime(df["DATA"]).dt.year

# Keep valid observations only
df = df.loc[
    (df["IDANOMAL"] != "9") &
    (df["heat_event"].notna())
].copy()

# Cast types
df["IDANOMAL"] = df["IDANOMAL"].astype(int)
df["heat_event"] = df["heat_event"].astype(int)

# ======================================================
# Define causal variables
# ======================================================
df["Z"] = df["risk_score"]              # socioeconomic risk
df["W"] = df["IDANOMAL"]                # congenital anomaly
df["D"] = df["heat_event"]              # treatment
df["Y"] = df["early_neonatal_death"]    # outcome


df_calc = df[["Z", "W", "D", "Y"]].copy()

print("Analysis dataset shape:", df_calc.shape)

# ======================================================
# Continuous Data Experiments
# ======================================================
print("\nOBSERVATIONAL DATA EXPERIMENTS\n")

results = {}


# ------------------------------------------------------
# Linear regression (causal)
# ------------------------------------------------------
log_step("Linear regression (D + Z + W)")
results["linreg_causal_zw"] = bootstrap(
    df_calc,
    linreg_causal_estimator,
    model_exp="D+Z+W",
    rounds=ROUNDS,
    n_jobs=N_JOBS,
)
print("Estimated ATE:", results["linreg_causal_zw"])

log_step("Linear regression (D + Z)")
results["linreg_causal_z"] = bootstrap(
    df_calc,
    linreg_causal_estimator,
    model_exp="D+Z",
    rounds=ROUNDS,
    n_jobs=N_JOBS,
)
print("Estimated ATE:", results["linreg_causal_z"])

log_step("Linear regression (D + W)")
results["linreg_causal_w"] = bootstrap(
    df_calc,
    linreg_causal_estimator,
    model_exp="D+W",
    rounds=ROUNDS,
    n_jobs=N_JOBS,
)
print("Estimated ATE:", results["linreg_causal_w"])

# ------------------------------------------------------
# Potential outcome model
# ------------------------------------------------------
log_step("Linear regression (potential outcomes)")
results["linreg_potentialoutcome"] = bootstrap(
    df_calc,
    linreg_potentialoutcome_estimator,
    rounds=ROUNDS,
    n_jobs=N_JOBS,
)
print("Estimated ATE:", results["linreg_potentialoutcome"])

# ------------------------------------------------------
# IPW estimators
# ------------------------------------------------------
log_step("IPW")
results["ipw"] = bootstrap(
    df_calc,
    ipw_estimator,
    rounds=ROUNDS,
    n_jobs=N_JOBS,
)
print("Estimated ATE:", results["ipw"])

log_step("IPW (stabilized)")
results["ipw_stabilized"] = bootstrap(
    df_calc,
    ipw_stabilized_estimator,
    rounds=ROUNDS,
    n_jobs=N_JOBS,
)
print("Estimated ATE:", results["ipw_stabilized"])

# ------------------------------------------------------
# Propensity score regression
# ------------------------------------------------------
log_step("Propensity score regression")
results["ps_linreg"] = bootstrap(
    df_calc,
    ps_linreg_estimator,
    rounds=ROUNDS,
    n_jobs=N_JOBS,
)
print("Estimated ATE:", results["ps_linreg"])

# ------------------------------------------------------
# Propensity score matching (intentionally disabled)
# ------------------------------------------------------
# log_step("Propensity score matching")
# results["ps_matching"] = bootstrap(
#     df_calc,
#     ps_matching_estimator,
#     rounds=ROUNDS,
#     n_jobs=N_JOBS,
#     n_jobs_knn=N_JOBS_KNN,
# )
# print("Estimated ATE:", results["ps_matching"])

# ------------------------------------------------------
# Doubly robust estimator
# ------------------------------------------------------
log_step("Doubly robust estimator")
results["double_robust"] = bootstrap(
    df_calc,
    double_robust_estimator,
    rounds=ROUNDS,
    n_jobs=N_JOBS,
)
print("Estimated ATE:", results["double_robust"])

# ======================================================
# Save results
# ======================================================
df_results = results_to_df(results)

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
filename = (
    f"{RESULTS_DIR}"
    f"causal_results_2018-2022_"
    f"{ROUNDS}r_{timestamp}.csv"
)

df_results.to_csv(filename, index=False)

print("\nSaved results to:")
print(filename)
print("\nFinal results:")
print(df_results)

print("\nFINISHED")
