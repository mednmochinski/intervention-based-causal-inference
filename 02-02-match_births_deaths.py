import os
import pandas as pd

print("\nSTART\n")

# ======================================================
# Parameters
# ======================================================
YEARS = [2022, 2021, 2020, 2019, 2018]

# Columns used for linkage
COLUMNS_JOIN_SIM = [
    "DTNASC",
    "CODMUNNATU",  # municipality of birth
    "IDADEMAE",
    "RACACOR",
    "SEXO",
]

# SINASC uses a different name for municipality of birth
COLUMN_RENAME_SIM_TO_SINASC = {
    "CODMUNNATU": "CODMUNNASC"
}

COLUMNS_JOIN_SINASC = [
    COLUMN_RENAME_SIM_TO_SINASC.get(col, col)
    for col in COLUMNS_JOIN_SIM
]

# ======================================================
# Paths
# ======================================================
SIM_PATH = (
    "experiments/observational_data/processed/"
    "deaths_processed_2010-2022.parquet"
)

SINASC_PATH = (
    "experiments/observational_data/processed/"
    "births_processed_2010-2022.parquet"
)

# ======================================================
# Read deaths (SIM)
# ======================================================
print("\nReading files: DEATHS (SIM)\n")
df_sim = pd.read_parquet(SIM_PATH, engine="fastparquet")

df_sim["id_sim"] = df_sim["id_sim"].astype(int)
df_sim["DTNASC"] = pd.to_datetime(df_sim["DTNASC"], format="%d%m%Y")

print("SIM shape (before year filter):", df_sim.shape)
df_sim = df_sim.loc[df_sim["DTNASC"].dt.year.isin(YEARS)]
print("SIM shape (after year filter): ", df_sim.shape)

# ======================================================
# Read births (SINASC)
# ======================================================
print("\nReading files: BIRTHS (SINASC)\n")
df_sinasc = pd.read_parquet(SINASC_PATH, engine="fastparquet")

df_sinasc["id_sinasc"] = df_sinasc["id_sinasc"].astype(int)
df_sinasc["DTNASC"] = pd.to_datetime(df_sinasc["DTNASC"], format="%d%m%Y")

print("SINASC shape (before year filter):", df_sinasc.shape)
df_sinasc = df_sinasc.loc[df_sinasc["DTNASC"].dt.year.isin(YEARS)]
print("SINASC shape (after year filter): ", df_sinasc.shape)

# ======================================================
# First join: potential matches
# ======================================================
df_join = df_sinasc.merge(
    df_sim,
    how="left",
    left_on=COLUMNS_JOIN_SINASC,
    right_on=COLUMNS_JOIN_SIM,
    suffixes=("_sinasc", "_sim"),
)

# ======================================================
# Aggregate possible matches per death record
# ======================================================
df_matches = (
    df_join
    .groupby("id_sim")
    .agg(
        sinasc_count=("id_sinasc", "count"),
        sinasc_list=("id_sinasc", lambda x: sorted(x.dropna().tolist())),
    )
    .sort_values("id_sim")
    .reset_index()
)

# ======================================================
# Greedy one-to-one matching
# ======================================================
id_sim_match = []
id_sinasc_match = []

for _, row in df_matches.iterrows():
    # available SINASC ids not yet matched
    available_matches = sorted(
        set(row["sinasc_list"]) - set(id_sinasc_match)
    )

    if not available_matches:
        continue

    id_sim_match.append(int(row["id_sim"]))
    id_sinasc_match.append(available_matches[0])

df_matches_clean = pd.DataFrame(
    {
        "id_sim": id_sim_match,
        "id_sinasc": id_sinasc_match,
    }
)

# ======================================================
# Diagnostics
# ======================================================
print("\nMatch diagnostics\n")

print("id_sim total:        ", df_sim["id_sim"].nunique())
print("id_sim with any hit: ", df_join["id_sim"].nunique())
print("id_sim matched:     ", df_matches_clean["id_sim"].nunique())

print("id_sinasc total:    ", df_sinasc["id_sinasc"].nunique())
print("id_sinasc matched:  ", df_matches_clean["id_sinasc"].nunique())

# ======================================================
# Save
# ======================================================
print("\nSave\n")

df_matches_clean.to_parquet(
    "observational_data/processed_data/"
    "match_birth_death_2018-2022.parquet",
    index=False,
)

print("\nFINISHED\n")
