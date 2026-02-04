import os
from datetime import datetime, timedelta

import pandas as pd

print("\nSTART\n")

# ======================================================
# Paths
# ======================================================
INPUT_PATH = "observational_data/raw_CLIMATERNA_data/health/sim_2010_2022.parquet"
OUTPUT_PATH = (
    "observational_data/processed_data/"
    "deaths_processed_2010-2022.parquet"
)

# ======================================================
# Columns
# ======================================================
COLUMNS_JOIN_SIM = [
    "DTNASC",
    "CODMUNNATU",
    "ESCMAE",
    "IDADEMAE",
    "RACACOR",
    "SEXO",
]

CITY_FALLBACK_COLS = [
    "CODMUNNATU",
    "CODMUNOCOR",
    "CODMUNRES",
    "COMUNSVOIM",
    "CODMUNCART",
]

# ======================================================
# Read data
# ======================================================
print("\nReading files\n")
df = pd.read_parquet(INPUT_PATH, engine="fastparquet")
print("n rows (total deaths):", df.shape[0])

# ======================================================
# Select columns of interest
# ======================================================
df2 = df[
    COLUMNS_JOIN_SIM
    + [
        "IDADE",
        "DTOBITO",
        "HORAOBITO",
        "CODMUNOCOR",
        "CODMUNRES",
        "COMUNSVOIM",
        "CODMUNCART",
    ]
].copy()

# ======================================================
# Process age and filter early neonatal deaths (<24h)
# ======================================================
# IDADE format: first digit = unit (0 minutes, 1 hours), rest = value
df2["IDADE_01"] = pd.to_numeric(df2["IDADE"].str[0], errors="coerce").astype("Int64")
df2["IDADE_02"] = pd.to_numeric(df2["IDADE"].str[1:], errors="coerce").astype("Int64")

# Keep only minutes (0) or hours (1)
df2 = df2.loc[df2["IDADE_01"].isin([0, 1])].copy()
print("n rows (very early neonatal death):", df2.shape[0])

# ======================================================
# Treat missing municipality of birth
# ======================================================
df2["CODMUNNATU_fill"] = (
    df2[CITY_FALLBACK_COLS].bfill(axis=1).iloc[:, 0]
)

# ======================================================
# Construct death datetime
# ======================================================
df2["DTOBITO_dt"] = pd.to_datetime(
    df2["DTOBITO"], format="%d%m%Y", errors="coerce"
)

df2["HORAOBITO_dt"] = (
    pd.to_datetime(
        df2["HORAOBITO"].astype(str).str.zfill(4),
        format="%H%M",
        errors="coerce",
    )
    .dt.time
    .fillna(datetime.strptime("0000", "%H%M").time())
)

df2["OBITO_datetime"] = df2.apply(
    lambda row: datetime.combine(row["DTOBITO_dt"], row["HORAOBITO_dt"]),
    axis=1,
)

# ======================================================
# Compute birth datetime from age
# ======================================================
df2["IDADE_delta"] = df2.apply(
    lambda row: (
        timedelta(minutes=row["IDADE_02"])
        if row["IDADE_01"] == 0
        else timedelta(hours=row["IDADE_02"])
    ),
    axis=1,
)

df2["DTNASC_calc"] = df2["OBITO_datetime"] - df2["IDADE_delta"]

# Format calculated birth date
df2["DTNASC_format"] = df2["DTNASC_calc"].dt.strftime("%d%m%Y")

# ======================================================
# Coalesce original and calculated birth date
# ======================================================
df2["DTNASC_fill"] = (
    df2[["DTNASC", "DTNASC_format"]].bfill(axis=1).iloc[:, 0]
)

# ======================================================
# Final column overwrite and cleanup
# ======================================================
df2["DTNASC"] = df2["DTNASC_fill"]
df2["CODMUNNATU"] = df2["CODMUNNATU_fill"]
df2["ESCMAE"] = df2["ESCMAE"].fillna("9")
df2["IDADEMAE"] = df2["IDADEMAE"].fillna("200").astype(int)
df2["RACACOR"] = df2["RACACOR"].fillna("9")

# ======================================================
# Create unique ID
# ======================================================
df2 = df2.reset_index(drop=True).reset_index(names="id_sim")

# ======================================================
# Save
# ======================================================
print("\nSaving\n")
df2.to_parquet(OUTPUT_PATH, index=False)

print("\nFINISHED\n")
