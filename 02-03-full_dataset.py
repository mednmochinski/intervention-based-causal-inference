import numpy as np
import pandas as pd

print("\nSTART\n")

# ======================================================
# Paths
# ======================================================
PROCESSED_FOLDER ="observational_data/processed_data/"

CLIMATE_PATH = (
    PROCESSED_FOLDER+
    "climate_processed_2010-2024.parquet"
)

BIRTHS_PATH = (
    PROCESSED_FOLDER+
    "births_processed_2010-2022.parquet"
)

MATCH_PATH = (
    PROCESSED_FOLDER+
    "match_birth_death_2018-2022.parquet"
)

OUTPUT_PATH = (
    PROCESSED_FOLDER+
    "climate_births_deaths_2018-2022.parquet"
)

YEARS = [2022,2021,2020,2019,2018]

# ======================================================
# Read climate data
# ======================================================
print("\nReading files: CLIMATE\n")

climate = pd.read_parquet(
    CLIMATE_PATH,
    columns=["code_muni", "date", "heat_event"],
)

climate["DATA"] = pd.to_datetime(climate["date"])
climate['YEAR'] = pd.to_datetime(climate['DATA']).dt.year
climate["CODMUNICIPIO"] = climate["code_muni"].str[:-1]

climate = climate.drop(columns=["date", "code_muni"])
climate = climate.loc[climate['YEAR'].isin(YEARS)]
# ======================================================
# Read births data (SINASC)
# ======================================================
print("\nReading files: BIRTHS\n")

births = pd.read_parquet(
    BIRTHS_PATH,
    columns=[
        "id_sinasc",
        "DTNASC",
        "CODMUNNASC",
        "IDANOMAL",
        "risk_score",
    ],
)

births["id_sinasc"] = births["id_sinasc"].astype(int)
births["DATA"] = pd.to_datetime(births["DTNASC"], format="%d%m%Y")
births['YEAR'] = pd.to_datetime(births['DATA']).dt.year
births["CODMUNICIPIO"] = births["CODMUNNASC"]

births = births.drop(columns=["DTNASC", "CODMUNNASC"])
births = births.loc[births['YEAR'].isin(YEARS)]

# ======================================================
# Read birth–death matches
# ======================================================
print("\nReading files: MATCH (births × deaths)\n")

matches = pd.read_parquet(MATCH_PATH)

matches["id_sim"] = matches["id_sim"].astype(int)
matches["id_sinasc"] = matches["id_sinasc"].astype(int)

# ======================================================
# Join: births × deaths
# ======================================================
print("\nJoining data\n")

df = births.merge(
    matches,
    how="left",
    on="id_sinasc",
)

df["early_neonatal_death"] = np.where(df["id_sim"].isna(), 0, 1)

# ======================================================
# Join: climate exposure
# ======================================================
df = df.merge(
    climate,
    how="left",
    on=["DATA", "CODMUNICIPIO"],
)

# ======================================================
# Save
# ======================================================
print("\nSave\n")

df.to_parquet(OUTPUT_PATH, index=False)

print("\nFINISHED\n")
