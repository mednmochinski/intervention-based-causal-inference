import pandas as pd
import numpy as np
import os

print('\nSTART\n')

# =============================================================
# Read data
# =============================================================

print('\nReading files\n')

file_path = 'observational_data/raw_CLIMATERNA_data/health/sinasc_2010_2022.parquet'
df_sinasc = pd.read_parquet(file_path, engine='fastparquet')

print('n rows: (total births)', df_sinasc.shape[0])


# =============================================================
# Column definitions
# =============================================================

# Columns used for joins
colunas_join_sim = [
    'DTNASC',
    'CODMUNNATU',  # -> CODMUNNASC
    'ESCMAE',
    'IDADEMAE',
    'RACACOR',
    'SEXO',
]

dic_join_nasc = {
    'CODMUNNATU': 'CODMUNNASC'
}

colunas_join_sinasc = [
    dic_join_nasc.get(col, col) for col in colunas_join_sim
]

# Confounders
colunas_socioecon = [
    'ESCMAE',
    'IDADEMAE',
    'RACACOR',
    'RACACORMAE',
]

colunas_comorbidade = [
    'IDANOMAL'
]


# =============================================================
# Filter and clean data
# =============================================================

selected_cols = list(
    set(colunas_join_sinasc + colunas_socioecon + colunas_comorbidade)
)

df2 = df_sinasc[selected_cols].copy()

# Fill missing categorical values
df2['ESCMAE'] = df2['ESCMAE'].fillna('9')
df2['RACACOR'] = df2['RACACOR'].fillna('9')
df2['RACACORMAE'] = df2['RACACORMAE'].fillna('9')
df2['IDANOMAL'] = df2['IDANOMAL'].fillna('9')

# Fill and cast maternal age
df2['IDADEMAE'] = df2['IDADEMAE'].fillna('200').astype(int)


# =============================================================
# Risk score construction
# =============================================================

def build_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---------------------------------------------------------
    # ESCMAE — Maternal schooling
    # Lower schooling → higher risk
    # ---------------------------------------------------------
    esc_map = {
        1: 1.0,   # none
        2: 0.8,
        3: 0.6,
        4: 0.3,
        5: 0.1,   # highest schooling
        9: 1.2    # ignored → penalize
    }

    esc_risk = df['ESCMAE'].map(esc_map).fillna(1.0)

    # ---------------------------------------------------------
    # IDADEMAE — Maternal age
    # Safe zone: 18–35
    # Ignored value: 200
    # ---------------------------------------------------------
    idade = df['IDADEMAE']

    # initialize with NaN
    age_risk = np.full(len(df), np.nan)

    # ignored ages
    age_risk[idade == 200] = 1.5

    # distance from safe zone
    dist = np.where(
        (idade >= 18) & (idade <= 35),
        0,
        np.minimum(np.abs(idade - 18), np.abs(idade - 35))
    )

    # quadratic penalty
    computed_age_risk = 0.1 + 0.02 * (dist ** 2)

    # fill where not ignored
    age_risk = np.where(np.isnan(age_risk), computed_age_risk, age_risk)

    # ---------------------------------------------------------
    # RACACOR — Infant race/color
    # ---------------------------------------------------------
    race_risk_map = {
        1: 0.0,   # white
        2: 0.3,   # black
        3: 0.2,   # yellow
        4: 0.3,   # parda
        5: 0.4    # indigenous
    }

    race_inf_risk = df['RACACOR'].map(race_risk_map).fillna(0.3)

    # ---------------------------------------------------------
    # RACACORMAE — Mother's race/color
    # ---------------------------------------------------------
    race_mom_risk = df['RACACORMAE'].map(race_risk_map).fillna(0.3)

    # ---------------------------------------------------------
    # Combine components (weighted sum)
    # ---------------------------------------------------------
    raw_score = (
        0.40 * esc_risk +
        0.20 * age_risk +
        0.20 * race_inf_risk +
        0.30 * race_mom_risk
    )

    # ---------------------------------------------------------
    # Normalize to [0, 1000]
    # ---------------------------------------------------------
    min_raw = raw_score.min()
    max_raw = raw_score.max()

    if max_raw == min_raw:
        df['risk_score'] = 500
    else:
        df['risk_score'] = 1000 * (raw_score - min_raw) / (max_raw - min_raw)

    return df


df2 = build_risk_score(df2)


# =============================================================
# Create ID
# =============================================================

df2 = df2.reset_index(drop=True)
df2 = df2.reset_index(names='id_sinasc')


# =============================================================
# Save
# =============================================================

print('\nsave\n')

output_path = (
    'observational_data/processed_data/'
    'births_processed_2010-2022.parquet'
)

df2.to_parquet(output_path, index=False)

print('\nFINISHED\n')
