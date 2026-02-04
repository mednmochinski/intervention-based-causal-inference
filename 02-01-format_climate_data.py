import os
import pandas as pd
from joblib import Parallel, delayed

print('\nSTART\n')

# =============================================================
# Definitions
# =============================================================

DATE_COL = 'date'
CITY_COL = 'code_muni'
TEMPERATURE_COL = 'TMAX_max'

# Heat-event parameters
WINDOW_SIZE = 90          # last trimester
TEMPERATURE_THRESHOLD = 30
MIN_DAYS = 30

# Parallelization
N_JOBS = 12


# =============================================================
# Read and combine yearly climate files
# =============================================================

print('\nReading files\n')

input_folder = (
    'observational_data/'
    'raw_CLIMATERNA_data/climate/'
)

list_dfs = []

for filename in os.listdir(input_folder):
    if filename.endswith('.parquet'):
        print(filename)
        df_year = pd.read_parquet(os.path.join(input_folder, filename))
        list_dfs.append(df_year)

df = pd.concat(list_dfs, ignore_index=True)

# Save combined raw dataset
df.to_parquet(
    'observational_data/processed_data/'
    'BR-DWGD_2010-2024.parquet',
    index=False
)


# =============================================================
# Heat-event flagging
# =============================================================

print('\nFlag heat event\n')


def flag_heat_event(
    group: pd.DataFrame,
    window_size: int,
    temperature_threshold: float,
    min_days: int,
    date_col: str,
    city_col: str,
    temperature_col: str,
) -> pd.DataFrame:
    """
    Flag heat events based on the number of hot days
    in a rolling window.
    """
    group = group.sort_values(date_col).copy()

    # Daily heat indicator
    group['heat'] = group[temperature_col] > temperature_threshold

    # Count hot days in the previous window (excluding today)
    group['n_hot_days_last_X'] = (
        group['heat']
        .shift(1)
        .rolling(window=window_size, min_periods=1)
        .sum()
    )

    # Heat-event flag
    group['heat_event'] = group['n_hot_days_last_X'] >= min_days

    return group[
        [city_col, date_col, temperature_col, 'heat_event']
    ]


# =============================================================
# Parallel processing by municipality
# =============================================================

results = Parallel(
    n_jobs=N_JOBS,
    backend='loky',
    verbose=5
)(
    delayed(flag_heat_event)(
        group,
        WINDOW_SIZE,
        TEMPERATURE_THRESHOLD,
        MIN_DAYS,
        DATE_COL,
        CITY_COL,
        TEMPERATURE_COL,
    )
    for _, group in df.groupby(CITY_COL, sort=False)
)

df_final = pd.concat(results, ignore_index=True)


# =============================================================
# Save processed dataset
# =============================================================

print('\nsave\n')

df_final.to_parquet(
    'observational_data/processed_data/'
    'climate_processed_2010-2024.parquet',
    index=False
)

print('\nFINISHED\n')
