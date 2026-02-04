from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import datetime 

# -------------------------------
# Bootstrap
# -------------------------------
def bootstrap(df, 
              estimator, 
              n_jobs = 8, 
              rounds =500, 
              seed = 1944, 
              percentiles = [2.5,97.5], 
              **kwargs
              ):
    """
    Bootstrap an estimator and return the mean estimate and confidence interval.
    """

    np.random.seed(seed)

    if n_jobs == 1:
        stats = []
        for i in range(rounds):
            sample = df.sample(frac=1, replace = True)
            stats.append(estimator(sample, **kwargs))
    else:
        stats = Parallel(n_jobs = n_jobs, backend='loky', verbose=5)(
            delayed(estimator)(
                df.sample(frac=1, replace = True), 
                **kwargs
                )
            for _ in range(rounds)
        )

    return np.mean(stats), np.percentile(stats, percentiles)


# --------------------------------
# results to df
# --------------------------------
def results_to_df(results_dict):
    """
    Convert a dict of results into a tidy DataFrame.

    Expected format:
        {method: value}
        {method: (value, (ci_low, ci_high))}
    """
    rows = []
    for method, val in results_dict.items():
        if isinstance(val, tuple):
            mean, ci = val
            ci_low, ci_high = ci
        else:
            mean = val
            ci_low = ci_high = np.nan

        rows.append({
            'method': method,
            'value': mean,
            'ci_low': ci_low,
            'ci_high': ci_high
        })

    return pd.DataFrame(rows, columns=['method', 'value', 'ci_low', 'ci_high'])

def log_step(name):
    print("\n", datetime.datetime.now())
    print(name)