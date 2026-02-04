import pandas as pd
import numpy as np
import scipy.stats as stats

def generate_data_discrete(n = 1000, true_ATE = 2.0):
    """
    Generate synthetic data with a discrete confounder Z.
    n: int, number of samples
    true_ATE: float, true causal effect of D on Y
    """
    # Discrete confounder Z ∈ {0, 1, 2}
    Z = stats.randint.rvs(low = 0, high = 3, size = n)

    # W depends on Z (categorical 3,4).
    transition = {
        0: [0.7, 0.3],
        1: [0.4, 0.6],
        2: [0.1, 0.9],
    }
    W = np.array([np.random.choice([3,4], p=transition[z]) for z in Z])

    # Treatment assignment depends on Z
    alpha_z = np.array([0.25, 0.5, 0.7]) 
    p_treat = alpha_z[Z]
    D = np.random.binomial(1, p_treat, size=n)

    # outcome depends on D and W plus noise
    noise = np.random.normal(0,1,size = n)
    coef_W = 3
    Y = 1.5 + true_ATE*D + coef_W*W + noise

    df = pd.DataFrame({
                    "Z": Z, 
                    "W": W, 
                    "D": D, 
                    "Y": Y, 
                    "p_treat": p_treat
                    })

    return df, true_ATE


def generate_data_continuous(n = 1000, true_ATE = 2.0):
    """
    Generate synthetic data with a continuous confounder Z.
    n: int, number of samples
    true_ATE: float, true causal effect of D on Y
    """
    # Continuous confounder Z
    Z = np.random.normal(0, 0.5, size=n)

    # W depends on Z 
    p_w = 1 / (1 + np.exp(-Z))   
    W = np.random.binomial(1, p_w, size=n)+3

    # Treatment assignment depends on Z
    logits = -2.5 + 2.4*Z 
    p_treat = 1 / (1 + np.exp(-logits))
    D = np.random.binomial(1, p_treat, size=n)

    # Outcome depends on treatment, W, and noise
    noise = np.random.normal(0,1,size = n)
    coef_W = 3
    Y = 1.5 + true_ATE*D + coef_W*W + noise

    df = pd.DataFrame({
                    "Z": Z, 
                    "W": W, 
                    "D": D, 
                    "Y": Y, 
                    "p_treat": p_treat
                    })

    return df, true_ATE

