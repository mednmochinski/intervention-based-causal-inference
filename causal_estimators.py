import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from patsy import dmatrix


# ------------------------------------
# Naive estimator
# ------------------------------------
def naive_estimator(df):
    """
    Unadjusted ATE: difference in mean outcomes
    between treated and control groups.
    """

    mean_Y_treated = df.loc[df["D"] == 1, "Y"].mean()
    mean_Y_control = df.loc[df["D"] == 0, "Y"].mean()

    return mean_Y_treated - mean_Y_control


# ------------------------------------
# Adjustment formula estimator
# ------------------------------------
def adjustment_formula_estimator(df, adjustment_set):
    """
    ATE via the adjustment formula over a given set of confounders.
    """

    group = df.groupby(adjustment_set)

    # P(Z = z)
    p = group.size() / len(df)

    # E[Y | D = 1, Z = z] and E[Y | D = 0, Z = z]
    mu1 = group.apply(lambda g: g.loc[g["D"] == 1, "Y"].mean())
    mu0 = group.apply(lambda g: g.loc[g["D"] == 0, "Y"].mean())

    # Fallback for empty strata
    mu1 = mu1.fillna(df.loc[df["D"] == 1, "Y"].mean())
    mu0 = mu0.fillna(df.loc[df["D"] == 0, "Y"].mean())

    return ((mu1 - mu0) * p).sum()


# ------------------------------------
# Linear regression: causal estimate
# ------------------------------------
def linreg_causal_estimator(df, model_exp, outcome_var="Y"):
    """
    Linear regression coefficient on treatment indicator.
    """

    X = dmatrix(model_exp, df)
    model = LinearRegression().fit(X, df[outcome_var])

    return model.coef_[1]


# ------------------------------------
# Linear regression: outcome model
# ------------------------------------
def linreg_potentialoutcome_estimator(
    df,
    model_exp="W+Z",
    treatment_var="D",
    outcome_var="Y",
):
    """
    ATE from separate outcome models for treated and control units.
    """

    df_control = df.loc[df[treatment_var] == 0]
    control_model = LinearRegression().fit(
        dmatrix(model_exp, df_control),
        df_control[outcome_var],
    )

    df_treated = df.loc[df[treatment_var] == 1]
    treated_model = LinearRegression().fit(
        dmatrix(model_exp, df_treated),
        df_treated[outcome_var],
    )

    ate = np.mean(
        df[treatment_var]
        * (df[outcome_var] - control_model.predict(dmatrix(model_exp, df)))
        + (1 - df[treatment_var])
        * (treated_model.predict(dmatrix(model_exp, df)) - df[outcome_var])
    )

    return ate


# ------------------------------------
# IPW estimator
# ------------------------------------
def ipw_estimator(df, model_exp="Z", treatment_var="D", outcome_var="Y"):
    """
    Inverse Probability Weighting (IPW) estimator.
    """

    propensity_score = (
        LogisticRegression()
        .fit(dmatrix(model_exp, df), df[treatment_var])
        .predict_proba(dmatrix(model_exp, df))[:, 1]
    )

    return np.mean(
        df[outcome_var]
        * (df[treatment_var] - propensity_score)
        / (propensity_score * (1 - propensity_score))
    )


# ------------------------------------
# IPW stabilized estimator
# ------------------------------------
def ipw_stabilized_estimator(df, model_exp="Z", treatment_var="D", outcome_var="Y"):
    """
    Stabilized IPW estimator.
    """

    prob_d = df[treatment_var].mean()

    ps_model = LogisticRegression().fit(
        dmatrix(model_exp, df),
        df[treatment_var],
    )

    df_control = df.loc[df[treatment_var] == 0]
    df_treated = df.loc[df[treatment_var] == 1]

    ps_control = ps_model.predict_proba(dmatrix(model_exp, df_control))[:, 1]
    ps_treated = ps_model.predict_proba(dmatrix(model_exp, df_treated))[:, 1]

    weight_control = (1 - prob_d) / (1 - ps_control)
    weight_treated = prob_d / ps_treated

    y1 = np.sum(df_treated[outcome_var] * weight_treated) / len(df_treated)
    y0 = np.sum(df_control[outcome_var] * weight_control) / len(df_control)

    return y1 - y0


# ------------------------------------
# Propensity score linear regression
# ------------------------------------
def ps_linreg_estimator(df, model_exp="Z", treatment_var="D", outcome_var="Y"):
    """
    Linear regression adjusted by the estimated propensity score.
    """

    propensity_score = (
        LogisticRegression()
        .fit(dmatrix(model_exp, df), df[treatment_var])
        .predict_proba(dmatrix(model_exp, df))[:, 1]
    )

    df_model = df.assign(propensity_score=propensity_score)

    X = dmatrix(f"{treatment_var} + propensity_score", df_model)
    model = LinearRegression().fit(X, df_model[outcome_var])

    return model.coef_[1]


# ------------------------------------
# Propensity score matching
# ------------------------------------
def ps_matching_estimator(
    df,
    model_exp="Z",
    treatment_var="D",
    outcome_var="Y",
    n_jobs_knn=1,
):
    """
    Nearest-neighbor matching on the propensity score.
    """

    propensity_score = (
        LogisticRegression()
        .fit(dmatrix(model_exp, df), df[treatment_var])
        .predict_proba(dmatrix(model_exp, df))[:, 1]
    )

    df_ps = df.assign(propensity_score=propensity_score)

    treated = df_ps[df_ps[treatment_var] == 1].reset_index(drop=True)
    control = df_ps[df_ps[treatment_var] == 0].reset_index(drop=True)

    knn_treated = KNeighborsRegressor(
        n_neighbors=1,
        n_jobs=n_jobs_knn,
    ).fit(treated[["propensity_score"]], treated[[outcome_var]])

    knn_control = KNeighborsRegressor(
        n_neighbors=1,
        n_jobs=n_jobs_knn,
    ).fit(control[["propensity_score"]], control[[outcome_var]])

    matches = pd.concat(
        [
            treated.assign(
                matched_outcome=knn_control.predict(treated[["propensity_score"]])
            ),
            control.assign(
                matched_outcome=knn_treated.predict(control[["propensity_score"]])
            ),
        ]
    )

    ate = np.mean(
        matches[treatment_var]
        * (matches[outcome_var] - matches["matched_outcome"])
        + (1 - matches[treatment_var])
        * (matches["matched_outcome"] - matches[outcome_var])
    )

    return ate


# ------------------------------------
# Double robust estimator
# ------------------------------------
def double_robust_estimator(
    df,
    linreg_model_exp="W",
    ps_model_exp="Z",
    treatment_var="D",
    outcome_var="Y",
):
    """
    Doubly robust ATE estimator.
    """

    df_control = df.loc[df[treatment_var] == 0]
    control_model = LinearRegression().fit(
        dmatrix(linreg_model_exp, df_control),
        df_control[outcome_var],
    )

    df_treated = df.loc[df[treatment_var] == 1]
    treated_model = LinearRegression().fit(
        dmatrix(linreg_model_exp, df_treated),
        df_treated[outcome_var],
    )

    propensity_score = (
        LogisticRegression()
        .fit(dmatrix(ps_model_exp, df), df[treatment_var])
        .predict_proba(dmatrix(ps_model_exp, df))[:, 1]
    )

    treated_mean = np.mean(
        treated_model.predict(dmatrix(linreg_model_exp, df))
        + (df[outcome_var] - treated_model.predict(dmatrix(linreg_model_exp, df)))
        * df[treatment_var]
        / propensity_score
    )

    untreated_mean = np.mean(
        control_model.predict(dmatrix(linreg_model_exp, df))
        + (df[outcome_var] - control_model.predict(dmatrix(linreg_model_exp, df)))
        * (1 - df[treatment_var])
        / (1 - propensity_score)
    )

    return treated_mean - untreated_mean
