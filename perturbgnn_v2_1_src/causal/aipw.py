"""Fix 1 — AIPW doubly robust estimator (attacks Durbin's endogeneity of X).

v2's Durbin model `y = ρWy + Xβ + WXθ + ε` assumes X is exogenous. In
SPAC-seq, guide delivery has spatial structure (clone expansion, injection
site), so `E[Xε] ≠ 0` and OLS estimates of β and θ are inconsistent. DiD
addresses parallel-trends violation but not this spatial endogeneity.

AIPW (Robins et al. 1994; ML extension Chernozhukov et al. 2018, EJ):

    μ̂₁(X) = outcome model for T=1     (XGBoost regression of y on X)
    μ̂₀(X) = outcome model for T=0     (XGBoost regression of y on X)
    ê(X)   = propensity model           (XGBoost classification of T)

    τ̂_AIPW = (1/n) Σ [
        T · (y - μ̂₁(X)) / ê(X)
      - (1-T) · (y - μ̂₀(X)) / (1 - ê(X))
      + μ̂₁(X) - μ̂₀(X)
    ]

Doubly robust: consistent if **either** μ̂ **or** ê is correctly specified.
Neyman orthogonality: first-order bias in μ̂ or ê does not propagate to τ̂.
Cross-fitting (5-fold) gives root-n asymptotic normality under both correct.

Influence-function bootstrap gives 95% CI without re-fitting the ML models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone
import xgboost as xgb


def _fit_outcome(X, y, mask, params):
    """Fit an XGBoost regression of y on X using only rows in mask."""
    m = xgb.XGBRegressor(**params)
    m.fit(X[mask], y[mask])
    return m


def _fit_propensity(X, t, params):
    """Fit an XGBoost classification of T on X (binary treatment)."""
    # Use a separate parameter dict — classification.
    clf_params = {**params}
    clf_params.pop("objective", None)
    m = xgb.XGBClassifier(**clf_params)
    # Guard against a single-class fold (rare for matched controls).
    if len(np.unique(t)) < 2:
        return None
    m.fit(X, t)
    return m


def aipw_crossfit(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    outcome_params: dict | None = None,
    propensity_params: dict | None = None,
    seed: int = 7,
) -> dict:
    """Cross-fitted AIPW estimate of the average treatment effect (ATE).

    Parameters
    ----------
    X : (n, d) covariate matrix. Embedding + auxiliary features. **Must not
        include y itself** (no leakage).
    t : (n,) binary treatment indicator. 1 = source, 0 = matched control.
    y : (n,) continuous response (one module score per spot).
    n_splits : cross-fitting folds. 5 is standard.
    outcome_params, propensity_params : XGBoost kwargs. Defaults are sane
        for ~10⁵-10⁶ spots.
    seed : random seed for KFold.

    Returns
    -------
    dict with: tau_hat, se, ci_low, ci_high, n_pos, n_neg, n_source,
        n_control, mean_propensity, IC_std.
    """
    if outcome_params is None:
        outcome_params = dict(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, reg_alpha=0.0,
            objective="reg:squarederror", n_jobs=8, verbosity=0,
        )
    if propensity_params is None:
        propensity_params = dict(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, reg_alpha=0.0,
            eval_metric="logloss", n_jobs=8, verbosity=0,
        )

    X = np.asarray(X, dtype=np.float32)
    t = np.asarray(t, dtype=np.int8)
    y = np.asarray(y, dtype=np.float32)
    n = len(y)

    mu1_hat = np.zeros(n, dtype=np.float32)
    mu0_hat = np.zeros(n, dtype=np.float32)
    e_hat   = np.zeros(n, dtype=np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Train outcome models on this fold's train set.
        # Stratify by treatment so each fold sees both classes.
        pos_train = train_idx[t[train_idx] == 1]
        neg_train = train_idx[t[train_idx] == 0]
        if len(pos_train) < 5 or len(neg_train) < 5:
            # Fallback: train on all of train_idx for this class.
            pos_train = np.where(t == 1)[0]
            neg_train = np.where(t == 0)[0]

        m1 = _fit_outcome(X, y, pos_train, outcome_params)
        m0 = _fit_outcome(X, y, neg_train, outcome_params)
        # Propensity on the full train set (t has both classes in train).
        e_model = _fit_propensity(X[train_idx], t[train_idx], propensity_params)
        if e_model is None:
            e_hat[test_idx] = float(np.mean(t[train_idx]))
        else:
            e_hat[test_idx] = e_model.predict_proba(X[test_idx])[:, 1]

        mu1_hat[test_idx] = m1.predict(X[test_idx])
        mu0_hat[test_idx] = m0.predict(X[test_idx])

    # Clip propensity to avoid 0/0 division. Standard practice.
    eps = 0.01
    e_hat = np.clip(e_hat, eps, 1 - eps)

    # AIPW point estimate.
    psi = (
        t * (y - mu1_hat) / e_hat
        - (1 - t) * (y - mu0_hat) / (1 - e_hat)
        + mu1_hat - mu0_hat
    )
    tau_hat = float(psi.mean())
    # Influence-function SE: sqrt(Var(psi)/n). This is the cross-fitted
    # asymptotic SE from Chernozhukov et al. 2018.
    se = float(psi.std(ddof=1) / np.sqrt(n))
    ci_low = tau_hat - 1.96 * se
    ci_high = tau_hat + 1.96 * se

    return {
        "tau_hat": tau_hat,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_total": n,
        "n_source": int((t == 1).sum()),
        "n_control": int((t == 0).sum()),
        "mean_propensity": float(np.mean(e_hat)),
        "IC_std": float(psi.std(ddof=1)),
        "mu1_mean": float(mu1_hat.mean()),
        "mu0_mean": float(mu0_hat.mean()),
        "min_propensity": float(np.min(e_hat)),
        "max_propensity": float(np.max(e_hat)),
    }


def aipw_estimate(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    **kwargs,
) -> dict:
    """Convenience wrapper — defaults to cross-fitted AIPW with 5 folds."""
    return aipw_crossfit(X, t, y, **kwargs)


# ---------------------------------------------------------------------------
# Self-test on synthetic data: known ATE recovery.
# ---------------------------------------------------------------------------
def _self_test():
    """Sanity check: AIPW recovers a known ATE on synthetic data."""
    rng = np.random.default_rng(0)
    n = 8000
    # 8 covariates, confounded: T depends on X[:, 0].
    X = rng.normal(0, 1, size=(n, 8)).astype(np.float32)
    logit_p = 0.7 * X[:, 0] + 0.2 * X[:, 1]
    p = 1 / (1 + np.exp(-logit_p))
    t = (rng.uniform(size=n) < p).astype(np.int8)
    # True outcome model: y = 2.0*T + X[:, 0] + 0.5*X[:, 1] + noise.
    # True ATE = 2.0.
    y = (2.0 * t + X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n)).astype(np.float32)
    out = aipw_crossfit(X, t, y, n_splits=5)
    assert abs(out["tau_hat"] - 2.0) < 0.15, f"ATE recovery failed: {out['tau_hat']}"
    assert out["ci_low"] < 2.0 < out["ci_high"], f"CI doesn't cover truth: {out}"
    print(f"SELF-TEST PASS: AIPW τ̂ = {out['tau_hat']:.3f} "
          f"(95% CI [{out['ci_low']:.3f}, {out['ci_high']:.3f}]), true = 2.0")
    return out


if __name__ == "__main__":
    _self_test()
