"""Market model and abnormal returns.

Market model (estimated per event, on a window BEFORE the event):

    r_i,t = alpha_i + beta_i * r_m,t + e_i,t

Abnormal return = actual return minus what the market model predicts:

    AR_i,t = r_i,t - (alpha_i + beta_i * r_m,t)

The estimation window ends well before the event (default days -250 to -30)
so the event itself cannot contaminate the "normal" relationship.
"""

from __future__ import annotations

import numpy as np


def market_model(r: np.ndarray, m: np.ndarray, min_obs: int = 100) -> tuple[float, float] | None:
    """OLS alpha, beta of stock returns `r` on market returns `m`.

    Returns None if fewer than `min_obs` paired, finite observations.
    """
    ok = np.isfinite(r) & np.isfinite(m)
    if ok.sum() < min_obs:
        return None
    r, m = r[ok], m[ok]
    var_m = m.var()
    if var_m == 0:
        return None
    beta = ((m - m.mean()) * (r - r.mean())).mean() / var_m
    alpha = r.mean() - beta * m.mean()
    return float(alpha), float(beta)


def abnormal_returns(r: np.ndarray, m: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """AR = r - (alpha + beta * m), elementwise."""
    return r - (alpha + beta * m)
