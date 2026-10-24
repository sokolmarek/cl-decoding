"""Forward-backward copula Lasso-Granger causality networks (Sections 4.2 and 4.3, Algorithm 1).

A segment is split into ``n_series`` non-overlapping sub-series. Each sub-series is mapped to the copula space
with its empirical CDF followed by the standard normal quantile function (Eq. 2). For every target sub-series a
Lasso regression on ``lag`` past values of all sub-series is fitted (Eq. 4) for a grid of penalties and the
penalty with the lowest AIC-type score is kept. The causal strength of source ``j`` on target ``i`` is the sum
of the absolute coefficients of the lags of ``j``. The same estimate on the time-reversed sub-series gives the
backward graph, and the result is the mean of the forward graph and the transposed backward graph.

The Lasso follows the glmnet convention: predictors are centred and scaled to unit variance, an intercept is
fitted and the objective is ``1 / (2 m) * ||y - X b||^2 + lambda * ||b||_1``.
"""

import numpy as np
from numba import njit
from scipy.stats import norm

N_SERIES = 32
LAG = 4
LAMBDAS = np.logspace(-3, 2, 6)


def copula_transform(series):
    """Map each row of ``series`` (n_series, T) to standard normal scores (Eq. 2)."""
    T = series.shape[1]
    delta = 1.0 / (4.0 * T ** 0.25 * np.sqrt(np.pi * np.log(T)))
    u = (series[:, None, :] < series[:, :, None]).sum(axis=-1) / T
    return norm.ppf(np.clip(u, delta, 1.0 - delta))


@njit(cache=True)
def lasso_path_cd(X, y, lambdas, max_iter=1000, tol=1e-4):
    """Coordinate descent Lasso path with warm starts.

    Minimizes ``1 / (2 m) * ||y - X b||^2 + lambda * ||b||_1`` for each ``lambda`` in ``lambdas`` (descending).
    ``X`` and ``y`` must be centred. Returns coefficients of shape ``(n_features, n_lambdas)``.
    """
    m, p = X.shape
    G = X.T @ X
    q = X.T @ y
    w = np.zeros(p)
    grad = q.copy()  # X^T (y - X w)
    out = np.zeros((p, lambdas.shape[0]))
    for k in range(lambdas.shape[0]):
        thresh = m * lambdas[k]
        for _ in range(max_iter):
            max_step, max_w = 0.0, 0.0
            for j in range(p):
                if G[j, j] == 0.0:
                    continue
                old = w[j]
                rho = grad[j] + G[j, j] * old
                new = np.sign(rho) * max(abs(rho) - thresh, 0.0) / G[j, j]
                step = new - old
                if step != 0.0:
                    w[j] = new
                    for i in range(p):
                        grad[i] -= G[i, j] * step
                max_step = max(max_step, abs(step))
                max_w = max(max_w, abs(new))
            if max_w == 0.0 or max_step / max_w < tol:
                break
        out[:, k] = w
    return out


def lasso_granger(z, lag=LAG, lambdas=LAMBDAS):
    """Lasso-Granger causal strengths.

    Parameters
    ----------
    z : ndarray, shape (n_series, T)
    lag : int
    lambdas : array_like

    Returns
    -------
    ndarray, shape (n_series, n_series)
        Entry ``[j, i]`` is the strength with which series ``j`` Granger-causes series ``i``.
    """
    K, T = z.shape
    m = T - lag
    if m < 2:
        raise ValueError(f"Sub-series of length {T} are too short for lag {lag}.")

    # Row t holds z[j, t-1], ..., z[j, t-lag] for every series j (grouped by series)
    lags = np.arange(lag, T)[:, None] - np.arange(1, lag + 1)[None, :]
    X = z[:, lags].transpose(1, 0, 2).reshape(m, K * lag)
    Y = z[:, lag:].T

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    y_mean = Y.mean(axis=0)

    alphas = np.sort(np.asarray(lambdas, dtype=float))[::-1]
    Yc = np.ascontiguousarray(Y - y_mean)
    coefs = np.stack([lasso_path_cd(Xs, Yc[:, k], alphas) for k in range(K)])  # (K, K * lag, n_lambdas)
    coefs = coefs / sd[None, :, None]
    intercept = y_mean[:, None] - np.einsum("p,kpl->kl", mu, coefs)

    pred = np.einsum("mp,kpl->kml", X, coefs) + intercept[:, None, :]
    rss = ((pred - Y.T[:, :, None]) ** 2).sum(axis=1)
    aic = rss / m + 2.0 * (coefs != 0).sum(axis=1) / m
    best = np.argmin(aic, axis=1)

    w = coefs[np.arange(K), :, best]  # (target, K * lag)
    strength = np.abs(w).reshape(K, K, lag).sum(axis=-1)  # (target, source)
    return strength.T


def fb_copula_granger(segment, n_series=N_SERIES, lag=LAG, lambdas=LAMBDAS):
    """Causal adjacency matrix ``(n_series, n_series)`` of one segment (Algorithm 1)."""
    segment = np.asarray(segment, dtype=float)
    if segment.size % n_series:
        raise ValueError(f"Segment length {segment.size} is not divisible by n_series={n_series}.")
    z = copula_transform(segment.reshape(n_series, -1))
    forward = lasso_granger(z, lag, lambdas)
    backward = lasso_granger(z[:, ::-1], lag, lambdas)
    return 0.5 * (forward + backward.T)


def gcn(segments, n_series=N_SERIES, lag=LAG, lambdas=LAMBDAS, n_jobs=1):
    """Causal adjacency matrices of a batch of segments, shape ``(n_segments, n_series, n_series)``."""
    segments = np.atleast_2d(segments)
    if n_jobs == 1:
        return np.stack([fb_copula_granger(x, n_series, lag, lambdas) for x in segments])

    from joblib import Parallel, delayed

    out = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(fb_copula_granger)(x, n_series, lag, lambdas) for x in segments
    )
    return np.stack(out)
