"""Evaluation metrics for equation recovery, forecasting and uncertainty."""
import numpy as np
from scipy.stats import wasserstein_distance

from .integrators import rollout_np


# ---------------------------------------------------------------------------
# Equation recovery
# ---------------------------------------------------------------------------
def coef_error(Xi_hat, Xi_true):
    """Relative Frobenius error ||Xi_hat - Xi||_F / ||Xi||_F."""
    if Xi_hat is None or not np.isfinite(Xi_hat).all():
        return np.nan
    return float(np.linalg.norm(Xi_hat - Xi_true) / np.linalg.norm(Xi_true))


def support_scores(Xi_hat, Xi_true, tol=1e-8):
    est, tru = np.abs(Xi_hat) > tol, np.abs(Xi_true) > tol
    tp = np.sum(est & tru)
    fp = np.sum(est & ~tru)
    fn = np.sum(~est & tru)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=float(prec), recall=float(rec), f1=float(f1),
                n_terms=int(est.sum()), exact_support=bool(np.array_equal(est, tru)))


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
def nrmse_t(pred, true, scale):
    """Normalised error e(t) = sqrt(mean_i ((xhat_i - x_i) / s_i)^2), shape (T, ...)."""
    return np.sqrt(np.mean(((pred - true) / scale) ** 2, axis=-1))


def valid_prediction_time(pred, true, t, scale, lyap, thresh=0.4):
    """First time e(t) exceeds `thresh`, in Lyapunov times. pred/true: (T, B, d)."""
    e = nrmse_t(pred, true, scale)
    e = np.where(np.isfinite(e), e, np.inf)
    over = e > thresh
    first = np.where(over.any(axis=0), over.argmax(axis=0), len(t) - 1)
    return (t[first] - t[0]) * lyap


def invariant_measure_distance(X_model, X_ref, scale):
    """Mean over coordinates of the 1-D Wasserstein-1 distance between
    marginal densities, in units of the reference standard deviation."""
    if X_model is None or not np.isfinite(X_model).all():
        return np.nan
    return float(np.mean([wasserstein_distance(X_model[:, i] / scale[i], X_ref[:, i] / scale[i])
                          for i in range(X_ref.shape[1])]))


def lyapunov_benettin(f, x0, h, n_steps, renorm_every=10, d0=1e-8, transient=1000, rng=None):
    """Largest Lyapunov exponent by the two-trajectory method of Benettin et al.:
    a perturbation is advected alongside the reference orbit and renormalised to
    length d0 every `renorm_every` steps; the exponent is the time average of
    the logarithmic stretching factors."""
    rng = np.random.default_rng(1) if rng is None else rng
    x = rollout_np(f, x0, transient, h, save_every=transient)[-1]
    if not np.isfinite(x).all():
        return np.nan
    v = rng.standard_normal(x.shape)
    y = x + d0 * v / np.linalg.norm(v)
    acc, n = 0.0, 0
    for _ in range(n_steps // renorm_every):
        x = rollout_np(f, x, renorm_every, h, save_every=renorm_every)[-1]
        y = rollout_np(f, y, renorm_every, h, save_every=renorm_every)[-1]
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            return np.nan
        dist = np.linalg.norm(y - x)
        if dist == 0:
            y = x + d0 * v / np.linalg.norm(v)
            continue
        acc += np.log(dist / d0)
        n += 1
        y = x + (y - x) * (d0 / dist)
    return acc / (n * renorm_every * h) if n else np.nan


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def ensemble_crps(samples, y):
    """Sample CRPS, E|X - y| - 0.5 E|X - X'|, averaged over all other axes.
    samples: (M, ...), y: (...)."""
    m = samples.shape[0]
    t1 = np.nanmean(np.abs(samples - y[None]), axis=0)
    diffs = np.abs(samples[:, None] - samples[None, :])
    t2 = np.nansum(diffs, axis=(0, 1)) / (2.0 * m * m)
    return t1 - t2


def interval_coverage(samples, y, lo=5, hi=95):
    ql, qh = np.nanpercentile(samples, [lo, hi], axis=0)
    return (y >= ql) & (y <= qh)
