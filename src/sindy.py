"""Classical sparse system identification.

C1  dense least squares on finite-difference derivatives,
C2  SINDy with sequentially thresholded least squares (STLSQ) on finite-difference
    or Savitzky-Golay derivatives,
C3  weak-form SINDy, where derivatives are moved onto smooth compactly supported
    test functions by integration by parts,
plus bootstrap aggregation (E-SINDy) for coefficient uncertainty.

All regressions run in nondimensional coordinates z = x/s, tau = t/t_char so
that one relative threshold grid serves every system. Coefficients are mapped
back to physical units at the end (see library.PolyLibrary).
"""
import numpy as np
from scipy.signal import savgol_filter

from .integrators import rollout_np


# ---------------------------------------------------------------------------
# Derivative estimates
# ---------------------------------------------------------------------------
def fd_derivative(Y, t):
    """Second-order central differences (one-sided at the ends).
    Truncation error O(dt^2 |x'''|); noise amplification ~ sigma / dt."""
    return np.gradient(Y, t, axis=0, edge_order=2)


def sg_derivative(Y, dt, window, order=3):
    """Savitzky-Golay derivative: local polynomial least squares, which trades
    the O(sigma/dt) noise gain of differencing for a bias that grows with the
    window length relative to the local time scale of the signal."""
    window = int(window) | 1
    window = max(window, order + 2 if (order + 2) % 2 else order + 3)
    return savgol_filter(Y, window, order, deriv=1, delta=dt, axis=0, mode="interp")


# ---------------------------------------------------------------------------
# Sparse regression
# ---------------------------------------------------------------------------
def ridge_lstsq(A, B, alpha=0.0):
    if alpha == 0.0:
        return np.linalg.lstsq(A, B, rcond=None)[0]
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + alpha * np.eye(n), A.T @ B)


def stlsq(Theta, dX, lam, alpha=1e-10, max_iter=25):
    """Sequentially thresholded least squares (Brunton et al., 2016).

    Each sweep solves a least-squares problem on the current support and then
    removes coefficients with |xi| < lam. The objective ||dX - Theta Xi||^2 +
    lam^2 ||Xi||_0 is nonconvex and combinatorial; STLSQ is a greedy descent
    that converges in finitely many sweeps because the support can only shrink.
    A tiny ridge term keeps the normal equations well posed when columns of
    Theta are nearly collinear (e.g. x and x^2 on a narrow attractor)."""
    p, d = Theta.shape[1], dX.shape[1]
    Xi = ridge_lstsq(Theta, dX, alpha)
    for _ in range(max_iter):
        small = np.abs(Xi) < lam
        Xi_old = Xi.copy()
        Xi[small] = 0.0
        for j in range(d):
            big = ~small[:, j]
            if big.any():
                Xi[big, j] = ridge_lstsq(Theta[:, big], dX[:, j], alpha)
        if np.array_equal(np.abs(Xi_old) < lam, np.abs(Xi) < lam) and np.allclose(Xi, Xi_old):
            break
    Xi[np.abs(Xi) < lam] = 0.0
    return Xi


# ---------------------------------------------------------------------------
# Weak formulation
# ---------------------------------------------------------------------------
def _bump(s, p):
    """phi(s) = (1 - s^2)^p on [-1, 1]; C^{p-1} at the boundary."""
    return np.where(np.abs(s) < 1.0, (1.0 - s ** 2) ** p, 0.0)


def _dbump(s, p):
    return np.where(np.abs(s) < 1.0, -2.0 * p * s * (1.0 - s ** 2) ** (p - 1), 0.0)


def weak_system(Y, t, lib, half_width, p=4, stride=None):
    """Assemble G Xi = b with
        G_kj = int phi_k(t) theta_j(y(t)) dt,   b_ki = -int phi_k'(t) y_i(t) dt.

    Integration by parts removes the derivative from the data entirely; the
    boundary term vanishes because phi_k is compactly supported. With the
    trapezoidal rule and phi in C^{p-1}, the Euler-Maclaurin correction terms
    vanish up to order p-1, so the quadrature error decays like dt^p for a
    resolved integrand, instead of the dt^2 of a central difference."""
    t = np.asarray(t)
    if stride is None:
        stride = max(1, int(np.searchsorted(t - t[0], half_width) // 2))
    centres = np.arange(0, len(t), stride)
    centres = centres[(t[centres] - half_width >= t[0]) & (t[centres] + half_width <= t[-1])]
    Th = lib(Y)
    G, B = [], []
    for c in centres:
        s = (t - t[c]) / half_width
        m = np.abs(s) <= 1.0
        tt, ss = t[m], s[m]
        if len(tt) < 3:
            continue
        phi, dphi = _bump(ss, p), _dbump(ss, p) / half_width
        G.append(np.trapezoid(phi[:, None] * Th[m], tt, axis=0))
        B.append(-np.trapezoid(dphi[:, None] * Y[m], tt, axis=0))
    return np.array(G), np.array(B)


# ---------------------------------------------------------------------------
# Model evaluation and selection
# ---------------------------------------------------------------------------
def poly_field(lib, Xi):
    return lambda x: lib(x) @ Xi


def short_horizon_error(f, segs_val, h, horizon, n_starts=24, scale=None):
    """Mean squared normalised error of short rollouts started from noisy
    observations in the held-out blocks (one block per experiment). Measured
    against noisy data, so the noise floor is common to every candidate and
    does not bias the ranking. Observation times need not be regular: the
    rollout is evaluated at the RK4 step nearest to each observation."""
    if scale is None:
        scale = np.concatenate([Y for _, Y in segs_val]).std(axis=0)
    per_seg = max(2, n_starts // len(segs_val))
    errs = []
    for t_val, Y_val in segs_val:
        t_val = np.asarray(t_val)
        hz = min(horizon, 0.5 * (t_val[-1] - t_val[0]))
        last_start = np.searchsorted(t_val, t_val[-1] - hz, side="right") - 1
        starts = np.unique(np.linspace(0, max(last_start, 0), per_seg).astype(int))
        n_steps = int(np.ceil(hz / h - 1e-9))
        traj = rollout_np(f, Y_val[starts], n_steps, h)             # (n_steps+1, S, d)
        for j, s in enumerate(starts):
            sel = np.where((t_val > t_val[s]) & (t_val <= t_val[s] + hz + 1e-12))[0]
            k = np.clip(np.round((t_val[sel] - t_val[s]) / h).astype(int), 0, n_steps)
            errs.append((((traj[k, j] - Y_val[sel]) / scale) ** 2).ravel())
    err = np.concatenate(errs) if errs else np.array([])
    if err.size == 0 or not np.isfinite(err).all():
        return np.inf
    return float(err.mean())


def select_by_validation(candidates, errors, rel_tol=0.05, abs_tol=1e-4):
    """Among candidates, return the index of the sparsest one whose validation
    error is within best * (1 + rel_tol) + abs_tol (ties broken towards lower
    error). The absolute term (a normalised RMS error of 1%) prevents
    differences at the level of integrator round-off from deciding the support
    on noise-free data."""
    errors = np.asarray(errors, float)
    finite = np.isfinite(errors)
    if not finite.any():
        return 0
    best = errors[finite].min()
    ok = np.where(finite & (errors <= best * (1.0 + rel_tol) + abs_tol))[0]
    nnz = np.array([np.count_nonzero(c) for c in candidates])
    order = np.lexsort((errors[ok], nnz[ok]))
    return int(ok[order[0]])


def lambda_grid(Xi_ref, n=24, lo=-4.0, hi=-0.3):
    return np.max(np.abs(Xi_ref)) * np.logspace(lo, hi, n)


# ---------------------------------------------------------------------------
# High-level fitting routines (all return physical-unit coefficients)
# ---------------------------------------------------------------------------
class Nondim:
    """z = x / s, tau = t / T. Holds the scales and the coefficient maps."""
    def __init__(self, lib, s, T):
        self.lib, self.s, self.T = lib, np.asarray(s, float), float(T)

    def z(self, X):
        return X / self.s

    def tau(self, t):
        return t / self.T

    def to_phys(self, Xi_z):
        return self.lib.from_scaled(Xi_z, self.s) / self.T

    def to_z(self, Xi):
        return self.lib.to_scaled(Xi, self.s) * self.T


def _stack_regression(segs, nd, deriv):
    """Rows of (Theta(z), dz/dtau) from every experiment; derivatives are
    always estimated within a segment, never across segment boundaries."""
    Th, dZ = [], []
    for t, Y in segs:
        Z, tau = nd.z(Y), nd.tau(t)
        Th.append(nd.lib(Z))
        dZ.append(deriv(Z, tau))
    return np.concatenate(Th), np.concatenate(dZ)


def fit_dense_fd(segs_fit, nd):
    """C1: ordinary least squares on central differences."""
    Theta, dZ = _stack_regression(segs_fit, nd, lambda Z, tau: fd_derivative(Z, tau))
    return nd.to_phys(ridge_lstsq(Theta, dZ, 1e-10)), {}


def _fit_select(Theta, dZ, nd, segs_val, h, horizon, lam_grid=None):
    Xi0 = ridge_lstsq(Theta, dZ, 1e-10)
    lams = lambda_grid(Xi0) if lam_grid is None else lam_grid
    cands, errs = [], []
    for lam in lams:
        Xi = nd.to_phys(stlsq(Theta, dZ, lam))
        cands.append(Xi)
        errs.append(short_horizon_error(poly_field(nd.lib, Xi), segs_val, h, horizon))
    k = select_by_validation(cands, errs)
    return cands[k], lams[k], errs[k]


def fit_sindy(segs_fit, nd, segs_val, h, horizon, deriv="sg", windows=(5, 7, 9, 13, 17, 25)):
    """C2: STLSQ with FD or Savitzky-Golay derivatives; threshold (and SG window)
    chosen on the held-out blocks."""
    n_min = min(len(t) for t, _ in segs_fit)
    options = [None] if deriv == "fd" else [w for w in windows if w < n_min // 2]
    best = None
    for w in options:
        if w is None:
            fn = lambda Z, tau: fd_derivative(Z, tau)
        else:
            fn = lambda Z, tau, w=w: sg_derivative(Z, np.median(np.diff(tau)), w)
        Theta, dZ = _stack_regression(segs_fit, nd, fn)
        Xi, lam, err = _fit_select(Theta, dZ, nd, segs_val, h, horizon)
        if best is None or err < best[2] * (1 - 1e-9):
            best = (Xi, dict(lam=lam, window=w), err)
    return best[0], dict(best[1], val_err=best[2])


def weak_system_multi(segs, nd, half_width, p=4):
    G, B = [], []
    for t, Y in segs:
        g, b = weak_system(nd.z(Y), nd.tau(t), nd.lib, half_width, p=p)
        if len(g):
            G.append(g)
            B.append(b)
    if not G:
        return np.zeros((0, nd.lib.n_terms)), np.zeros((0, nd.lib.d))
    return np.concatenate(G), np.concatenate(B)


def fit_weak_sindy(segs_fit, nd, segs_val, h, horizon, half_widths=(0.1, 0.2, 0.35, 0.5, 0.75), p=4,
                   return_system=False):
    """C3: weak-form SINDy. half_widths are in units of t_char; a width is
    admissible only if the test function covers at least five samples."""
    dtau = np.median(np.concatenate([np.diff(nd.tau(t)) for t, _ in segs_fit]))
    best = None
    tried = [hw for hw in half_widths if hw >= 2.0 * dtau] or [max(2.5 * dtau, half_widths[-1])]
    for hw in tried:
        G, B = weak_system_multi(segs_fit, nd, hw, p)
        if len(G) < 2 * nd.lib.n_terms:
            continue
        Xi, lam, err = _fit_select(G, B, nd, segs_val, h, horizon)
        if best is None or err < best[2] * (1 - 1e-9):
            best = (Xi, dict(lam=lam, half_width=hw), err, (G, B))
    if best is None:
        raise ValueError("too few samples for any admissible test function")
    info = dict(best[1], val_err=best[2])
    if return_system:
        info["system"] = best[3]
    return best[0], info


def esindy(G, B, nd, lam, n_boot=100, rng=None):
    """Bootstrap aggregation over rows of the (weak) regression problem.

    Returns the bootstrap coefficient samples (n_boot, p, d) in physical units,
    the inclusion probabilities P(xi_kj != 0) and the median ('bragging') model."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = G.shape[0]
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        samples.append(nd.to_phys(stlsq(G[idx], B[idx], lam)))
    samples = np.array(samples)
    incl = (np.abs(samples) > 0).mean(axis=0)
    med = np.median(samples, axis=0)
    med[incl < 0.5] = 0.0
    return samples, incl, med
