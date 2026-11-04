"""Experimental protocol: datasets, test sets, and the evaluation of any
learned vector field under one common set of forecasting metrics.

Time is measured in units of a characteristic period t_char (the mean
interval between successive maxima of the oscillating coordinate). Lorenz-63
and the four-species LV model have almost the same Lyapunov time in these
units (lambda_max * t_char = 0.68 and 0.70), so one sampling grid expressed
in t_char probes both systems at comparable resolution.
"""
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from . import metrics as M
from .integrators import rollout_np
from .library import PolyLibrary
from .sindy import Nondim, fit_dense_fd, fit_sindy, fit_weak_sindy, poly_field
from .systems import get_system, simulate, corrupt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

T_CHAR = {"lorenz": 0.75, "lv4": 35.0, "lv2": 8.0}
DT_RATIOS = [1 / 75, 1 / 15, 2 / 15, 4 / 15]          # Lorenz: 0.01, 0.05, 0.10, 0.20
NOISES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
T_TRAIN = 80.0 / 3.0                                   # in t_char (20 time units for Lorenz)
VAL_FRAC = 0.2
H_RATIO = 1.0 / 75.0                                   # rollout step h = t_char / 75
N_TEST = 16
N_EXP = 4                                              # independent experiments per record
PERT = 0.5                                             # IC perturbation, in attractor std
OOD_DIST = 2.0                                         # OOD ICs: >= 2 std from the attractor


def time_unit(sysobj):
    """Forecast horizons are reported in Lyapunov times for chaotic systems and
    in characteristic periods otherwise."""
    return sysobj.lyap if sysobj.lyap > 0 else 1.0 / T_CHAR[sysobj.name]


# neural window-length curriculum (fractions of t_char, iterations per stage)
N1_SCHEDULE = [(0.1, 200), (0.25, 250), (0.5, 300), (1.0, 450)]


def get_sys(name):
    s = get_system(name)
    s.t_char = T_CHAR[name]
    return s


# ---------------------------------------------------------------------------
# Reference data (one per system, cached)
# ---------------------------------------------------------------------------
def reference(name, force=False):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"reference_{name}.npz")
    if os.path.exists(path) and not force:
        return dict(np.load(path))
    s = get_sys(name)
    tc = s.t_char
    h = H_RATIO * tc
    # burn-in onto the attractor
    _, Xb = simulate(s, s.x0, 100 * tc, h)
    x_att = Xb[-1]
    t_ref, X_ref = simulate(s, x_att, 220 * tc, h)
    X_ref = X_ref[int(20 * tc / h):]                     # discard 20 periods
    std = X_ref.std(axis=0)
    rng = np.random.default_rng(1000)
    # in-distribution initial conditions: points on the attractor, far apart in time
    id_idx = np.sort(rng.choice(np.arange(0, len(X_ref), int(5 * tc / h)), N_TEST, replace=False))
    X0_id = X_ref[id_idx]
    # out-of-distribution initial conditions: at least OOD_DIST standard
    # deviations away from every point of the reference orbit
    tree = cKDTree(X_ref / std)
    lo, hi = X_ref.min(0), X_ref.max(0)
    span = hi - lo
    if name == "lorenz":
        box_lo, box_hi = lo - 0.5 * span, hi + 0.5 * span
    else:   # stay inside the invariant positive orthant
        box_lo, box_hi = np.maximum(1e-3, lo - 0.5 * span), hi + 0.5 * span
    rng = np.random.default_rng(2000)
    X0_ood = []
    while len(X0_ood) < N_TEST:
        c = box_lo + (box_hi - box_lo) * rng.random(s.d)
        if tree.query(c / std)[0] > OOD_DIST:
            X0_ood.append(c)
    X0_ood = np.array(X0_ood)
    # ground-truth forecasts
    unit = time_unit(s)
    horizon = 15.0 / unit if s.lyap > 0 else 10.0 * tc
    n_steps = int(round(horizon / h))
    def truth(X0):
        return np.stack([simulate(s, x0, n_steps * h, h)[1] for x0 in X0], axis=1)
    out = dict(x_att=x_att, X_ref=X_ref, std=std, X0_id=X0_id, X0_ood=X0_ood,
               X_id=truth(X0_id), X_ood=truth(X0_ood), h=h, n_steps=n_steps,
               t_char=tc, lyap=s.lyap)
    np.savez_compressed(path, **out)
    return out


def make_dataset(name, dt_ratio, noise, seed, irregular=False, n_exp=N_EXP, pert=PERT):
    """A training record of total length T_TRAIN t_char, made of n_exp
    independent experiments. Each experiment starts from a random point of the
    attractor displaced by a Gaussian perturbation of `pert` standard
    deviations per coordinate, and is split in time into a fitting block (80%)
    and a validation block (20%).

    Several short experiments excite directions of the library that a single
    orbit on the attractor never visits: for the LV4 model the smallest singular
    value of the normalised library matrix grows by a factor of ~25 at equal
    record length (notebook 01)."""
    s = get_sys(name)
    ref = reference(name)
    tc = s.t_char
    dt_fine = tc / 750.0
    rng = np.random.default_rng(seed)
    segs = []
    for e in range(n_exp):
        x0 = ref["X_ref"][rng.integers(0, len(ref["X_ref"]))] + pert * ref["std"] * rng.standard_normal(s.d)
        if name != "lorenz":
            x0 = np.maximum(x0, 1e-3)            # positive orthant is invariant for LV
        t_f, X_f = simulate(s, x0, T_TRAIN * tc / n_exp, dt_fine)
        idx, Y = corrupt(X_f, dt_fine, dt_ratio * tc, noise, rng, scale=ref["std"], irregular=irregular)
        t = t_f[idx]
        n_fit = int(np.searchsorted(t, (1 - VAL_FRAC) * t[-1]))
        segs.append(dict(t=t, Y=Y, X=X_f[idx], n_fit=n_fit, x0=x0))
    return dict(system=name, dt=dt_ratio * tc, dt_ratio=dt_ratio, noise=noise, seed=seed,
                irregular=irregular, segs=segs,
                fit=[(g["t"][:g["n_fit"]], g["Y"][:g["n_fit"]]) for g in segs],
                val=[(g["t"][g["n_fit"]:], g["Y"][g["n_fit"]:]) for g in segs])


def nondim_for(ds, lib):
    """Scales from the noisy fitting data only (no oracle): RMS per coordinate."""
    Y = np.concatenate([Y for _, Y in ds["fit"]])
    return Nondim(lib, np.sqrt(np.mean(Y ** 2, axis=0)), T_CHAR[ds["system"]])


# ---------------------------------------------------------------------------
# Common evaluation of a vector field
# ---------------------------------------------------------------------------
def evaluate_field(f, name, rollout=None, lyap_steps=None, long_periods=200):
    """Forecast and long-term statistics of a vector field f (NumPy callable on
    batches). `rollout(f, X0, n_steps, h)` can be overridden for torch models."""
    s = get_sys(name)
    ref = reference(name)
    h, tc, n = float(ref["h"]), float(ref["t_char"]), int(ref["n_steps"])
    unit = time_unit(s)
    roll = rollout or (lambda f_, X0, k, hh: rollout_np(f_, X0, k, hh))
    t = h * np.arange(n + 1)
    std = ref["std"]
    res = {}
    for tag in ("id", "ood"):
        pred = roll(f, ref[f"X0_{tag}"], n, h)
        true = ref[f"X_{tag}"]
        vpt = M.valid_prediction_time(pred, true, t, std, unit)
        e = M.nrmse_t(pred, true, std)
        k1 = int(round(tc / h))
        res[f"vpt_{tag}"] = float(np.mean(vpt))
        res[f"vpt_{tag}_sd"] = float(np.std(vpt))
        res[f"nrmse1_{tag}"] = float(np.nanmean(np.where(np.isfinite(e[:k1 + 1]), e[:k1 + 1], np.nan)))
        res[f"diverged_{tag}"] = float(np.mean(~np.isfinite(pred[-1]).all(axis=1)))
    # long-run statistics on the attractor
    n_long = int(round(long_periods * tc / h))
    n_burn = int(round(20 * tc / h))
    Xl = roll(f, ref["X_ref"][:1], n_long + n_burn, h)[n_burn:, 0]
    res["diverged_long"] = float(not np.isfinite(Xl).all())
    res["w1"] = M.invariant_measure_distance(Xl if np.isfinite(Xl).all() else None, ref["X_ref"], std)
    if s.lyap > 0 and np.isfinite(Xl).all():
        ls = lyap_steps or int(round(100 * tc / h))
        res["lyap_est"] = float(M.lyapunov_benettin(f, ref["X_ref"][0], h, ls, transient=10))
    else:
        res["lyap_est"] = np.nan
    return res, Xl


# ---------------------------------------------------------------------------
# Classical pipeline
# ---------------------------------------------------------------------------
def run_classical(name, dt_ratio, noise, seed, irregular=False, evaluate=True, methods=None, **ds_kw):
    ds = make_dataset(name, dt_ratio, noise, seed, irregular, **ds_kw)
    s = get_sys(name)
    lib = PolyLibrary(s.d, 2, s.symbols)
    nd = nondim_for(ds, lib)
    h = H_RATIO * s.t_char
    horizon = 1.0 * s.t_char
    vargs = (ds["val"], h, horizon)
    methods = methods or ["C1-OLS", "C2-SINDy-FD", "C2-SINDy-SG", "C3-WSINDy"]
    rows = []
    for m in methods:
        if m == "C2-SINDy-SG" and irregular:
            continue
        t0 = time.time()
        try:
            if m == "C1-OLS":
                Xi, info = fit_dense_fd(ds["fit"], nd)
            elif m == "C2-SINDy-FD":
                Xi, info = fit_sindy(ds["fit"], nd, *vargs, deriv="fd")
            elif m == "C2-SINDy-SG":
                Xi, info = fit_sindy(ds["fit"], nd, *vargs, deriv="sg")
            elif m == "C3-WSINDy":
                Xi, info = fit_weak_sindy(ds["fit"], nd, *vargs)
        except Exception as exc:          # record failures rather than abort a sweep
            Xi, info = None, dict(error=repr(exc))
        fit_time = time.time() - t0
        row = dict(system=name, method=m, dt=ds["dt"], dt_ratio=dt_ratio, noise=noise, seed=seed,
                   irregular=irregular, fit_time=fit_time,
                   **{k: v for k, v in info.items() if np.isscalar(v)})
        if Xi is not None:
            row["coef_err"] = M.coef_error(Xi, s.Xi_true)
            row.update(M.support_scores(Xi, s.Xi_true))
            if evaluate:
                ev, _ = evaluate_field(poly_field(lib, Xi), name)
                row.update(ev)
        row["Xi"] = Xi
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Neural pipeline
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(ROOT, "results", "models")


def _tag(name, method, dt_ratio, noise, seed, extra=""):
    return f"{name}_{method}_dt{dt_ratio:.4f}_n{noise:.3f}_s{seed}{extra}"


def build_neural(method, ds, lib, nd, p_drop=0.0, width=64, depth=2):
    import torch
    from .node import MLPField, LibraryField, DEVICE
    from .sindy import short_horizon_error
    s = get_sys(ds["system"])
    Y = np.concatenate([Y for _, Y in ds["fit"]])
    if method.startswith("N1"):
        return MLPField(s.d, Y.mean(0), Y.std(0), s.t_char, width=width, depth=depth, p_drop=p_drop).to(DEVICE)
    # N2: start from whichever of {0, dense FD least squares} predicts the
    # validation blocks better; a poor derivative-based start at coarse
    # sampling is worse than the trivial field.
    h, hz = H_RATIO * s.t_char, s.t_char
    Xi_ols, _ = fit_dense_fd(ds["fit"], nd)
    e_ols = short_horizon_error(poly_field(lib, Xi_ols), ds["val"], h, hz)
    e_zero = short_horizon_error(lambda x: np.zeros_like(x), ds["val"], h, hz)
    Xi0 = nd.to_z(Xi_ols) if e_ols < e_zero else np.zeros((lib.n_terms, s.d))
    return LibraryField(lib, nd.s, s.t_char, Xi0).to(DEVICE)


def library_inits(ds, lib, nd, warm=False):
    """Initial coefficient matrices for the multistart: the trivial field and
    the dense finite-difference least-squares fit; with warm=True also the
    weak-SINDy estimate (the 'hybrid' variant N2-WS)."""
    s = get_sys(ds["system"])
    inits = {"zero": np.zeros((lib.n_terms, s.d)),
             "ols": nd.to_z(fit_dense_fd(ds["fit"], nd)[0])}
    if warm:
        try:
            Xi_w, _ = fit_weak_sindy(ds["fit"], nd, ds["val"], H_RATIO * s.t_char, s.t_char)
            inits["wsindy"] = nd.to_z(Xi_w)
        except ValueError:
            pass
    return inits


def run_neural(name, dt_ratio, noise, seed, method, cfg=None, evaluate=True, p_drop=0.0,
               train_seed=None, tag_extra="", force=False, verbose=False, ds_kw=None, bootstrap=False):
    """Train (or load from cache) one neural model and score it.
    method: 'N1-MLP' (black box + symbolic distillation) or 'N2-Lib'."""
    import torch
    from .node import (TrainConfig, train_multiple_shooting, select_library_support,
                       distill_equations, numpy_field)
    os.makedirs(MODEL_DIR, exist_ok=True)
    ds = make_dataset(name, dt_ratio, noise, seed, **(ds_kw or {}))
    s = get_sys(name)
    lib = PolyLibrary(s.d, 2, s.symbols)
    nd = nondim_for(ds, lib)
    h, horizon = H_RATIO * s.t_char, 1.0 * s.t_char
    Yfit = np.concatenate([Y for _, Y in ds["fit"]])
    scale = Yfit.std(0)
    cfg = cfg or TrainConfig(schedule=list(N1_SCHEDULE))
    cfg.seed = seed if train_seed is None else train_seed
    if method.startswith("N2") and cfg.l1 == 0.0:
        cfg.l1 = 1e-6
    tag = _tag(name, method, dt_ratio, noise, seed, tag_extra)
    path = os.path.join(MODEL_DIR, tag + ".pt")
    t0 = time.time()
    if os.path.exists(path) and not force:
        ck = torch.load(path, weights_only=False, map_location="cpu")
        model = build_neural(method, ds, lib, nd, p_drop=p_drop)
        model.load_state_dict(ck["state"])
        tr, info, fit_time = ck["train"], ck["info"], ck["fit_time"]
        Xi = ck["Xi"]
    else:
        if method.startswith("N2"):
            from .node import multistart_library
            inits = library_inits(ds, lib, nd, warm=method.endswith("WS"))
            model, tr, table = multistart_library(lib, nd.s, s.t_char, inits, ds["fit"], ds["val"], nd,
                                                  scale, h, horizon, verbose=verbose, seed=cfg.seed,
                                                  bootstrap=bootstrap)
            model, Xi, info = select_library_support(model, tr, ds["val"], nd, scale, h, horizon)
            info["init"] = min(table, key=lambda r: r[1])[0]
            info["K_final"] = int(tr["K_final"])
        else:
            model = build_neural(method, ds, lib, nd, p_drop=p_drop)
            tr = train_multiple_shooting(model, ds["fit"], s.t_char, scale, cfg, verbose=verbose)
            Xi, info = distill_equations(model, tr["S"], nd, ds["val"], h, horizon)
        fit_time = time.time() - t0
        torch.save(dict(state={k: v.cpu() for k, v in model.state_dict().items()}, train=tr, info=info, fit_time=fit_time, Xi=Xi,
                        cfg=cfg.__dict__), path)
    row = dict(system=name, method=method + tag_extra, dt=ds["dt"], dt_ratio=dt_ratio, noise=noise,
               seed=seed, irregular=False, fit_time=fit_time,
               **{k: v for k, v in info.items() if np.isscalar(v)})
    row["coef_err"] = M.coef_error(Xi, s.Xi_true)
    row.update(M.support_scores(Xi, s.Xi_true))
    if evaluate:
        ev, _ = evaluate_field(numpy_field(model), name)
        row.update(ev)
        if method.startswith("N1"):
            # forecasting skill of the distilled equations, for reference
            evd, _ = evaluate_field(poly_field(lib, Xi), name)
            row.update({f"distilled_{k}": v for k, v in evd.items()})
    row["Xi"] = Xi
    return row, model, tr
