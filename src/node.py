"""Neural ODE models and their training by multiple shooting.

N1  black-box vector field f_theta = MLP, with optional trajectory-wise
    (frozen-mask) dropout for Monte Carlo uncertainty;
N2  sparse library vector field f_Xi(x) = Theta(x) Xi, fitted by
    back-propagating through the integrator and pruned by sequential
    thresholding with validation-based selection.

Both are trained on the same objective (Section 3 of the paper): latent
shooting states S_i at every observation time, short rollouts of K
observation intervals started from randomly chosen S_i, and a quadratic
data-fidelity term tying S to the observations. Short windows keep the
Lipschitz constant of the rollout map moderate, which is what makes the
landscape tractable for chaotic flows; the window length is increased by a
curriculum.
"""
import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .integrators import rk4_step
from .sindy import Nondim, _fit_select, poly_field, short_horizon_error, select_by_validation

import os

DTYPE = torch.float64
# Device for training and rollouts. CPU by default; set SCD_DEVICE=cuda to run
# on a GPU (see RUN_ON_COLAB.ipynb). Index tensors and random generators stay
# on the CPU; only floating-point state lives on DEVICE.
DEVICE = torch.device(os.environ.get("SCD_DEVICE", "cpu"))


def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=DEVICE)


# ---------------------------------------------------------------------------
# Vector fields
# ---------------------------------------------------------------------------
class MLPField(nn.Module):
    """dx/dt = s * g_theta((x - mu) / s) / T.

    Dropout masks, when enabled, are sampled once per trajectory and held
    fixed along the rollout, so that every Monte Carlo sample is itself an
    autonomous ODE. Resampling the mask at every RK stage would instead turn
    the model into a random dynamical system with a different (noise-driven)
    invariant measure."""

    def __init__(self, d, mu, s, T, width=64, depth=2, p_drop=0.0):
        super().__init__()
        self.register_buffer("mu", _t(mu))
        self.register_buffer("s", _t(s))
        self.T = float(T)
        self.hidden = nn.ModuleList([nn.Linear(d if i == 0 else width, width) for i in range(depth)])
        self.out = nn.Linear(width, d)
        self.act = nn.Tanh()
        self.p_drop = p_drop
        self.width = width
        self.masks = None
        # A small output layer starts training from a nearly stationary field,
        # which keeps early multi-step rollouts bounded.
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            self.out.bias.zero_()
        self.to(DTYPE)

    def sample_masks(self, batch, generator=None):
        if self.p_drop <= 0:
            self.masks = None
            return
        keep = 1.0 - self.p_drop
        self.masks = [(torch.bernoulli(torch.full((batch, self.width), keep, dtype=DTYPE),
                                       generator=generator) / keep).to(self.mu.device)
                      for _ in self.hidden]

    def clear_masks(self):
        self.masks = None

    def forward(self, x):
        h = (x - self.mu) / self.s
        for i, layer in enumerate(self.hidden):
            h = self.act(layer(h))
            if self.masks is not None:
                h = h * self.masks[i]
        return self.s * self.out(h) / self.T


class LibraryField(nn.Module):
    """dx/dt = s * Theta(x / s) (Xi * M) / T with a binary support mask M."""

    def __init__(self, lib, s, T, Xi_init):
        super().__init__()
        self.lib = lib
        self.register_buffer("s", _t(s))
        self.T = float(T)
        self.Xi = nn.Parameter(_t(Xi_init).clone())
        self.register_buffer("mask", torch.ones_like(self.Xi))

    def forward(self, x):
        return self.s * (self.lib.torch(x / self.s) @ (self.Xi * self.mask)) / self.T

    def coef_z(self):
        return (self.Xi * self.mask).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------
def rollout_windows(model, x0, tw, n_sub):
    """Integrate a batch of windows. x0: (B, d); tw: (B, K+1) observation
    times. Each observation interval is split into n_sub RK4 steps of
    per-window size, which handles irregular sampling with fixed cost."""
    xs = [x0]
    x = x0
    for k in range(tw.shape[1] - 1):
        hk = ((tw[:, k + 1] - tw[:, k]) / n_sub).unsqueeze(-1)
        for _ in range(n_sub):
            x = rk4_step(model, x, hk)
        xs.append(x)
    return torch.stack(xs, dim=1)          # (B, K+1, d)


@torch.no_grad()
def rollout_model(model, x0, n_steps, h, save_every=1, blowup=1e4, masks_batch=None, generator=None):
    """Forecast with a trained torch field. Returns a NumPy array (n_saved, B, d);
    trajectories are set to NaN after they leave the ball of radius `blowup`."""
    x = _t(np.atleast_2d(x0))
    if isinstance(model, MLPField):
        if masks_batch:
            model.sample_masks(x.shape[0], generator=generator)
        else:
            model.clear_masks()
    out = [x.cpu().numpy().copy()]
    dead = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for k in range(1, n_steps + 1):
        x = rk4_step(model, x, h)
        bad = ~torch.isfinite(x).all(dim=1) | (x.abs().max(dim=1).values > blowup)
        dead |= bad
        x = torch.where(dead.unsqueeze(1), torch.zeros_like(x), x)   # keep arithmetic finite
        if k % save_every == 0:
            xn = x.cpu().numpy().copy()
            xn[dead.cpu().numpy()] = np.nan
            out.append(xn)
    if isinstance(model, MLPField):
        model.clear_masks()
    return np.stack(out, axis=0)


def numpy_field(model):
    """Deterministic NumPy view of a torch field (dropout off)."""
    def f(x):
        with torch.no_grad():
            if isinstance(model, MLPField):
                model.clear_masks()
            return model(_t(x)).cpu().numpy()
    return f


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    schedule: list = field(default_factory=lambda: [(0.1, 300), (0.25, 300), (0.5, 400), (1.0, 600)])
    batch: int = 128
    lr: float = 3e-3
    lr_state: float = 1e-2
    stage_decay: float = 0.6
    w_state: float = 0.5
    l1: float = 0.0
    weight_decay: float = 0.0
    clip: float = 1.0
    h_over_tchar: float = 1.0 / 75.0
    seed: int = 0
    log_every: int = 200


def _smooth_init(Y):
    """Initial shooting states: the raw observations. (A fixed smoothing
    window would bias the states whenever it spans a non-negligible part of
    an oscillation, which is exactly the coarse-sampling regime of interest;
    denoising is left to the joint optimisation over S.)"""
    return Y.copy()


def pack_segments(segs):
    """Concatenate experiments; windows are later drawn so that they never
    straddle two experiments."""
    t = np.concatenate([s[0] for s in segs])
    Y = np.concatenate([s[1] for s in segs])
    seg = np.concatenate([np.full(len(s[0]), k) for k, s in enumerate(segs)])
    S0 = np.concatenate([_smooth_init(s[1]) for s in segs])
    return t, Y, seg, S0


def valid_starts(seg, K):
    i = np.arange(len(seg) - K)
    return torch.as_tensor(i[seg[i] == seg[i + K]])


def window_len(t, seg, H, t_char):
    dt_med = float(np.median(np.concatenate([np.diff(t[seg == k]) for k in np.unique(seg)])))
    n_min = min(int(np.sum(seg == k)) for k in np.unique(seg))
    return int(min(max(1, round(H * t_char / dt_med)), n_min - 1))


def train_multiple_shooting(model, segs, t_char, scale, cfg: TrainConfig, verbose=False):
    """Minimise, over the field parameters and the shooting states S,

        (1/BK) sum_b sum_k || (Phi_{t_{i_b} -> t_{i_b+k}}(S_{i_b}) - y_{i_b+k}) / s ||^2
        + w_state (1/N) sum_i || (S_i - y_i) / s ||^2 + l1 ||Xi||_1,

    with Adam, gradient-norm clipping, and a window-length curriculum."""
    g = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    t, Y, seg, S0 = pack_segments(segs)
    T = _t(t)
    Yt = _t(Y)
    sc = _t(scale)
    S = nn.Parameter(_t(S0))
    gaps = np.diff(t)[seg[1:] == seg[:-1]]
    h_target = cfg.h_over_tchar * t_char
    n_sub = max(1, math.ceil(float(gaps.max()) / h_target - 1e-9))
    is_lib = isinstance(model, LibraryField)

    history = []
    for stage, (H, iters) in enumerate(cfg.schedule):
        K = window_len(t, seg, H, t_char)
        starts_all = valid_starts(seg, K)
        lr = cfg.lr * cfg.stage_decay ** stage
        opt = torch.optim.Adam([{"params": model.parameters(), "lr": lr, "weight_decay": cfg.weight_decay},
                                {"params": [S], "lr": cfg.lr_state * cfg.stage_decay ** stage}])
        last = stage == len(cfg.schedule) - 1
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.1 * lr) if last else None
        for it in range(iters):
            starts = starts_all[torch.randint(0, len(starts_all), (cfg.batch,), generator=g)]
            idx = starts.unsqueeze(1) + torch.arange(K + 1)
            if isinstance(model, MLPField):
                model.sample_masks(cfg.batch, generator=g)
            P = rollout_windows(model, S[starts], T[idx], n_sub)
            l_pred = (((P[:, 1:] - Yt[idx[:, 1:]]) / sc) ** 2).mean()
            l_state = (((S - Yt) / sc) ** 2).mean()
            loss = l_pred + cfg.w_state * l_state
            if is_lib and cfg.l1 > 0:
                loss = loss + cfg.l1 * (model.Xi * model.mask).abs().sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            if sched is not None:
                sched.step()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at stage {stage}, iter {it}")
            if it % 20 == 0:
                history.append((stage, K, it, l_pred.item(), l_state.item()))
            if verbose and it % cfg.log_every == 0:
                print(f"stage {stage} K={K:3d} it {it:4d}  pred {l_pred.item():.4e}  state {l_state.item():.4e}")
    if isinstance(model, MLPField):
        model.clear_masks()
    return dict(S=S.detach().cpu().numpy(), t=t, Y=Y, seg=seg, history=np.array(history), n_sub=n_sub, K_final=K)


def window_loss(model, tr, scale, K, starts):
    """Full-batch shooting loss over a fixed set of windows (used for refits)."""
    T = _t(tr["t"])
    Yt = _t(tr["Y"])
    sc = _t(scale)
    St = _t(tr["S"])
    idx = starts.unsqueeze(1) + torch.arange(K + 1)
    P = rollout_windows(model, St[starts], T[idx], tr["n_sub"])
    return (((P[:, 1:] - Yt[idx[:, 1:]]) / sc) ** 2).mean()


def refit_support(model, tr, scale, mask, iters=15, max_windows=256, seed=0):
    """Re-estimate the active coefficients on a fixed support with L-BFGS.
    Once the support is fixed and the windows are short, the problem is close
    to a nonlinear least-squares problem near its minimiser, where a
    quasi-Newton method converges in a handful of iterations."""
    m = copy.deepcopy(model)
    m.mask.copy_(_t(mask))
    K = tr["K_final"]
    starts = valid_starts(tr["seg"], K)
    g = torch.Generator().manual_seed(seed)
    if len(starts) > max_windows:
        starts = starts[torch.randperm(len(starts), generator=g)[:max_windows]]
    opt = torch.optim.LBFGS([m.Xi], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = window_loss(m, tr, scale, K, starts)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        pass
    with torch.no_grad():
        if not torch.isfinite(m.Xi).all():
            m.Xi.copy_(model.Xi)
    return m


def select_library_support(model, tr, segs_val, nd, scale, h, horizon, n_lam=12, lo=-3.0, hi=-0.4,
                           max_sweeps=4):
    """Sequential thresholding through the integrator (an STLSQ analogue in
    which every least-squares solve is replaced by a trajectory-matching
    refit). Thresholds increase along a relative grid; at each threshold the
    support is pruned and the survivors refitted by L-BFGS until the support
    stops changing, and each threshold starts from the previous solution
    (continuation along the regularisation path). Every model on the path is
    scored on the held-out blocks and the sparsest near-optimal one is kept."""
    Xi_dense = model.coef_z()
    lams = np.max(np.abs(Xi_dense)) * np.logspace(lo, hi, n_lam)
    Xi_phys = nd.to_phys(Xi_dense)
    cands = [Xi_phys]
    errs = [short_horizon_error(poly_field(nd.lib, Xi_phys), segs_val, h, horizon)]
    models = [model]
    current = model
    for lam in lams:
        mask = ((np.abs(current.coef_z()) >= lam) & (current.mask.cpu().numpy() > 0)).astype(float)
        for _ in range(max_sweeps):
            m = refit_support(current, tr, scale, mask)
            new_mask = ((np.abs(m.coef_z()) >= lam) & (mask > 0)).astype(float)
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
            current = m
        current = m
        Xi_phys = nd.to_phys(current.coef_z())
        cands.append(Xi_phys)
        errs.append(short_horizon_error(poly_field(nd.lib, Xi_phys), segs_val, h, horizon))
        models.append(current)
        if not current.mask.any():
            break
    k = select_by_validation(cands, errs)
    lam_list = [0.0] + list(lams)
    return models[k], cands[k], dict(lam=float(lam_list[k]), val_err=float(errs[k]),
                                     path_err=list(map(float, errs)),
                                     path_nnz=[int(np.count_nonzero(c)) for c in cands])


def distill_equations(model, S, nd, segs_val, h, horizon):
    """Symbolic distillation of a black-box field: sparse regression of
    f_theta(S_i) on Theta(S_i) over the denoised shooting states."""
    f = numpy_field(model)
    F = f(S)
    Theta = nd.lib(nd.z(S))
    dZ = F * nd.T / nd.s
    Xi, lam, err = _fit_select(Theta, dZ, nd, segs_val, h, horizon)
    return Xi, dict(lam=float(lam), val_err=float(err))


# ---------------------------------------------------------------------------
# Library model: full-batch quasi-Newton multiple shooting with continuation
# ---------------------------------------------------------------------------
def _ms_loss(model, S, T, Yt, sc, idx, n_sub, w_state):
    P = rollout_windows(model, S[idx[:, 0]], T[idx], n_sub)
    l_pred = (((P[:, 1:] - Yt[idx[:, 1:]]) / sc) ** 2).mean()
    l_state = (((S - Yt) / sc) ** 2).mean()
    return l_pred + w_state * l_state, l_pred, l_state


def train_library_lbfgs(model, segs, t_char, scale, horizons=(0.0, 0.1, 0.25, 0.5, 1.0),
                        iters=60, iters_flowmap=50, w_state=0.5, max_windows=2048, seed=0, verbose=False,
                        continuation=True, bootstrap=False):
    """Fit Xi and the shooting states S jointly by L-BFGS on the full set of
    windows, increasing the window length K along `horizons` (in t_char; 0
    means a single observation interval).

    Continuation in K is the central device against the rugged landscape of
    long-horizon trajectory matching. The first stage holds S at the
    (smoothed) data and fits Xi alone with K = 1: a one-step flow-map
    regression whose basin around the true coefficients is wide. Later stages
    free S (errors-in-variables) and lengthen the windows, each starting inside
    the basin found at the previous length. If a
    stage produces a non-finite or larger loss than the previous stage, its
    parameters are discarded and the curriculum stops there."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    t, Y, seg, S0 = pack_segments(segs)
    T = _t(t)
    Yt = _t(Y)
    sc = _t(scale)
    S = nn.Parameter(_t(S0))
    gaps = np.diff(t)[seg[1:] == seg[:-1]]
    n_sub = max(1, math.ceil(float(gaps.max()) / (t_char / 75.0) - 1e-9))
    K_max = max(1, window_len(t, seg, max(horizons), t_char))
    # geometric schedule 1, 2, 3, 4, 6, 9, 13, ... : each stage starts from a
    # model that is stable on windows only ~1.5x shorter than the new ones
    Ks = sorted({min(K_max, int(round(1.5 ** k))) for k in range(0, 40) if 1.5 ** (k - 1) <= K_max})
    if not continuation:          # ablation: go straight to the longest window
        Ks = [K_max]
    history, K_done = [], None
    prev_loss = np.inf
    # stage 0: flow-map regression with the states held at the data, solved by
    # Levenberg-Marquardt (see lm_flowmap)
    idx1 = valid_starts(seg, 1).unsqueeze(1) + torch.arange(2)
    S_fixed = S.detach()
    if continuation:
        lm_flowmap(model, S_fixed, T, Yt, sc, idx1, n_sub, iters=iters_flowmap)
    if verbose:
        with torch.no_grad():
            print(f"flow-map stage  pred {_ms_loss(model, S_fixed, T, Yt, sc, idx1, n_sub, 0.0)[1].item():.4e}")
    for K in Ks:
        starts = valid_starts(seg, K)
        if bootstrap:             # bagging over shooting windows (ensemble members)
            starts = starts[torch.randint(0, len(starts), (len(starts),), generator=g)]
        if len(starts) > max_windows:
            starts = starts[torch.randperm(len(starts), generator=g)[:max_windows]]
        idx = starts.unsqueeze(1) + torch.arange(K + 1)
        snap = (model.Xi.detach().clone(), S.detach().clone())
        with torch.no_grad():
            loss_start = _ms_loss(model, S, T, Yt, sc, idx, n_sub, w_state)[0].item()
        opt = torch.optim.LBFGS([model.Xi, S], lr=1.0, max_iter=iters, history_size=20,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss, _, _ = _ms_loss(model, S, T, Yt, sc, idx, n_sub, w_state)
            if not torch.isfinite(loss):
                return loss
            loss.backward()
            return loss

        ok = True
        try:
            opt.step(closure)
        except Exception:
            ok = False
        with torch.no_grad():
            loss, lp, ls = _ms_loss(model, S, T, Yt, sc, idx, n_sub, w_state)
        finite = ok and np.isfinite(loss.item()) and torch.isfinite(model.Xi).all() and torch.isfinite(S).all()
        if not finite or (np.isfinite(loss_start) and loss.item() > loss_start * (1 + 1e-9)):
            with torch.no_grad():
                model.Xi.copy_(snap[0])
                S.copy_(snap[1])
            if verbose:
                print(f"K={K}: rejected (loss {loss.item():.3e} vs start {loss_start:.3e}); curriculum stops")
            break
        l1_now = loss.item()
        prev_loss = min(prev_loss, l1_now)
        K_done = K
        history.append((K, loss.item(), lp.item(), ls.item()))
        if verbose:
            print(f"K={K:3d}  loss {loss.item():.4e}  pred {lp.item():.4e}  state {ls.item():.4e}")
    K_done = K_done or 1
    return dict(S=S.detach().cpu().numpy(), t=t, Y=Y, seg=seg, history=np.array(history),
                n_sub=n_sub, K_final=K_done)


def multistart_library(lib, s, t_char, inits, segs, segs_val, nd, scale, h, horizon, verbose=False, **kw):
    kw.setdefault("seed", 0)
    """Run the continuation from several initial coefficient matrices and keep
    the run with the lowest held-out error. Returns (model, train_out, table)."""
    best, table = None, []
    for name, Xi0 in inits.items():
        m = LibraryField(lib, s, t_char, Xi0)
        tr = train_library_lbfgs(m, segs, t_char, scale, verbose=verbose, **kw)
        err = short_horizon_error(poly_field(lib, nd.to_phys(m.coef_z())), segs_val, h, horizon)
        table.append((name, float(err), int(tr["K_final"])))
        if verbose:
            print(f"init {name}: val {err:.4e}, K_final {tr['K_final']}")
        if best is None or err < best[0]:
            best = (err, m, tr)
    return best[1], best[2], table


def lm_flowmap(model, S, T, Yt, sc, idx, n_sub, iters=50, mu0=1e-3):
    """Levenberg-Marquardt on the one-step residuals r(Xi) = (Phi_Xi(S_i) - y_{i+1}) / s.

    The library Gram matrix of a dissipative attractor can be very badly
    conditioned (kappa ~ 1e4 for the LV4 model even with several experiments),
    and first-order or limited-memory methods stall along its small singular
    directions. With only n_terms * d unknowns the Jacobian is cheap to form
    exactly (forward-mode AD through the RK4 steps), and the damped
    Gauss-Newton step with Marquardt's diagonal scaling is invariant to the
    column scaling of Theta, which is what makes it robust here."""
    lib, s_, Tc = model.lib, model.s, model.T
    mask = model.mask.detach()
    shape = model.Xi.shape
    x0 = S[idx[:, 0]].detach()
    tw = T[idx]
    target = Yt[idx[:, 1]]

    def resid(theta):
        Xi = theta.view(shape) * mask
        f = lambda x: s_ * (lib.torch(x / s_) @ Xi) / Tc
        P = rollout_windows(f, x0, tw, n_sub)[:, 1]
        return ((P - target) / sc).reshape(-1)

    theta = model.Xi.detach().reshape(-1).clone()
    r = resid(theta)
    cost = float((r ** 2).sum())
    mu = mu0
    for _ in range(iters):
        J = torch.func.jacfwd(resid)(theta)
        A = J.T @ J
        g = J.T @ r
        D = torch.diag(torch.clamp(torch.diagonal(A), min=1e-12))
        improved = False
        for _ in range(12):
            try:
                delta = torch.linalg.solve(A + mu * D, -g)
            except RuntimeError:
                mu *= 10.0
                continue
            r_new = resid(theta + delta)
            c_new = float((r_new ** 2).sum())
            if np.isfinite(c_new) and c_new < cost:
                theta, r, improved = theta + delta, r_new, True
                rel = (cost - c_new) / max(cost, 1e-300)
                cost = c_new
                mu = max(mu / 3.0, 1e-12)
                break
            mu *= 4.0
        if not improved or rel < 1e-12:
            break
    with torch.no_grad():
        model.Xi.copy_(theta.view(shape))
    return cost
