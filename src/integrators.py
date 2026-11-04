"""Fixed-step explicit Runge-Kutta integrators (NumPy and PyTorch).

Learned models are always rolled out with the same classical RK4 scheme and
the same step h, so differences in forecast quality reflect the vector field
and not the time stepper. Rollouts are batched over initial conditions.
"""
import numpy as np
import torch


def rk4_step(f, x, h):
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout_np(f, x0, n_steps, h, save_every=1, blowup=1e4):
    """Integrate a batch of initial conditions x0 (B, d) for n_steps of size h.

    Trajectories whose norm exceeds `blowup` are frozen and set to NaN from
    that point on; this records finite-time blow-up of a misidentified model
    without letting overflow contaminate the rest of the batch."""
    x = np.array(x0, dtype=float, copy=True)
    squeeze = x.ndim == 1
    x = np.atleast_2d(x)
    n_save = n_steps // save_every + 1
    out = np.full((n_save, *x.shape), np.nan)
    out[0] = x
    alive = np.ones(x.shape[0], bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, n_steps + 1):
            if alive.any():
                x[alive] = rk4_step(f, x[alive], h)
                bad = ~np.isfinite(x).all(axis=1) | (np.abs(x).max(axis=1) > blowup)
                alive &= ~bad
                x[~alive] = np.nan
            if k % save_every == 0:
                out[k // save_every] = x
    return out[:, 0] if squeeze else out


def rollout_torch(f, x0, n_steps, h, save_every=1):
    """Differentiable RK4 rollout. x0: (B, d). Returns (n_saved, B, d)."""
    xs = [x0]
    x = x0
    for k in range(1, n_steps + 1):
        x = rk4_step(f, x, h)
        if k % save_every == 0:
            xs.append(x)
    return torch.stack(xs, dim=0)
