"""Reference dynamical systems, their exact coefficient matrices in a monomial
library, and high-accuracy integration with SciPy.

All three systems are polynomial of degree <= 2, so the ground-truth vector
field lies exactly in span(Theta_2), the degree-2 monomial library. This makes
equation recovery a well-posed sparse regression problem and lets us score
support recovery without ambiguity.
"""
from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Monomial library bookkeeping (shared with src/library.py)
# ---------------------------------------------------------------------------
def monomial_powers(d, degree=2):
    """Exponent tuples of all monomials of total degree <= `degree` in d variables,
    ordered as 1, x_1..x_d, x_1^2, x_1 x_2, ... (graded lexicographic)."""
    powers = []
    for deg in range(degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            p = [0] * d
            for i in combo:
                p[i] += 1
            powers.append(tuple(p))
    return powers


def monomial_names(d, degree=2, symbols=None):
    symbols = symbols or [f"x_{i+1}" for i in range(d)]
    names = []
    for p in monomial_powers(d, degree):
        if sum(p) == 0:
            names.append("1")
            continue
        parts = []
        for s, e in zip(symbols, p):
            if e == 1:
                parts.append(s)
            elif e > 1:
                parts.append(f"{s}^{e}")
        names.append(" ".join(parts))
    return names


def _index(powers, p):
    return powers.index(tuple(p))


# ---------------------------------------------------------------------------
# System definitions
# ---------------------------------------------------------------------------
@dataclass
class System:
    name: str
    d: int
    params: dict
    symbols: list
    x0: np.ndarray                 # a point on (or near) the attractor
    t_char: float                  # characteristic (dominant) period
    lyap: float                    # largest Lyapunov exponent (0 if not chaotic)
    Xi_true: np.ndarray = field(default=None)   # (n_terms, d)

    def rhs(self, t, x):
        raise NotImplementedError

    def rhs_batch(self, x):
        """Vectorised vector field, x of shape (..., d)."""
        raise NotImplementedError

    @property
    def lyap_time(self):
        return np.inf if self.lyap <= 0 else 1.0 / self.lyap


class Lorenz63(System):
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        super().__init__(
            name="lorenz", d=3,
            params=dict(sigma=sigma, rho=rho, beta=beta),
            symbols=["x", "y", "z"],
            x0=np.array([-8.0, 7.0, 27.0]),
            t_char=0.75,
            lyap=0.9056,
        )
        P = monomial_powers(3)
        Xi = np.zeros((len(P), 3))
        Xi[_index(P, (1, 0, 0)), 0] = -sigma
        Xi[_index(P, (0, 1, 0)), 0] = sigma
        Xi[_index(P, (1, 0, 0)), 1] = rho
        Xi[_index(P, (0, 1, 0)), 1] = -1.0
        Xi[_index(P, (1, 0, 1)), 1] = -1.0
        Xi[_index(P, (1, 1, 0)), 2] = 1.0
        Xi[_index(P, (0, 0, 1)), 2] = -beta
        self.Xi_true = Xi

    def rhs(self, t, s):
        x, y, z = s
        p = self.params
        return [p["sigma"] * (y - x), x * (p["rho"] - z) - y, x * y - p["beta"] * z]

    def rhs_batch(self, s):
        x, y, z = s[..., 0], s[..., 1], s[..., 2]
        p = self.params
        return np.stack([p["sigma"] * (y - x), x * (p["rho"] - z) - y,
                         x * y - p["beta"] * z], axis=-1)


class CompetitiveLV4(System):
    """Four-species competitive Lotka-Volterra model,
        dx_i/dt = r_i x_i (1 - sum_j a_ij x_j),
    with the parameters of Vano et al. (2006), for which the flow on the
    positive orthant carries a chaotic attractor."""
    R = np.array([1.0, 0.72, 1.53, 1.27])
    A = np.array([[1.00, 1.09, 1.52, 0.00],
                  [0.00, 1.00, 0.44, 1.36],
                  [2.33, 0.00, 1.00, 0.47],
                  [1.21, 0.51, 0.35, 1.00]])

    def __init__(self):
        super().__init__(
            name="lv4", d=4,
            params=dict(r=self.R.copy(), A=self.A.copy()),
            symbols=["x_1", "x_2", "x_3", "x_4"],
            x0=np.array([0.30, 0.46, 0.13, 0.54]),
            t_char=np.nan,   # estimated from the power spectrum in notebook 01
            lyap=0.0203,
        )
        P = monomial_powers(4)
        Xi = np.zeros((len(P), 4))
        for i in range(4):
            e_i = [0] * 4
            e_i[i] = 1
            Xi[_index(P, e_i), i] = self.R[i]
            for j in range(4):
                if self.A[i, j] == 0.0:
                    continue
                p = [0] * 4
                p[i] += 1
                p[j] += 1
                Xi[_index(P, p), i] += -self.R[i] * self.A[i, j]
        self.Xi_true = Xi

    def rhs(self, t, x):
        return self.R * x * (1.0 - self.A @ x)

    def rhs_batch(self, x):
        return self.R * x * (1.0 - x @ self.A.T)


class PredatorPreyLV2(System):
    """Classical two-species predator-prey model (periodic, not chaotic).
    Used only as a sanity check on a conservative, non-chaotic flow."""
    def __init__(self, alpha=2.0 / 3.0, beta=4.0 / 3.0, delta=1.0, gamma=1.0):
        super().__init__(
            name="lv2", d=2,
            params=dict(alpha=alpha, beta=beta, delta=delta, gamma=gamma),
            symbols=["u", "v"],
            x0=np.array([1.8, 0.9]),
            t_char=np.nan,
            lyap=0.0,
        )
        P = monomial_powers(2)
        Xi = np.zeros((len(P), 2))
        Xi[_index(P, (1, 0)), 0] = alpha
        Xi[_index(P, (1, 1)), 0] = -beta
        Xi[_index(P, (1, 1)), 1] = delta
        Xi[_index(P, (0, 1)), 1] = -gamma
        self.Xi_true = Xi

    def rhs(self, t, s):
        u, v = s
        p = self.params
        return [p["alpha"] * u - p["beta"] * u * v, p["delta"] * u * v - p["gamma"] * v]

    def rhs_batch(self, s):
        u, v = s[..., 0], s[..., 1]
        p = self.params
        return np.stack([p["alpha"] * u - p["beta"] * u * v,
                         p["delta"] * u * v - p["gamma"] * v], axis=-1)


SYSTEMS = {"lorenz": Lorenz63, "lv4": CompetitiveLV4, "lv2": PredatorPreyLV2}


def get_system(name):
    return SYSTEMS[name]()


# ---------------------------------------------------------------------------
# Ground-truth simulation
# ---------------------------------------------------------------------------
def simulate(system, x0, t_end, dt, t0=0.0, rtol=1e-10, atol=1e-12):
    """Integrate with an 8th-order Dormand-Prince scheme (DOP853) and dense
    output. Tight tolerances keep the global error well below the smallest
    noise level we study, so the 'truth' is not itself a source of error on
    horizons of a few Lyapunov times."""
    t_eval = t0 + dt * np.arange(int(round((t_end - t0) / dt)) + 1)
    sol = solve_ivp(system.rhs, (t0, t_eval[-1]), np.asarray(x0, float),
                    method="DOP853", t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T


def corrupt(X, dt_fine, dt_obs, noise_level, rng, scale=None, irregular=False):
    """Sub-sample a fine trajectory and add Gaussian measurement noise.

    noise_level is relative: eta_k ~ N(0, (noise_level * std_i)^2) per coordinate.
    With irregular=True, observation times are drawn uniformly at random with the
    same expected density as the regular grid (sorted, endpoints kept)."""
    stride = int(round(dt_obs / dt_fine))
    n_fine = X.shape[0]
    if irregular:
        n_obs = (n_fine - 1) // stride + 1
        idx = np.sort(rng.choice(np.arange(1, n_fine - 1), size=n_obs - 2, replace=False))
        idx = np.concatenate([[0], idx, [n_fine - 1]])
    else:
        idx = np.arange(0, n_fine, stride)
    scale = X.std(axis=0) if scale is None else scale
    Y = X[idx] + noise_level * scale * rng.standard_normal((len(idx), X.shape[1]))
    return idx, Y
