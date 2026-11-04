"""Polynomial feature library Theta(x) in NumPy and PyTorch, with diagonal
rescaling that preserves the monomial support.

We work in scaled coordinates z = x / s (no centring). For a monomial with
exponent vector p, theta_p(x) = c_p * theta_p(z) with c_p = prod_i s_i^{p_i},
so dx_i/dt = sum_p xi_{p,i} theta_p(x) becomes
    dz_i/dt = sum_p (xi_{p,i} c_p / s_i) theta_p(z).
The map Xi <-> Xi_z is diagonal in both indices, so the sparsity pattern is
identical in both coordinate systems. Centring would mix monomials of
different degree and destroy the support, which is why we avoid it.
"""
import numpy as np
import torch

from .systems import monomial_powers, monomial_names


class PolyLibrary:
    def __init__(self, d, degree=2, symbols=None):
        self.d, self.degree = d, degree
        self.powers = np.array(monomial_powers(d, degree), dtype=int)   # (p, d)
        self.names = monomial_names(d, degree, symbols)
        self.n_terms = len(self.powers)
        self._powers_t = None

    # -- NumPy ------------------------------------------------------------
    def __call__(self, X):
        """Theta(X) for X of shape (..., d) -> (..., n_terms)."""
        X = np.asarray(X)
        return np.prod(X[..., None, :] ** self.powers, axis=-1)

    # -- PyTorch ----------------------------------------------------------
    def torch(self, X):
        if self._powers_t is None or self._powers_t.device != X.device:
            self._powers_t = torch.as_tensor(self.powers, device=X.device)
        # Integer powers via repeated products are exact and differentiable;
        # torch.pow with an integer tensor exponent is equivalent here.
        return torch.prod(X.unsqueeze(-2) ** self._powers_t.to(X.dtype), dim=-1)

    # -- coordinate changes -------------------------------------------------
    def term_scale(self, s):
        """c_p = prod_i s_i^{p_i} for every monomial."""
        return np.prod(np.asarray(s)[None, :] ** self.powers, axis=1)

    def to_scaled(self, Xi, s):
        """Coefficients in x-coordinates -> coefficients in z = x/s coordinates."""
        return Xi * self.term_scale(s)[:, None] / np.asarray(s)[None, :]

    def from_scaled(self, Xi_z, s):
        return Xi_z * np.asarray(s)[None, :] / self.term_scale(s)[:, None]

    def pretty(self, Xi, symbols=None, tol=1e-10, fmt="{:+.3f}"):
        symbols = symbols or [f"x_{i+1}" for i in range(self.d)]
        lines = []
        for i in range(self.d):
            terms = []
            for k in range(self.n_terms):
                if abs(Xi[k, i]) <= tol:
                    continue
                c = fmt.format(Xi[k, i])
                terms.append(c if self.names[k] == "1" else f"{c} {self.names[k]}")
            lines.append(f"d{symbols[i]}/dt = " + (" ".join(terms) if terms else "0"))
        return "\n".join(lines)
