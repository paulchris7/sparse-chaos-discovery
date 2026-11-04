"""Picklable job wrappers for parallel parameter sweeps (one process per core)."""
import os
import pickle
import traceback

import numpy as np

from . import experiment as E


def _threads():
    import torch
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def classical_job(args):
    name, r, eps, seed, kw = args
    try:
        return E.run_classical(name, r, eps, seed, **kw)
    except Exception:
        return [dict(system=name, dt_ratio=r, noise=eps, seed=seed, error=traceback.format_exc())]


def neural_job(args):
    _threads()
    name, r, eps, seed, method, kw = args
    try:
        row, _, _ = E.run_neural(name, r, eps, seed, method, **kw)
        return row
    except Exception:
        return dict(system=name, dt_ratio=r, noise=eps, seed=seed, method=method, error=traceback.format_exc())


def run_parallel(fn, jobs, workers=2, log_every=10):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs)):
            res = f.result()
            out.extend(res if isinstance(res, list) else [res])
            if (k + 1) % log_every == 0 or k + 1 == len(jobs):
                print(f"{k + 1}/{len(jobs)} jobs done", flush=True)
    return out


def split_rows(rows):
    """Separate scalar metrics (for CSV) from coefficient matrices (for pickle)."""
    import pandas as pd
    coefs = {}
    flat = []
    for r in rows:
        r = dict(r)
        Xi = r.pop("Xi", None)
        key = (r.get("system"), r.get("method"), round(r.get("dt_ratio", np.nan), 5), r.get("noise"),
               r.get("seed"), r.get("irregular", False))
        coefs[key] = Xi
        flat.append({k: v for k, v in r.items() if np.isscalar(v) or v is None})
    return pd.DataFrame(flat), coefs


def save_rows(rows, csv_path, pkl_path):
    df, coefs = split_rows(rows)
    df.to_csv(csv_path, index=False)
    with open(pkl_path, "wb") as fh:
        pickle.dump(coefs, fh)
    return df, coefs


def basin_job(args):
    """One random initialisation Xi_0 = scale * N(0, mean(Xi_OLS^2)) of the
    sparse-library model (nondimensional coordinates), trained either
    with the full continuation or directly at the longest window. Returns the
    dense coefficient error (no thresholding) and the final window loss."""
    _threads()
    import torch
    from .library import PolyLibrary
    from .node import LibraryField, train_library_lbfgs
    name, r, eps, init_seed, continuation, scale = args
    ds = E.make_dataset(name, r, eps, 0)
    s = E.get_sys(name)
    lib = PolyLibrary(s.d, 2, s.symbols)
    nd = E.nondim_for(ds, lib)
    Yfit = np.concatenate([Y for _, Y in ds["fit"]])
    Xi_ols = nd.to_z(E.fit_dense_fd(ds["fit"], nd)[0])
    rng = np.random.default_rng(10_000 + init_seed)
    Xi0 = scale * rng.standard_normal(Xi_ols.shape) * np.sqrt(np.mean(Xi_ols ** 2))
    m = LibraryField(lib, nd.s, s.t_char, Xi0)
    try:
        tr = train_library_lbfgs(m, ds["fit"], s.t_char, Yfit.std(0), continuation=continuation)
        Xi = nd.to_phys(m.coef_z())
        err = float(np.linalg.norm(Xi - s.Xi_true) / np.linalg.norm(s.Xi_true))
        loss = float(tr["history"][-1][2]) if len(tr["history"]) else np.nan
        K = int(tr["K_final"])
    except Exception:
        err, loss, K = np.nan, np.nan, 0
    if not np.isfinite(err):
        err = np.nan
    return dict(system=name, dt_ratio=r, noise=eps, init_seed=init_seed, init_scale=scale,
                strategy="continuation" if continuation else "direct", coef_err_dense=err,
                final_loss=loss, K_final=K)
