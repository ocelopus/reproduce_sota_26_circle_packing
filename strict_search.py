"""Center-space ILS scored by a STRICT-feasibility LP objective.

Motivation: `polish.py` maximised the raw HiGHS LP value, which carries ~6e-9 of
solver-tolerance slop (the returned radii violate the constraints). Both the
objective noise and the sought improvements are ~1e-8, so improvements found that
way are not reproducible. Here the objective is the *largest sum of radii that is
strictly feasible* (constraints tightened iteratively until the LP solution has no
violation), which is deterministic and free of slop.

Usage: python strict_search.py <budget_s> <seed> <init.npz> <out.npz>
"""
import numpy as np
import time
import sys
from strict_lp import build, viol_abs, N, TIGHT
from scipy.optimize import linprog


def strict_sum(c, max_iter=15):
    """Max sum radii with (iteratively tightened) constraints, guaranteed feasible."""
    c = np.clip(c, 1e-7, 1 - 1e-7)
    A, b = build(c)
    margins = np.zeros(len(b))
    r = None
    for _ in range(max_iter):
        r = linprog(-np.ones(N), A_ub=A, b_ub=b - margins, bounds=[(0, None)] * N,
                    method='highs', options=TIGHT).x
        v = A @ r - (b - margins)
        if v.max() <= 0.0:
            return float(r.sum()), r
        margins = margins + np.maximum(v, 0.0) * 1.05
    return None, None


if __name__ == "__main__":
    budget = float(sys.argv[1])
    seed = int(sys.argv[2])
    init = sys.argv[3]
    out = sys.argv[4]
    rng = np.random.default_rng(seed)
    c = np.load(init)['centers'].copy()
    best, best_r = strict_sum(c)
    print(f"s{seed} start strict_sum={best:.12f}", flush=True)
    t0 = time.time()
    it = 0
    while time.time() - t0 < budget:
        it += 1
        nc = c.copy()
        scale = rng.choice([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
        k = int(rng.integers(1, 6))
        idx = rng.choice(N, size=k, replace=False)
        nc[idx] += rng.normal(0, scale, (k, 2))
        s, r = strict_sum(nc)
        if s is None:
            continue
        if s > best:
            best, best_r, c = s, r, nc
            print(f"[{time.time()-t0:6.1f}s] s{seed} it={it} strict={best:.12f} viol={viol_abs(c, best_r):+.1e}", flush=True)
            np.savez(out, centers=c, radii=best_r)
    print(f"FINAL s{seed} strict={best:.12f} iters={it}", flush=True)
    np.savez(out, centers=c, radii=best_r)
