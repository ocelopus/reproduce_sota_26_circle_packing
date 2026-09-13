"""SLSQP candidate generation + STRICT-LP scoring (counterpart of refine.py).

refine.py's acceptance test used the raw SLSQP sum, which tolerates violations up
to 1e-6 and inflates the value by ~1e-7. Here every candidate is re-scored with
the strict LP (`strict_search.strict_sum`), so acceptance is based on a sum that
is genuinely achievable.

Usage: python strict_refine.py <budget_s> <seed> <init.npz> <out.npz>
"""
import numpy as np
import time
import sys
from optimize2 import optimize, unpack, N
from strict_search import strict_sum
from strict_lp import viol_abs

if __name__ == "__main__":
    budget = float(sys.argv[1])
    seed = int(sys.argv[2])
    init = sys.argv[3]
    out = sys.argv[4]
    rng = np.random.default_rng(seed)
    d = np.load(init)
    c = d['centers']
    best, best_r = strict_sum(c)
    best_c = c.copy()
    best_z = np.concatenate([c[:, 0], c[:, 1], d['radii']])
    print(f"s{seed} start strict={best:.12f}", flush=True)
    t0 = time.time()
    it = 0
    while time.time() - t0 < budget:
        it += 1
        bx, by, br = unpack(best_z)
        scale = rng.choice([5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2])
        k = int(rng.integers(1, 4))
        idx = rng.choice(N, size=k, replace=False)
        nx = bx.copy(); ny = by.copy()
        nx[idx] += rng.normal(0, scale, k)
        ny[idx] += rng.normal(0, scale, k)
        nx = np.clip(nx, 1e-5, 1 - 1e-5); ny = np.clip(ny, 1e-5, 1 - 1e-5)
        z = optimize(nx, ny, np.full(N, 0.03), maxiter=500)
        if z is None:
            continue
        xx, yy, rr = unpack(z)
        s, r = strict_sum(np.stack([xx, yy], 1))
        if s is None:
            continue
        if s > best:
            best, best_r, best_c, best_z = s, r, np.stack([xx, yy], 1), z.copy()
            print(f"[{time.time()-t0:6.1f}s] s{seed} it={it} strict={best:.12f} viol={viol_abs(best_c, best_r):+.1e}", flush=True)
            np.savez(out, centers=best_c, radii=best_r)
    print(f"FINAL s{seed} strict={best:.12f} iters={it}", flush=True)
    np.savez(out, centers=best_c, radii=best_r)
