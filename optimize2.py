"""Iterated local search for n=26 circle packing (maximize sum of radii)."""
import numpy as np
from scipy.optimize import minimize
import time
import sys
import os

N = 26
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]


def unpack(z):
    return z[0:N], z[N:2 * N], z[2 * N:3 * N]


def objective(z):
    return -np.sum(z[2 * N:3 * N])


def grad_objective(z):
    g = np.zeros_like(z)
    g[2 * N:3 * N] = -1.0
    return g


def make_cons():
    cons = []

    def ov(z, i, j):
        x, y, r = unpack(z)
        return np.hypot(x[i] - x[j], y[i] - y[j]) - r[i] - r[j]

    def ovj(z, i, j):
        x, y, r = unpack(z)
        dx = x[i] - x[j]; dy = y[i] - y[j]
        d = max(np.hypot(dx, dy), 1e-12)
        g = np.zeros_like(z)
        g[i] = dx / d; g[j] = -dx / d
        g[N + i] = dy / d; g[N + j] = -dy / d
        g[2 * N + i] = -1.0; g[2 * N + j] = -1.0
        return g

    for (i, j) in PAIRS:
        cons.append({'type': 'ineq', 'fun': lambda z, i=i, j=j: ov(z, i, j),
                     'jac': lambda z, i=i, j=j: ovj(z, i, j)})

    def bnd(z, i, w):
        x, y, r = unpack(z)
        return [x[i] - r[i], 1 - x[i] - r[i], y[i] - r[i], 1 - y[i] - r[i]][w]

    def bndj(z, i, w):
        g = np.zeros_like(z)
        if w == 0: g[i] = 1; g[2 * N + i] = -1
        elif w == 1: g[i] = -1; g[2 * N + i] = -1
        elif w == 2: g[N + i] = 1; g[2 * N + i] = -1
        else: g[N + i] = -1; g[2 * N + i] = -1
        return g

    for i in range(N):
        for w in range(4):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i, w=w: bnd(z, i, w),
                         'jac': lambda z, i=i, w=w: bndj(z, i, w)})
    return cons


CONS = make_cons()
BOUNDS = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(1e-9, 0.5)] * N


def viol(x, y, r):
    v = 0.0
    for (i, j) in PAIRS:
        v += max(0.0, r[i] + r[j] - np.hypot(x[i] - x[j], y[i] - y[j]))
    for i in range(N):
        v += max(0.0, r[i] - min(x[i], 1 - x[i], y[i], 1 - y[i]))
    return v


def optimize(x0, y0, r0, maxiter=400):
    x0 = np.clip(x0, 1e-4, 1 - 1e-4)
    y0 = np.clip(y0, 1e-4, 1 - 1e-4)
    z0 = np.concatenate([x0, y0, r0])
    try:
        res = minimize(objective, z0, jac=grad_objective, bounds=BOUNDS,
                       constraints=CONS, method='SLSQP',
                       options={'maxiter': maxiter, 'ftol': 1e-13})
    except Exception:
        return None
    x, y, r = unpack(res.x)
    if viol(x, y, r) > 1e-6:
        return None
    return res.x


def radii_from_centers_lp(x, y):
    """Cheap feasible radii: start small, no LP; just tiny."""
    return np.full(N, 0.01)


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    outname = sys.argv[3] if len(sys.argv) > 3 else f"best_seed{seed}.npz"
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best_val = -1e9
    best_z = None
    it = 0
    # initial pool
    while time.time() - t0 < budget:
        it += 1
        if best_z is not None and rng.random() < 0.6:
            # ILS: perturb best
            x, y, r = unpack(best_z)
            scale = rng.choice([0.01, 0.02, 0.04, 0.08])
            k = rng.integers(1, 5)
            idx = rng.choice(N, size=k, replace=False)
            x = x.copy(); y = y.copy()
            x[idx] += rng.normal(0, scale, k)
            y[idx] += rng.normal(0, scale, k)
            x = np.clip(x, 1e-4, 1 - 1e-4); y = np.clip(y, 1e-4, 1 - 1e-4)
            z = optimize(x, y, np.full(N, 0.02))
        else:
            # random restart
            x = rng.random(N); y = rng.random(N)
            z = optimize(x, y, np.full(N, 0.02))
        if z is None:
            continue
        x, y, r = unpack(z)
        val = np.sum(r)
        if val > best_val:
            best_val = val
            best_z = z.copy()
            print(f"[{time.time()-t0:6.1f}s] s{seed} it={it} NEW BEST sum={best_val:.7f} viol={viol(x,y,r):.2e}", flush=True)
            np.savez(outname, centers=np.stack([x, y], 1), radii=r)
    print(f"FINAL s{seed} sum={best_val:.7f} iters={it}")
