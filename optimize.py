"""Offline optimizer for the n=26 equal-circle sum-of-radii packing problem.

We solve, over centers (x_i,y_i) and radii r_i:
    maximize  sum_i r_i
    s.t.      r_i + r_j <= dist(center_i, center_j)   for all i<j
              r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
              r_i >= 0
with scipy SLSQP (analytic jacobians) from many random / structured restarts.
"""
import numpy as np
from scipy.optimize import minimize
import time
import sys

N = 26


def unpack(z):
    x = z[0:N]
    y = z[N:2 * N]
    r = z[2 * N:3 * N]
    return x, y, r


def objective(z):
    return -np.sum(z[2 * N:3 * N])


def grad_objective(z):
    g = np.zeros_like(z)
    g[2 * N:3 * N] = -1.0
    return g


# Build constraint list once (index pairs)
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]


def make_constraints():
    cons = []

    def overlap(z, i, j):
        x, y, r = unpack(z)
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        return np.sqrt(dx * dx + dy * dy) - r[i] - r[j]

    def overlap_jac(z, i, j):
        x, y, r = unpack(z)
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        d = max(d, 1e-12)
        g = np.zeros_like(z)
        g[i] = dx / d
        g[j] = -dx / d
        g[N + i] = dy / d
        g[N + j] = -dy / d
        g[2 * N + i] = -1.0
        g[2 * N + j] = -1.0
        return g

    for (i, j) in PAIRS:
        cons.append({'type': 'ineq',
                     'fun': lambda z, i=i, j=j: overlap(z, i, j),
                     'jac': lambda z, i=i, j=j: overlap_jac(z, i, j)})

    def bnd(z, i, which):
        x, y, r = unpack(z)
        if which == 0:
            return x[i] - r[i]
        if which == 1:
            return 1.0 - x[i] - r[i]
        if which == 2:
            return y[i] - r[i]
        return 1.0 - y[i] - r[i]

    def bnd_jac(z, i, which):
        g = np.zeros_like(z)
        if which == 0:
            g[i] = 1.0; g[2 * N + i] = -1.0
        elif which == 1:
            g[i] = -1.0; g[2 * N + i] = -1.0
        elif which == 2:
            g[N + i] = 1.0; g[2 * N + i] = -1.0
        else:
            g[N + i] = -1.0; g[2 * N + i] = -1.0
        return g

    for i in range(N):
        for w in range(4):
            cons.append({'type': 'ineq',
                         'fun': lambda z, i=i, w=w: bnd(z, i, w),
                         'jac': lambda z, i=i, w=w: bnd_jac(z, i, w)})
    return cons


CONS = make_constraints()


def solve_from(x0, y0, maxiter=300):
    """Given initial centers, build a feasible-ish start (radii from LP-ish) and optimize."""
    x0 = np.clip(np.asarray(x0, float), 1e-4, 1 - 1e-4)
    y0 = np.clip(np.asarray(y0, float), 1e-4, 1 - 1e-4)
    # initial radii: small, shrink until feasible
    r0 = np.full(N, 0.02)
    z0 = np.concatenate([x0, y0, r0])
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, 0.5)] * N
    try:
        res = minimize(objective, z0, jac=grad_objective, bounds=bounds,
                       constraints=CONS, method='SLSQP',
                       options={'maxiter': maxiter, 'ftol': 1e-12})
    except Exception as e:
        return -1e9, None
    x, y, r = unpack(res.x)
    val = np.sum(r)
    # feasibility check
    if not feasible(x, y, r):
        return val - 100, res.x
    return val, res.x


def feasibility_violation(x, y, r):
    v = 0.0
    for (i, j) in PAIRS:
        d = np.hypot(x[i] - x[j], y[i] - y[j])
        v += max(0.0, r[i] + r[j] - d)
    for i in range(N):
        v += max(0.0, r[i] - min(x[i], 1 - x[i], y[i], 1 - y[i]))
    return v


def feasible(x, y, r, tol=1e-6):
    return feasibility_violation(x, y, r) < tol


def random_start(rng):
    return rng.random(N), rng.random(N)


def grid_start(k):
    # hexagonal-ish structured restarts
    pts = []
    m = int(np.ceil(np.sqrt(N)))
    for i in range(m):
        for j in range(m):
            pts.append(((i + 0.5) / m, (j + 0.5) / m))
    pts = np.array(pts[:N])
    return pts[:, 0], pts[:, 1]


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best_val = -1e9
    best_z = None
    it = 0
    while time.time() - t0 < budget:
        it += 1
        if it % 5 == 0:
            x0, y0 = grid_start(it)
            x0 = x0 + rng.normal(0, 0.02, N)
            y0 = y0 + rng.normal(0, 0.02, N)
        else:
            x0, y0 = random_start(rng)
        val, z = solve_from(x0, y0)
        if z is not None and val > best_val:
            best_val = val
            best_z = z.copy()
            x, y, r = unpack(best_z)
            print(f"[{time.time()-t0:6.1f}s] it={it} NEW BEST sum={best_val:.6f} viol={feasibility_violation(x,y,r):.2e}", flush=True)
    x, y, r = unpack(best_z)
    print(f"FINAL sum={np.sum(r):.6f} viol={feasibility_violation(x,y,r):.2e} iters={it}")
    np.savez(f"best_seed{seed}.npz", centers=np.stack([x, y], 1), radii=r)
