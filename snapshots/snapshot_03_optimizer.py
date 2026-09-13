# EVOLVE-BLOCK-START
"""Circle packing n=26 via nonlinear programming.

We maximize sum(r_i) over centers and radii subject to non-overlap and boundary
constraints, using SLSQP (analytic gradients) from many random/hexagonal restarts
plus iterated local search. A fixed RNG seed makes the result deterministic.
"""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]


def _unpack(z):
    return z[:N], z[N:2 * N], z[2 * N:3 * N]


def _obj(z):
    return -np.sum(z[2 * N:3 * N])


def _grad(z):
    g = np.zeros_like(z)
    g[2 * N:3 * N] = -1.0
    return g


def _constraints():
    cons = []

    def ov(z, i, j):
        x, y, r = _unpack(z)
        return np.hypot(x[i] - x[j], y[i] - y[j]) - r[i] - r[j]

    def ovj(z, i, j):
        x, y, r = _unpack(z)
        dx, dy = x[i] - x[j], y[i] - y[j]
        d = max(np.hypot(dx, dy), 1e-12)
        g = np.zeros_like(z)
        g[i], g[j] = dx / d, -dx / d
        g[N + i], g[N + j] = dy / d, -dy / d
        g[2 * N + i] = g[2 * N + j] = -1.0
        return g

    for (i, j) in PAIRS:
        cons.append({'type': 'ineq', 'fun': lambda z, i=i, j=j: ov(z, i, j),
                     'jac': lambda z, i=i, j=j: ovj(z, i, j)})

    def bnd(z, i, w):
        x, y, r = _unpack(z)
        return [x[i] - r[i], 1 - x[i] - r[i], y[i] - r[i], 1 - y[i] - r[i]][w]

    def bndj(z, i, w):
        g = np.zeros_like(z)
        if w == 0: g[i], g[2 * N + i] = 1, -1
        elif w == 1: g[i], g[2 * N + i] = -1, -1
        elif w == 2: g[N + i], g[2 * N + i] = 1, -1
        else: g[N + i], g[2 * N + i] = -1, -1
        return g

    for i in range(N):
        for w in range(4):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i, w=w: bnd(z, i, w),
                         'jac': lambda z, i=i, w=w: bndj(z, i, w)})
    return cons


_CONS = _constraints()
_BOUNDS = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(1e-9, 0.5)] * N


def _viol(x, y, r):
    v = 0.0
    for (i, j) in PAIRS:
        v += max(0.0, r[i] + r[j] - np.hypot(x[i] - x[j], y[i] - y[j]))
    for i in range(N):
        v += max(0.0, r[i] - min(x[i], 1 - x[i], y[i], 1 - y[i]))
    return v


def _solve(x0, y0, maxiter=500):
    x0 = np.clip(x0, 1e-4, 1 - 1e-4)
    y0 = np.clip(y0, 1e-4, 1 - 1e-4)
    z0 = np.concatenate([x0, y0, np.full(N, 0.03)])
    try:
        res = minimize(_obj, z0, jac=_grad, bounds=_BOUNDS, constraints=_CONS,
                       method='SLSQP', options={'maxiter': maxiter, 'ftol': 1e-13})
    except Exception:
        return None
    x, y, r = _unpack(res.x)
    if _viol(x, y, r) > 1e-6:
        return None
    return res.x


def _lp_radii(centers):
    n = centers.shape[0]
    rows, ub = [], []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n); row[i] = row[j] = 1.0
            rows.append(row); ub.append(np.hypot(*(centers[i] - centers[j])))
    for i in range(n):
        row = np.zeros(n); row[i] = 1.0
        rows.append(row)
        ub.append(min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1]))
    res = linprog(-np.ones(n), A_ub=np.array(rows), b_ub=np.array(ub),
                  bounds=[(0, None)] * n, method='highs')
    return res.x


def construct_packing():
    rng = np.random.default_rng(12345)
    budget = 90.0
    t0 = time.time()
    best_val, best_z = -1e9, None
    it = 0
    while time.time() - t0 < budget:
        it += 1
        if best_z is not None and it % 2 == 0:          # iterated local search
            x, y, r = _unpack(best_z)
            scale = rng.choice([0.005, 0.02, 0.05, 0.1, 0.2])
            k = int(rng.integers(1, 8))
            idx = rng.choice(N, size=min(k, N), replace=False)
            x = x.copy(); y = y.copy()
            x[idx] += rng.normal(0, scale, len(idx))
            y[idx] += rng.normal(0, scale, len(idx))
        else:
            m = 6
            pts = np.array([((i + 0.5) / m, (j + 0.5) / m)
                            for i in range(m) for j in range(m)])[:N]
            x = pts[:, 0] + rng.normal(0, 0.05, N)
            y = pts[:, 1] + rng.normal(0, 0.05, N)
        z = _solve(x, y)
        if z is None:
            continue
        _, _, r = _unpack(z)
        val = np.sum(r)
        if val > best_val:
            best_val, best_z = val, z.copy()
    x, y, r = _unpack(best_z)
    centers = np.stack([x, y], 1)
    radii = _lp_radii(centers)  # exact max radii for the found centers
    return centers, radii, float(np.sum(radii))


# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    _, _, s = run_packing()
    print(f"Sum of radii: {s}")
