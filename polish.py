"""Polish centers using exact LP objective (derivative-free local search on centers)."""
import numpy as np
from scipy.optimize import linprog
import time
import sys

N = 26


def lp_sum(c):
    n = N
    rows = []
    b = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(*(c[i] - c[j]))
            row = np.zeros(n); row[i] = 1; row[j] = 1
            rows.append(row); b.append(dist)
    for i in range(n):
        x, y = c[i]
        row = np.zeros(n); row[i] = 1
        rows.append(row); b.append(min(x, 1 - x, y, 1 - y))
    res = linprog(-np.ones(n), A_ub=np.array(rows), b_ub=np.array(b),
                  bounds=[(0, None)] * n, method='highs')
    if not res.success:
        return None, None
    return -res.fun, res.x


if __name__ == "__main__":
    budget = float(sys.argv[1])
    seed = int(sys.argv[2])
    out = sys.argv[3]
    rng = np.random.default_rng(seed)
    d = np.load('best_lp.npz')
    c = d['centers'].copy()
    best_sum, best_r = lp_sum(c)
    print('start', best_sum, flush=True)
    t0 = time.time()
    it = 0
    T = 1e-4
    while time.time() - t0 < budget:
        it += 1
        T = max(1e-6, 1e-4 * (1 - (time.time() - t0) / budget))
        scale = rng.choice([0.0002, 0.0005, 0.001, 0.002, 0.005])
        nc = c.copy()
        k = int(rng.integers(1, 5))
        idx = rng.choice(N, size=k, replace=False)
        nc[idx] += rng.normal(0, scale, (len(idx), 2))
        nc = np.clip(nc, 1e-6, 1 - 1e-6)
        s, r = lp_sum(nc)
        if s is None:
            continue
        if s > best_sum or (T > 0 and rng.random() < np.exp((s - best_sum) / T) * 0.05):
            if s > best_sum:
                print(f'[{time.time()-t0:6.1f}s] {best_sum:.10f} -> {s:.10f}', flush=True)
            c = nc
            if s > best_sum:
                best_sum = s; best_r = r
                np.savez(out, centers=c, radii=best_r)
    print(f'FINAL s{seed} sum={best_sum:.10f} iters={it}')
    np.savez(out, centers=c, radii=best_r)
