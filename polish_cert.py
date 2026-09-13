"""Optimize centers against the CERTIFIABLE value.

The objective here is the value that can actually be shipped: the maximum sum of
radii subject to the (slightly conservative) constraint set, obtained from a
linear program whose units are scaled by 2**20 so HiGHS's absolute tolerances
(~1e-10) are negligible. Its numerical noise is ~1e-16, which is what allows the
last 1e-11 to be resolved at all -- the earlier searches optimised values that
carried ~1e-9 of solver slop, so improvements at this scale were invisible.

Usage: python polish_cert.py <secs> <seed> <init.npz> <out.npz>
"""
import sys
import time

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
SCALE = 2.0 ** 20
TIGHT = {'primal_feasibility_tolerance': 1e-10, 'dual_feasibility_tolerance': 1e-10}

# constraint matrix is constant: rows are (r_i + r_j) for pairs and (r_i) for walls
ROWS = []
for _i in range(N):
    for _j in range(_i + 1, N):
        v = np.zeros(N); v[_i] = v[_j] = 1.0
        ROWS.append(v)
for _i in range(N):
    v = np.zeros(N); v[_i] = 1.0
    ROWS.append(v)
A = np.array(ROWS)


def bvec(c):
    b = np.empty(len(A))
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            b[k] = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
            k += 1
    for i in range(N):
        x, y = c[i]
        b[k] = min(x, 1 - x, y, 1 - y)
        k += 1
    return b


def certifiable(c):
    """Highest sum that is achievable and verifiable for these centers."""
    res = linprog(-np.ones(N), A_ub=A, b_ub=bvec(c) * SCALE,
                  bounds=[(0, None)] * N, method='highs', options=TIGHT)
    if not res.success:
        return -np.inf
    return float(res.x.sum()) / SCALE


def neg(z):
    return -certifiable(z.reshape(N, 2))


if __name__ == '__main__':
    seconds, seed, init, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    rng = np.random.default_rng(seed)
    c0 = np.load(init)['centers'].copy()
    best = certifiable(c0)
    best_c = c0.copy()
    print(f's{seed} start certifiable={best:.16f}', flush=True)
    np.savez(out, centers=best_c)

    t0 = time.time()
    while time.time() - t0 < seconds:
        # restart from the incumbent with a random kick
        scale = rng.choice([1e-7, 5e-7, 2e-6, 1e-5, 5e-5, 2e-4])
        start = np.clip(best_c + rng.normal(0, scale, (N, 2)), 1e-8, 1 - 1e-8)
        res = minimize(neg, start.ravel(), method='L-BFGS-B',
                       options={'maxiter': 60, 'eps': 1e-7, 'ftol': 1e-18, 'gtol': 1e-14})
        cand = np.clip(res.x.reshape(N, 2), 1e-8, 1 - 1e-8)
        val = certifiable(cand)
        if val > best:
            best, best_c = val, cand
            print(f'[{time.time()-t0:6.1f}s] s{seed} certifiable={best:.16f}', flush=True)
            np.savez(out, centers=best_c)
    print(f'FINAL s{seed} certifiable={best:.16f}', flush=True)
    np.savez(out, centers=best_c)
