"""Ship-ready packing with an exact rational certificate.

Method
------
All constraint right-hand sides (the 351 pair distances and the 26 wall
clearances) are computed in 80-digit arithmetic and rounded DOWN onto a rational
grid, so every constraint used here is conservative with respect to the true
geometry: anything feasible for these numbers is feasible for the real packing.

The radii come from a linear program whose units are scaled by a power of two.
HiGHS's feasibility tolerance is absolute (~1e-10), so scaling by 2**20 shrinks
its effect in the true units to ~1e-16, well below anything that matters.
Finally every radius is rounded DOWN onto the rational grid, which can only
reduce constraint violations.

The result is verified against all 377 constraints in exact Fraction arithmetic,
so the quoted sum is a rigorous statement about the numbers being shipped.

Usage: python build_final2.py <config.npz> <out.npz>
"""
import sys
from fractions import Fraction as F

import numpy as np
import mpmath as mp
from scipy.optimize import linprog

mp.mp.dps = 80
GRID = 2 ** 70
SCALE = 2.0 ** 20
N = 26
TIGHT = {'primal_feasibility_tolerance': 1e-10, 'dual_feasibility_tolerance': 1e-10}


def round_down_grid(x, grid=GRID):
    return F(int(mp.floor(mp.mpf(x) * grid)), grid)


def floor_grid(fr, grid=GRID):
    """Floor a Fraction onto the 1/grid lattice."""
    return F(fr.numerator * grid // fr.denominator, grid)


def fdown(fr):
    return np.nextafter(float(fr), -np.inf)


def to_float_down(fr):
    """Nearest double that is <= fr, so the shipped values stay feasible."""
    f = float(fr)
    return np.nextafter(f, -np.inf) if F(f) > fr else f


def exact_geometry(c):
    """Conservative (rounded-down) distances and wall clearances as rationals."""
    dist, w = {}, {}
    for i in range(N):
        for j in range(i + 1, N):
            dx = mp.mpf(float(c[i, 0])) - mp.mpf(float(c[j, 0]))
            dy = mp.mpf(float(c[i, 1])) - mp.mpf(float(c[j, 1]))
            dist[(i, j)] = round_down_grid(mp.sqrt(dx * dx + dy * dy))
        xi, yi = mp.mpf(float(c[i, 0])), mp.mpf(float(c[i, 1]))
        w[i] = round_down_grid(min(xi, 1 - xi, yi, 1 - yi))
    return dist, w


def pattern(c):
    """Optimal dual support: the matching of touching pairs and wall circles."""
    rows, b, lab = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            row = np.zeros(N); row[i] = row[j] = 1.0
            rows.append(row); b.append(float(np.hypot(*(c[i] - c[j]))))
            lab.append(('pair', i, j))
    for i in range(N):
        x, y = c[i]
        row = np.zeros(N); row[i] = 1.0
        rows.append(row); b.append(float(min(x, 1 - x, y, 1 - y)))
        lab.append(('bnd', i, -1))
    A, b = np.array(rows), np.array(b)
    du = linprog(b, A_ub=-A.T, b_ub=-np.ones(N), bounds=[(0, None)] * len(b),
                 method='highs', options=TIGHT)
    return [lab[k] for k in np.where(du.x > 1e-8)[0]]


def check_exact(dist, w, r):
    worst, nviol, loc = F(0), 0, ''
    for (i, j), d in dist.items():
        v = r[i] + r[j] - d
        if v > 0:
            nviol += 1
            if v > worst:
                worst, loc = v, f'pair {i}-{j}'
    for i in range(N):
        v = r[i] - w[i]
        if v > 0:
            nviol += 1
            if v > worst:
                worst, loc = v, f'bnd {i}'
    return nviol, worst, loc


def exact_repair(dist, w, r, rounds=8):
    """Remove any residual violation in exact arithmetic (cost ~1e-17)."""
    tiny = F(1, 10 ** 30)
    r = list(r)
    for _ in range(rounds):
        nviol, _, _ = check_exact(dist, w, r)
        if nviol == 0:
            break
        for (i, j), d in dist.items():
            v = r[i] + r[j] - d
            if v > 0:
                half = v / 2 + tiny
                r[i] = floor_grid(r[i] - half)
                r[j] = floor_grid(r[j] - half)
        for i in range(N):
            v = r[i] - w[i]
            if v > 0:
                r[i] = floor_grid(r[i] - v - tiny)
    return r


def dec(fr, digits=22):
    sign = '-' if fr < 0 else ''
    fr = abs(fr)
    ip = fr.numerator // fr.denominator
    rem = fr - ip
    ds = ''
    for _ in range(digits):
        rem *= 10
        d = rem.numerator // rem.denominator
        ds += str(d)
        rem -= d
    return f'{sign}{ip}.{ds}'


if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]
    c = np.load(src)['centers']
    dist, w = exact_geometry(c)
    P = pattern(c)
    pairs = [(i, j) for k, i, j in P if k == 'pair']
    walls = [i for k, i, _ in P if k == 'bnd']
    print(f'pattern: {len(pairs)} touching pairs + {len(walls)} wall circles '
          f'(covers {2 * len(pairs) + len(walls)}/26 circles)')

    rows, rhs = [], []
    for i in range(N):
        for j in range(i + 1, N):
            v = np.zeros(N); v[i] = v[j] = 1.0
            rows.append(v); rhs.append(fdown(dist[(i, j)]) * SCALE)
    for i in range(N):
        v = np.zeros(N); v[i] = 1.0
        rows.append(v); rhs.append(fdown(w[i]) * SCALE)

    res = linprog(-np.ones(N), A_ub=np.array(rows), b_ub=np.array(rhs),
                  bounds=[(0, None)] * N, method='highs', options=TIGHT)
    if not res.success:
        raise SystemExit(f'LP failed: {res.message}')
    r_lp = res.x / SCALE
    r_rat = exact_repair(dist, w, [round_down_grid(v) for v in r_lp])

    nviol, worst, loc = check_exact(dist, w, r_rat)
    total = sum(r_rat, F(0))
    fl = np.array([to_float_down(v) for v in r_rat])
    nviol_f, worst_f, loc_f = check_exact(dist, w, [F(float(v)) for v in fl])

    print(f'exact feasibility : {nviol} violations (worst {float(worst):.2e} at {loc})')
    print()
    print(f'EXACT sum of radii = {dec(total)}')
    print(f'  float64 sum      = {fl.sum():.16f}')
    print(f'  previous SOTA    = 2.635983084893607')
    print(f'  improvement      = {float(total) - 2.635983084893607:+.3e}')
    print(f'  target_ratio     = {float(total) / 2.635:.16f}')
    print(f'  shipped float64  : {nviol_f} violations ({loc_f})')
    np.savez(out, centers=c, radii=fl)
    print(f'  saved -> {out}')
