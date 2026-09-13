"""Strict-feasibility LP tools for circle packing.

The raw HiGHS LP value carries ~1e-9 of feasibility slop, so it cannot be used to
compare or accept improvements of that size. These helpers instead return radii
that satisfy every constraint, and the value that goes with them.
"""
import numpy as np
from scipy.optimize import linprog

N = 26
TIGHT = {'primal_feasibility_tolerance': 1e-10, 'dual_feasibility_tolerance': 1e-10}


def build(c):
    rows, b = [], []
    for i in range(N):
        for j in range(i + 1, N):
            row = np.zeros(N); row[i] = row[j] = 1.0
            rows.append(row); b.append(float(np.hypot(*(c[i] - c[j]))))
    for i in range(N):
        x, y = c[i]
        row = np.zeros(N); row[i] = 1.0
        rows.append(row); b.append(float(min(x, 1 - x, y, 1 - y)))
    return np.array(rows), np.array(b)


def sum_uniform(c, delta, opts=TIGHT):
    """max sum r with every constraint tightened by a uniform delta."""
    A, b = build(c)
    r = linprog(-np.ones(N), A_ub=A, b_ub=b - delta, bounds=[(0, None)] * N,
                method='highs', options=opts).x
    return float(r.sum()), r


def feasible_max(c, opts=TIGHT, max_iter=30, grow=1.05):
    """Tighten only violated constraints until the LP solution is feasible."""
    A, b = build(c)
    margins = np.zeros(len(b))
    hist = []
    for _ in range(max_iter):
        r = linprog(-np.ones(N), A_ub=A, b_ub=b - margins, bounds=[(0, None)] * N,
                    method='highs', options=opts).x
        v = A @ r - (b - margins)
        hist.append((float(r.sum()), float(v.max())))
        if v.max() <= 0.0:
            return float(r.sum()), r, margins, hist
        margins = margins + np.maximum(v, 0.0) * grow
    return None, None, margins, hist


def viol_abs(c, r):
    A, b = build(c)
    return float((A @ r - b).max())
