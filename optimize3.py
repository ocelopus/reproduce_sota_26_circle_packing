"""Diverse multi-start + strong ILS for n=26, with large structural moves."""
import numpy as np
import time
import sys
from optimize2 import optimize, unpack, viol, N, PAIRS

rng_global = np.random.default_rng(0)


def hex_init(rng, rows_guess=5):
    # build hexagonal lattice covering square, take N points with jitter
    best = None
    for _ in range(3):
        pts = []
        dy = 1.0 / (rows_guess + 1)
        y = dy
        row = 0
        while y < 1.0 and len(pts) < N:
            dx = dy
            offset = (row % 2) * dx / 2
            x = dx / 2 + offset
            while x < 1.0 and len(pts) < N:
                pts.append((x, y))
                x += dx
            y += dy
            row += 1
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N, 0], pts[:N, 1]
    return rng.random(N), rng.random(N)


def poisson_init(rng):
    pts = []
    min_d = 0.12
    tries = 0
    while len(pts) < N and tries < 5000:
        tries += 1
        p = rng.random(2)
        if all(np.hypot(*(p - q)) > min_d for q in pts):
            pts.append(p)
    pts = np.array(pts)
    if len(pts) < N:
        while len(pts) < N:
            pts = np.vstack([pts, rng.random(2)])
    return pts[:N, 0], pts[:N, 1]


if __name__ == "__main__":
    budget = float(sys.argv[1])
    seed = int(sys.argv[2])
    outname = sys.argv[3]
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best_val = -1e9
    best_z = None
    it = 0
    while time.time() - t0 < budget:
        it += 1
        mode = it % 4
        if best_z is not None and mode in (0, 1):
            x, y, r = unpack(best_z)
            scale = rng.choice([0.003, 0.01, 0.03, 0.06, 0.12, 0.2])
            k = rng.integers(1, 8)
            idx = rng.choice(N, size=min(k, N), replace=False)
            x = x.copy(); y = y.copy()
            x[idx] += rng.normal(0, scale, len(idx))
            y[idx] += rng.normal(0, scale, len(idx))
        elif mode == 2:
            x, y = poisson_init(rng)
        else:
            x, y = hex_init(rng)
            x = x + rng.normal(0, 0.05, N)
            y = y + rng.normal(0, 0.05, N)
        x = np.clip(x, 1e-4, 1 - 1e-4); y = np.clip(y, 1e-4, 1 - 1e-4)
        z = optimize(x, y, np.full(N, 0.03), maxiter=500)
        if z is None:
            continue
        xx, yy, rr = unpack(z)
        val = rr.sum()
        if val > best_val:
            best_val = val
            best_z = z.copy()
            print(f"[{time.time()-t0:6.1f}s] s{seed} it={it} BEST sum={best_val:.8f}", flush=True)
            np.savez(outname, centers=np.stack([xx, yy], 1), radii=rr)
    print(f"FINAL s{seed} sum={best_val:.8f} iters={it}")
