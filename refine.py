"""Long intensive ILS refinement starting from a given solution."""
import numpy as np
import time
import sys
from optimize2 import optimize, unpack, viol, N

if __name__ == "__main__":
    budget = float(sys.argv[1])
    seed = int(sys.argv[2])
    init = sys.argv[3]
    outname = sys.argv[4]
    rng = np.random.default_rng(seed)
    d = np.load(init)
    c = d['centers']; r = d['radii']
    x, y = c[:, 0].copy(), c[:, 1].copy()
    best_val = r.sum()
    best_z = np.concatenate([x, y, r])
    t0 = time.time()
    it = 0
    accepted_since = 0
    while time.time() - t0 < budget:
        it += 1
        bx, by, br = unpack(best_z)
        scale = rng.choice([0.002, 0.005, 0.01, 0.02, 0.04])
        k = rng.integers(1, 4)
        idx = rng.choice(N, size=k, replace=False)
        nx = bx.copy(); ny = by.copy()
        nx[idx] += rng.normal(0, scale, k)
        ny[idx] += rng.normal(0, scale, k)
        nx = np.clip(nx, 1e-5, 1 - 1e-5); ny = np.clip(ny, 1e-5, 1 - 1e-5)
        z = optimize(nx, ny, np.full(N, 0.03), maxiter=500)
        if z is None:
            continue
        xx, yy, rr = unpack(z)
        val = rr.sum()
        if val > best_val:
            best_val = val
            best_z = z.copy()
            print(f"[{time.time()-t0:6.1f}s] s{seed} it={it} BEST sum={best_val:.8f} viol={viol(xx,yy,rr):.1e}", flush=True)
            np.savez(outname, centers=np.stack([xx, yy], 1), radii=rr)
    print(f"FINAL s{seed} sum={best_val:.8f} iters={it}")
