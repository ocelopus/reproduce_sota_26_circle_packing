"""Final wide search: ILS from a seed solution (all scales) + fresh restarts."""
import numpy as np
import time
import sys
from optimize2 import optimize, unpack, viol, N
from optimize3 import poisson_init, hex_init

if __name__ == "__main__":
    budget = float(sys.argv[1])
    seed = int(sys.argv[2])
    init = sys.argv[3]
    outname = sys.argv[4]
    rng = np.random.default_rng(seed)
    d = np.load(init)
    c = d['centers']; r = d['radii']
    best_z = np.concatenate([c[:, 0], c[:, 1], r])
    best_val = r.sum()
    t0 = time.time()
    it = 0
    scales = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.25]
    while time.time() - t0 < budget:
        it += 1
        mode = it % 5
        if mode == 4:
            x, y = poisson_init(rng)
        elif mode == 3:
            x, y = hex_init(rng)
            x += rng.normal(0, 0.06, N); y += rng.normal(0, 0.06, N)
        else:
            bx, by, br = unpack(best_z)
            scale = rng.choice(scales)
            k = int(rng.integers(1, 9))
            idx = rng.choice(N, size=min(k, N), replace=False)
            x = bx.copy(); y = by.copy()
            x[idx] += rng.normal(0, scale, len(idx))
            y[idx] += rng.normal(0, scale, len(idx))
        x = np.clip(x, 1e-4, 1 - 1e-4); y = np.clip(y, 1e-4, 1 - 1e-4)
        z = optimize(x, y, np.full(N, 0.03), maxiter=600)
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
