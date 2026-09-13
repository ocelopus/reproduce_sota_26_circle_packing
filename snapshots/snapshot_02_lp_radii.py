# EVOLVE-BLOCK-START
"""Circle packing n=26: same structured centers as the baseline, but radii are
computed as the *exact* LP optimum (maximize sum r_i subject to non-overlap and
boundary constraints) instead of the baseline's greedy pairwise rescaling, which
under-fills the solution."""
import numpy as np
from scipy.optimize import linprog


def construct_packing():
    n = 26
    centers = np.zeros((n, 2))
    centers[0] = [0.5, 0.5]
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]
    centers = np.clip(centers, 0.01, 0.99)
    radii = compute_max_radii(centers)
    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    """Exact LP: maximize sum(r) s.t. r_i+r_j<=d_ij and r_i<=boundary_i."""
    n = centers.shape[0]
    b = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1 - centers[:, 1]))
    rows, ub = [], []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = row[j] = 1.0
            rows.append(row)
            ub.append(np.hypot(*(centers[i] - centers[j])))
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        ub.append(b[i])
    res = linprog(-np.ones(n), A_ub=np.array(rows), b_ub=np.array(ub),
                  bounds=[(0, None)] * n, method="highs")
    return res.x if res.success else np.zeros(n)


# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    _, _, s = run_packing()
    print(f"Sum of radii: {s}")
