# EVOLVE-BLOCK-START
"""Circle packing n=26: certified near-optimal configuration.

    sum of radii = 2.6359830849014761     (exactly verified, see EXPERIMENT_LOG.md)

This exceeds the previous best report on the OpenEvolve issue tracker, which
quoted 2.635983084893607 (github.com/algorithmicsuperintelligence/openevolve/
issues/156#issuecomment-3419750877).

Provenance
----------
This file is a MODIFIED derivative of `examples/circle_packing/initial_program.py`
from OpenEvolve, https://github.com/algorithmicsuperintelligence/openevolve
(Apache License 2.0). Only the packing constructor was replaced by the solution
below; the module layout, `run_packing()` and `visualize()` keep the upstream
shape. Notice of modification as required by Apache-2.0 section 4(b).

Why this configuration
----------------------
At the optimum the LP that maximises the radii has an integral dual of value 1
supported on 17 constraints that cover all 26 circles exactly once:

    9 touching pairs          r_i + r_j = dist(i, j)
    8 circles on a wall       r_k       = min(x, 1-x, y, 1-y)

Because the 17 constraints are disjoint and cover every circle, the optimum is
the sum of 17 independent terms, and each pair may split its distance freely
between its two circles (the LP is 9-fold degenerate). The radii below are one
valid split, chosen so that every other constraint stays strictly slack.

Numerics
--------
The radii were produced with exact arithmetic instead of trusting a solver:

  * the 351 pair distances and the 26 wall clearances were computed at 80 digits
    and rounded DOWN onto a rational grid, so every constraint used is
    conservative with respect to the true geometry;
  * the split was found with a linear program rescaled by a power of two, which
    makes HiGHS's absolute feasibility tolerance (~1e-10) negligible;
  * every radius was rounded DOWN onto the rational grid, and the result was
    verified against all 377 constraints in exact Fraction arithmetic -- zero
    violations.

The radii below are the float64 image of that certificate, rounded down, so the
packing is strictly feasible and does not rely on the evaluator's 1e-6 tolerance.
"""
import numpy as np

# Verified configuration: 9 touching pairs + 8 wall circles.
CENTERS = np.array([
    [0.09573232930627384, 0.6832585349730643],
    [0.2946094887827298, 0.13022110106360607],
    [0.49866807550269454, 0.5299634197528075],
    [0.8888438205897514, 0.8888438205906881],
    [0.11077901279073225, 0.8892209872091789],
    [0.6859430219859982, 0.907407905047854],
    [0.5952197329394243, 0.7420494434608222],
    [0.29474605905021134, 0.3869235534105264],
    [0.7023095250900068, 0.13325857277098946],
    [0.08463950069468397, 0.08463950069580058],
    [0.8965327666423123, 0.4825955822109562],
    [0.8948174397310787, 0.2739528396244337],
    [0.49728444620367784, 0.07886037291549626],
    [0.27162985148647084, 0.5976347963892725],
    [0.10679014463035094, 0.2747832833514745],
    [0.4955317606731277, 0.2753426167718285],
    [0.9038486659544825, 0.682080042931806],
    [0.10306052014169007, 0.484600802650186],
    [0.4994283691393546, 0.9060726627224037],
    [0.915073737545795, 0.08492626245460033],
    [0.31311580997086363, 0.9076084484290329],
    [0.23971052792717185, 0.7636735693832257],
    [0.7269057143108829, 0.5960427019079275],
    [0.7593524015606512, 0.762958863654165],
    [0.7026096036991122, 0.3816658444529344],
    [0.40335878360283905, 0.7424170495018221],
])

# Strictly feasible radii: every constraint has a positive margin.
RADII = np.array([
    0.09573232930627383,
    0.13022110106360604,
    0.1370104301231409,
    0.11115617940931187,
    0.11077901279073224,
    0.0925920949502383,
    0.09601897575762984,
    0.11207708894849873,
    0.13325857277098943,
    0.08463950069468396,
    0.10346723335768769,
    0.10518256026792736,
    0.07886037291549625,
    0.09989835059142549,
    0.10679014463004077,
    0.11762968804638212,
    0.09615133404483144,
    0.10306052013776879,
    0.09392733727742257,
    0.08492626245420497,
    0.09239155157083438,
    0.06918067635698327,
    0.10060036781851757,
    0.06944019371190192,
    0.11514888015963375,
    0.09584232574531266,
])


def construct_packing():
    """Return the certified packing (centers, radii, sum of radii)."""
    centers = CENTERS.copy()
    radii = RADII.copy()
    return centers, radii, float(np.sum(radii))


# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


def visualize(centers, radii):
    """Visualize the circle packing (requires matplotlib)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)
    for i, (center, radius) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(center, radius, alpha=0.5))
        ax.text(center[0], center[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.savefig("packing.png", dpi=120)
    print("saved packing.png")


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii!r}")
    print(f"Target ratio (vs 2.635): {sum_radii / 2.635:.10f}")

    worst_pair, worst_bnd = -1e9, -1e9
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.hypot(*(centers[i] - centers[j])))
            worst_pair = max(worst_pair, radii[i] + radii[j] - d)
        x, y = centers[i]
        worst_bnd = max(worst_bnd, radii[i] - min(x, 1 - x, y, 1 - y))
    print(f"max overlap     : {worst_pair:+.3e}  (must be <= 0)")
    print(f"max wall excess : {worst_bnd:+.3e}  (must be <= 0)")
