# Circle Packing (n=26) — Experiment Log

**Goal.** Maximize the sum of radii of 26 non-overlapping circles inside the unit
square, as scored by `evaluator.py` (`combined_score = sum_radii / 2.635 * validity`).

---

## Result

| | sum of radii | feasible? |
|---|---|---|
| AlphaEvolve (the `2.635` target) | `2.6358627562135460` | reference value baked into `evaluator.py` |
| Previous report (`pudepiedj`) | `2.635983084893607` | not verified — printed by a program |
| Best known published (`Numaro`, tied with ThetaEvolve) | `2.6359830853` | reported as feasibility-checked |
| **This work (`initial_program.py`)** | **`2.6359830849014761`** | **yes — exact check, 0 violations** |

This **reproduces** the best-known value; it does not beat it. It sits `+7.9e-12`
above `pudepiedj`'s figure and `4.0e-10` below Numaro's.

Official evaluator on the final program:

```
Evaluation: valid=True, sum_radii=2.635983, target=2.635, ratio=1.000373, time=0.17s
RESULT {'sum_radii': 2.635983084901476, 'target_ratio': 1.0003730872491372,
        'validity': 1.0, 'eval_time': 0.17, 'combined_score': 1.0003730872491372}
```

`target_ratio = 1.0003730872`. The packing is *strictly* feasible — the largest
overlap is `-2.8e-17` — so it does not depend on the evaluator's `1e-6` tolerance.

---

## Rescuing the result: what was actually wrong

The program shipped by the YOLO agent returns `2.635982986575511`, not the
`2.635983` claimed in the README. Two independent numerical faults were found.

### Fault 1 — the search accepted infeasible solutions (≈9.9e-9)

`refine.py`/`final_search.py` accepted a candidate whenever the *raw* SLSQP sum
increased. SLSQP treats a constraint as satisfied within `ftol`, so its reported
sums exceed what the radii can actually support. The saved
`experiments/rf_303.npz` quotes `2.6359831781827268`, but those radii overlap by
`9.9e-9`:

```
experiments/rf_303.npz    saved_sum=2.6359831782  saved_viol=9.94e-09
```

The same applies to `best_final.npz`'s "exact LP" number: the raw HiGHS value
`2.6359830809` is inflated, and once the LP solution is made genuinely feasible it
drops to `2.6359830666`. **All improvements at the 1e-8 level reported by the
original search were numerical artefacts.**

### Fault 2 — the shipped program threw away 9.4e-8 (the real bug)

`initial_program.py` recovered the radii with

```python
scale = min(scale, dist(i, j) / (r[i] + r[j]))   # over ALL pairs
return radii * scale * (1 - 1e-12)
```

The factor is dominated by the pair with the worst *relative* rounding error —
a short contact between two small circles (`pair 5-23`, distance `0.162`), which
lost `3.6e-8` of relative precision for no reason:

```
program scale factor = 0.999999964219   ->  shrink loss = 9.432e-08
```

A **uniform** factor is the wrong tool: slack is per-constraint, so a tiny
numerical excess on one short contact must not rescale the whole packing. This is
the entire gap between the shipped `2.6359829866` and the `2.63598308` that the
centres actually support.

### Why the "better" number could not be trusted

`polish.py` maximised the raw LP value, whose solver slop (`~1e-9`) is the same
order as the improvements being hunted, so no verdict at the 1e-9 level is
possible from such values. Scoring candidates by the **strictly feasible** LP
optimum (`strict_search.py`, `strict_refine.py`) is what finally produced a real
improvement:

```
best_final.npz          strict = 2.635983066630
strict_refine (5 min)   strict = 2.635983083695   (+1.7e-8, zero violation)
```

---

## Structure of the optimum (why the final construction is exact)

At the optimum the radii LP has an **integral dual of value 1** supported on 17
constraints that cover all 26 circles exactly once:

```
9 touching pairs : (0,21) (1,15) (2,22) (5,6) (7,14) (11,24) (13,17) (16,23) (18,25)
8 wall circles   : 3 4 8 9 10 12 19 20
```

Consequences:

1. The optimum is a **sum of 17 independent terms**, so it needs no LP solver:
   `sum = Σ dist(i,j) + Σ clearance(k)` — the "pattern value".
2. The LP is **9-fold degenerate**: each pair may split its distance freely
   between its two circles. The splits are chosen to keep every other constraint
   strictly slack.
3. Because the 17 terms are disjoint, `sum r` cannot exceed the pattern value.

The exact pattern value of these centres is `2.6359830849151177`; the shipped,
verifiable value is `2.6359830849014761` — the difference (`1.4e-11`) is the part
of the pattern bound that the remaining cross-constraints do not permit.

## Exact certificate

The shipped radii were **not** obtained by trusting a solver. `build_final2.py`:

1. computes all 351 pair distances and 26 wall clearances at 80 digits and
   **rounds them down** onto a `2^-70` rational grid, so each constraint used is
   conservative with respect to the true geometry;
2. solves the split LP in units scaled by `2^20`, which makes HiGHS's absolute
   feasibility tolerance (`~1e-10`) negligible (`~1e-16` in true units);
3. rounds every radius **down** onto the rational grid (only ever helpful);
4. removes residue in exact arithmetic and verifies all 377 constraints with
   `Fraction`s: **0 violations**;
5. converts to float64 rounding **down**, so the shipped doubles are feasible too.

```
exact feasibility : 0 violations
EXACT sum of radii = 2.6359830849014761481173
float64 sum       = 2.6359830849014760
previous best     = 2.635983084893607
improvement       = +7.869e-12
```

---

## Files

| file | role |
|---|---|
| `initial_program.py` | final program — certified centres + radii, `sum = 2.6359830849014761` |
| `best_certified.npz` | certified centres/radii (`best_lp.npz` is the old, inferior one) |
| `build_final2.py` | builds the exact certificate (steps 1–5 above) |
| `strict_lp.py` | strict-feasibility LP helpers (no solver slop) |
| `strict_search.py`, `strict_refine.py` | searches scored by the strict LP optimum |
| `polish_cert.py` | centre-space polish against the *certifiable* value |
| `snapshots/snapshot_04_shipped_by_agent.py` | the original program, kept for reference |
| `run_eval.py` | local evaluator driver (patches `sys.executable` only) |

Historical search scripts (`optimize*.py`, `refine.py`, `final_search.py`,
`polish.py`) are kept but **their 1e-8-level claims are unreliable** for the
reasons above.

## Sources and credits

- **Evaluator and initial program** — `evaluator.py` is byte-identical to, and
  `initial_program.py` is a modified derivative of, `examples/circle_packing/` in
  [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)
  (Apache-2.0). Modification notice is in `initial_program.py`'s docstring.
- **Benchmark lineage** — the problem and the `2.635` target come from
  AlphaEvolve's `mathematical_results.ipynb` § B.12
  ([google-deepmind/alphaevolve_results](https://github.com/google-deepmind/alphaevolve_results),
  Apache-2.0).
- **Agent harness** — `oai_compat_chat.py`, `harness_tools.py`,
  `harness_tools_fixes.py` and `loop.py` wrap
  [LangChain](https://github.com/langchain-ai/langchain) and
  [deepagents](https://github.com/langchain-ai/deepagents) (MIT, © LangChain,
  Inc.); no upstream source was copied.
- **Prior results compared against** — `2.635977394756627` (ypwang61),
  `2.635983` (ArnaudDeza) and `2.635983084893607` (pudepiedj), all comments on
  [OpenEvolve issue #156](https://github.com/algorithmicsuperintelligence/openevolve/issues/156).
  The last is a number printed by an evolved program, not a certified value.
- **Best known published value** — `2.6359830853` for n=26, Numaro AI
  Autoresearch Team, *Circle packing in the unit square: new sum-of-radii
  layouts*, report NUMARO-2026-004 (2026-07-03),
  <https://numaro.tech/research/circle-packing-unit-square-2026/>; the same value
  is attributed there to ThetaEvolve. Numaro pins its baselines to Erich
  Friedman's *Circles in Squares* table.

See the licensing discussion in `README.md` before publishing.

## Reproduction

```bash
python3 initial_program.py     # prints the sum and the feasibility margins
python3 run_eval.py initial_program.py
python3 build_final2.py best_certified.npz /tmp/rebuilt.npz   # rebuild the certificate
```
