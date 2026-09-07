# EVOLVE-BLOCK-START
"""
Advanced 26‑circle packing.

We combine a diverse pool of feasible starts with a global
basinhopping search that repeatedly invokes a fairly long SLSQP
local optimiser.  The global search helps escape local minima,
while the SLSQP stage exploits gradient information to push the
solution toward a (locally) optimal sum of radii.

Key ideas
---------
* Greedy max‑min placement gives a strong deterministic start.
* Several perturbed and random starts increase diversity.
* For each start we run a basinhopping loop (time‑bounded) that:
    – Perturbs the full variable vector (centres + radii).
    – Calls SLSQP (maxiter = 1500) to locally maximise Σr.
* The best feasible solution found before the overall time budget
  expires is returned.
"""

import time
import numpy as np
from scipy.optimize import minimize, basinhopping

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _greedy_initial(n=26, step=0.02):
    """Deterministic greedy max‑min placement on a regular grid."""
    xs = np.arange(step / 2, 1.0, step)
    ys = np.arange(step / 2, 1.0, step)
    xv, yv = np.meshgrid(xs, ys)
    cand = np.column_stack([xv.ravel(), yv.ravel()])

    # distance to the square border for each candidate point
    border = np.minimum.reduce([cand[:, 0], cand[:, 1],
                               1.0 - cand[:, 0], 1.0 - cand[:, 1]])

    selected = []
    first = np.argmax(border)          # centre of the square
    selected.append(cand[first])

    # minimal distance to any selected point for each candidate
    mindist = np.full(len(cand), np.inf)
    d0 = np.linalg.norm(cand - cand[first], axis=1)
    mindist = np.minimum(mindist, d0)

    for _ in range(1, n):
        limiting = np.minimum(border, mindist)
        idx = np.argmax(limiting)
        selected.append(cand[idx])
        d = np.linalg.norm(cand - cand[idx], axis=1)
        mindist = np.minimum(mindist, d)

    return np.array(selected)


def _feasible_radii(centers):
    """Largest radii that respect borders and pairwise non‑overlap."""
    n = centers.shape[0]
    # distance to borders
    border_r = np.minimum.reduce([centers[:, 0], centers[:, 1],
                                 1.0 - centers[:, 0], 1.0 - centers[:, 1]])

    # pairwise centre distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)

    # feasible radii: limited by border and half the nearest neighbour distance
    radii = np.minimum(border_r, nearest / 2.0) * 0.999  # tiny safety margin
    return radii


def _make_constraints(n):
    """Create SLSQP inequality constraints for n circles."""
    cons = []

    # ---- border constraints ------------------------------------------------
    for i in range(n):
        # x_i - r_i >= 0
        cons.append({'type': 'ineq',
                     'fun': lambda v, i=i: v[i] - v[2 * n + i]})
        # 1 - (x_i + r_i) >= 0
        cons.append({'type': 'ineq',
                     'fun': lambda v, i=i: 1.0 - (v[i] + v[2 * n + i])})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq',
                     'fun': lambda v, i=i: v[n + i] - v[2 * n + i]})
        # 1 - (y_i + r_i) >= 0
        cons.append({'type': 'ineq',
                     'fun': lambda v, i=i: 1.0 - (v[n + i] + v[2 * n + i])})

    # ---- non‑overlap constraints -------------------------------------------
    for i in range(n):
        for j in range(i + 1, n):
            def con_fun(v, i=i, j=j):
                xi, yi, ri = v[i], v[n + i], v[2 * n + i]
                xj, yj, rj = v[j], v[n + j], v[2 * n + j]
                return (xi - xj) ** 2 + (yi - yj) ** 2 - (ri + rj) ** 2
            cons.append({'type': 'ineq', 'fun': con_fun})

    return cons


def _local_optimize(x0, n, bounds, constraints):
    """Run a relatively long SLSQP optimisation from x0."""
    def objective(v):
        # maximise sum(r) → minimise negative sum
        return -np.sum(v[2 * n:])

    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False}
    )
    # Enforce non‑negative radii (tiny numerical negatives may appear)
    v_opt = res.x if res.success else x0
    v_opt[2 * n:] = np.maximum(v_opt[2 * n:], 0.0)
    return v_opt, res.success


def construct_circles():
    """Return an (26,3) array with (x, y, r) for each circle."""
    n = 26
    rng = np.random.default_rng(20231123)

    # ------------------------------------------------------------------
    # 1. Assemble a pool of feasible starting layouts
    # ------------------------------------------------------------------
    starts = []

    # deterministic greedy start
    centers_g = _greedy_initial(n=n, step=0.02)
    radii_g = _feasible_radii(centers_g)
    starts.append((centers_g, radii_g))

    # a few perturbed greedy variants
    for _ in range(6):
        pert = centers_g + rng.uniform(-0.02, 0.02, size=centers_g.shape)
        pert = np.clip(pert, 0.0, 1.0)
        radii_p = _feasible_radii(pert)
        starts.append((pert, radii_p))

    # random feasible starts
    for _ in range(12):
        rand = rng.uniform(0.0, 1.0, size=(n, 2))
        radii_r = _feasible_radii(rand)
        starts.append((rand, radii_r))

    # ------------------------------------------------------------------
    # 2. Optimisation settings (shared across all starts)
    # ------------------------------------------------------------------
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    constraints = _make_constraints(n)

    best_sum = -np.inf
    best_solution = None

    start_time = time.time()
    time_budget = 470.0          # seconds; leave a safety margin

    # ------------------------------------------------------------------
    # 3. For each start run a basinhopping search (time‑bounded)
    # ------------------------------------------------------------------
    for centers0, radii0 in starts:
        # tiny random nudge to avoid exact degeneracies
        radii0 = np.maximum(radii0 + rng.normal(scale=1e-4, size=n), 0.0)
        x0 = np.hstack([centers0[:, 0], centers0[:, 1], radii0])

        # Local optimiser wrapper for basinhopping
        def local_minimiser(x):
            v_opt, _ = _local_optimize(x, n, bounds, constraints)
            return v_opt

        # Custom step taking a small Gaussian perturbation
        class StepSize:
            def __init__(self, sigma=0.04):
                self.sigma = sigma
            def __call__(self, x):
                step = rng.normal(scale=self.sigma, size=x.shape)
                x_new = x + step
                # clip to bounds
                for i, (lo, hi) in enumerate(bounds):
                    x_new[i] = np.clip(x_new[i], lo, hi)
                return x_new

        # basinhopping options
        bh = basinhopping(
            func=lambda v: -np.sum(v[2 * n:]),   # note: we *minimise* the negative sum
            x0=x0,
            niter=30,
            T=1.0,
            stepsize=0.04,
            take_step=StepSize(sigma=0.04),
            minimizer_kwargs={
                'method': 'SLSQP',
                'bounds': bounds,
                'constraints': constraints,
                'options': {'maxiter': 800, 'ftol': 1e-12, 'disp': False}
            },
            callback=lambda x, f, accept: None,
            disp=False
        )

        v_opt = bh.x
        # Ensure radii are non‑negative
        v_opt[2 * n:] = np.maximum(v_opt[2 * n:], 0.0)

        sum_r = np.sum(v_opt[2 * n:])
        if sum_r > best_sum:
            best_sum = sum_r
            best_solution = v_opt

        # Global time check
        if time.time() - start_time > time_budget:
            break

    # ------------------------------------------------------------------
    # 4. Fallback (should never be needed)
    # ------------------------------------------------------------------
    if best_solution is None:
        best_solution = np.hstack([centers_g[:, 0], centers_g[:, 1], radii_g])

    # ------------------------------------------------------------------
    # 5. Assemble final circles array
    # ------------------------------------------------------------------
    opt_centers = np.column_stack([best_solution[:n], best_solution[n:2 * n]])
    opt_radii = best_solution[2 * n:]

    circles = np.column_stack([opt_centers, opt_radii])
    return circles
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_code():
    """Run the circle packing constructor for n=26"""
    circles = construct_circles()
    sum_radii = float(np.sum(circles[:, 2]))
    return circles, sum_radii


