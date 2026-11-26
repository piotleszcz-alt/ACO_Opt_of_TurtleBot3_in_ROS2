import contextlib
import io
import math
import time

import matplotlib.pyplot as plt
import numpy as np

from aco_planner.aco_opt import ACOPlanner


# Scenariusze testowe: (nazwa, start (r, c), cel (r, c))
# DEFAULT_SCENARIOS = [
#     ("krótka, dół areny",       (160, 180), (165, 215)),
#     ("krótka, w pionie",        (165, 178), (187, 178)),
#     ("środkowa kolumna",        (160, 200), (240, 200)),
#     ("skos dół-góra",           (160, 180), (240, 210)),
#     ("wierzchołki lewo-prawo",  (200, 160), (200, 235)),   # przez środkowy rząd słupków
#     ("róg w róg",               (165, 180), (235, 220)),   # najdłuższa, po przekątnej
# ]

DEFAULT_SCENARIOS = [
    ("start -> koniec", (19, 19), (83, 83)),
    ("środek -> wyjście", (51, 51), (83, 83))
    ]

# Jak dobierać parametry zależne od długości trasy
DEFAULT_SCALING = {
    "ants_per_px": 0.25,     # mrówek na piksel odległości oktylnej
    "ants_min": 20,
    "ants_max": 50,
    "steps_factor": 15.0,    # max_steps = steps_factor * L_min
    "steps_min": 100,
    "scale_Q": False,       # True -> Q_const = L_min (depozyt niezależny od długości trasy)
}


def octile_distance(start, goal):
    """Shortest possible path length on an 8-connected grid (no obstacles)."""
    dr = abs(goal[0] - start[0])
    dc = abs(goal[1] - start[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)


def scaled_params(start, goal, base_params, scaling=DEFAULT_SCALING):
    """Returns a copy of base_params with m_ants, max_steps (and optionally Q) scaled to the distance."""
    L_min = octile_distance(start, goal)
    params = dict(base_params)

    params["m_ants"] = int(np.clip(round(scaling["ants_per_px"] * L_min),
                                   scaling["ants_min"], scaling["ants_max"]))
    params["max_steps"] = int(max(scaling["steps_min"],
                                  math.ceil(scaling["steps_factor"] * L_min)))
    if scaling.get("scale_Q", False):
        params["Q_const"] = L_min

    return params, L_min


def last_improvement(history):
    """Iteration number (from 1) in which the best cost improved for the last time."""
    last = 0
    for i in range(1, len(history)):
        if history[i] < history[i - 1]:
            last = i
    return last + 1


def run_scenario(grid, start, goal, params, seeds):
    """Runs one scenario for all seeds. Returns a list of results."""
    runs = []
    for seed in seeds:
        planner = ACOPlanner(grid, start=start, goal=goal, seed=seed, **params)

        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):   # wycisza print z run()
            planner.run()
        elapsed = time.perf_counter() - t0

        runs.append({
            "seed": seed,
            "planner": planner,
            "cost": planner.best_cost,
            "time": elapsed,
            "last_impr": last_improvement(planner.history),
        })
    return runs


def summarize(name, L_min, params, runs):
    """Computes statistics for a single scenario."""
    costs = [r["cost"] for r in runs if r["cost"] != np.inf]
    summary = {
        "name": name,
        "L_min": L_min,
        "m_ants": params["m_ants"],
        "max_steps": params["max_steps"],
        "found": len(costs),
        "total": len(runs),
        "time": np.mean([r["time"] for r in runs]),
        "last_impr": np.mean([r["last_impr"] for r in runs]),
    }
    if costs:
        summary.update({
            "mean": np.mean(costs),
            "std": np.std(costs),
            "min": min(costs),
            "ratio": np.mean(costs) / L_min,
        })
    return summary


def print_table(summaries, max_iter):
    header = (f"{'scenario':<24}{'L_min':>7}{'ants':>8}{'steps':>7}{'found':>7}"
              f"{'mean J':>8}{'std':>7}{'min J':>8}{'J/Lmin':>8}{'last impr':>10}{'time[s]':>9}")
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        if s["found"] == 0:
            print(f"{s['name']:<24}{s['L_min']:>7.1f}{s['m_ants']:>8}{s['max_steps']:>7}"
                  f"{'0/' + str(s['total']):>7}   NO PATH")
            continue
        print(f"{s['name']:<24}{s['L_min']:>7.1f}{s['m_ants']:>8}{s['max_steps']:>7}"
              f"{str(s['found']) + '/' + str(s['total']):>7}"
              f"{s['mean']:>8.2f}{s['std']:>7.2f}{s['min']:>8.2f}{s['ratio']:>8.3f}"
              f"{s['last_impr']:>6.0f}/{max_iter:<3}{s['time']:>9.2f}")

    ratios = [s["ratio"] for s in summaries if "ratio" in s]
    if ratios:
        print(f"\nMean J/L_min across scenarios: {np.mean(ratios):.3f}  "
              f"(worst: {max(ratios):.3f})")


def free_area_bounds(grid, margin=10):
    """Row and column range of the free space, used to zoom the plot onto the arena."""
    rows, cols = np.where(grid == 0)
    return (cols.min() - margin, cols.max() + margin,
            rows.min() - margin, rows.max() + margin)


def plot_results(grid, results, max_iter):
    """One row per scenario: map with the best path and convergence curves of all seeds."""
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.6 * n), squeeze=False)
    x0, x1, y0, y1 = free_area_bounds(grid)

    for row, (summary, runs) in enumerate(results):
        ax_map, ax_conv = axes[row]

        ax_map.imshow(grid, cmap='gray_r', origin='lower')
        ax_map.set_xlim(x0, x1)
        ax_map.set_ylim(y0, y1)

        best_run = min(runs, key=lambda r: r["cost"])
        planner = best_run["planner"]
        if planner.best_path is not None:
            rows_, cols_ = zip(*planner.best_path)
            ax_map.plot(cols_, rows_, 'r-', linewidth=2)
        ax_map.plot(planner.start[1], planner.start[0], 'go')
        ax_map.plot(planner.goal[1], planner.goal[0], 'bo')

        title = f"{summary['name']}"
        if summary["found"]:
            title += f"\nbest J = {summary['min']:.1f}, L_min = {summary['L_min']:.1f}"
        else:
            title += "\nbrak trasy"
        ax_map.set_title(title, fontsize=9)

        for r in runs:
            if r["cost"] != np.inf:
                ax_conv.plot(r["planner"].history, label=f"seed {r['seed']}")
        ax_conv.axhline(summary["L_min"], color='gray', linestyle='--', label="L_min")
        ax_conv.set_xlim(0, max_iter)
        ax_conv.set_xlabel("iteracja")
        ax_conv.set_ylabel("best cost")
        ax_conv.set_title(f"mrówki = {summary['m_ants']}, max_steps = {summary['max_steps']}",
                          fontsize=9)
        ax_conv.legend(fontsize=7)

    plt.tight_layout()
    plt.show()


def run_tuning(grid, base_params, scenarios=DEFAULT_SCENARIOS, seeds=range(3),
               scaling=DEFAULT_SCALING, plot=True):
    """
    Main tuning function.

    grid        - dilated map (0 = free, 1 = obstacle)
    base_params - ACOPlanner parameters (e.g. config.ACO_PARAMS); m_ants and max_steps
                  will be overridden with values scaled to the path length
    scenarios   - list of (name, start, goal)
    seeds       - seeds to test for each scenario
    scaling     - rules for choosing m_ants / max_steps / Q
    plot        - whether to draw the plot
    """
    results = []
    max_iter = base_params.get("max_iter", 50)

    for name, start, goal in scenarios:
        params, L_min = scaled_params(start, goal, base_params, scaling)
        print(f"[{name}] start={start}, goal={goal}, L_min={L_min:.1f}, "
              f"ants={params['m_ants']}, max_steps={params['max_steps']} ...")
        try:
            runs = run_scenario(grid, start, goal, params, seeds)
        except ValueError as e:      # np. start/cel na przeszkodzie
            print(f"  skipped: {e}")
            continue
        results.append((summarize(name, L_min, params, runs), runs))

    if not results:
        print("No scenario was executed.")
        return []

    print_table([s for s, _ in results], max_iter)

    if plot:
        plot_results(grid, results, max_iter)

    return results