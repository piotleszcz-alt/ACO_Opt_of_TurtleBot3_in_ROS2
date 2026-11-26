from aco_planner.map_data import MapData
import matplotlib.pyplot as plt
import aco_planner.config as config
import numpy as np
import aco_planner.inflation  as inflation
from aco_planner.aco_opt import ACOPlanner
from aco_planner.tuning import run_tuning, scaled_params


def main():

    profile = config.PROFILES[config.PROFILE]
    map = MapData(profile["map"])

    print("Resolution:", map.get_resolution())
    print("Origin:", map.get_origin())
    print("Map shape:", map.map_data.shape, map.map_data.dtype)

    # jakie wartości pikseli są w obrazie i ile ich jest
    values, counts = np.unique(map.map_data, return_counts=True)
    print("Pixel values:", dict(zip(values.tolist(), counts.tolist())))

    p = map.occupancy_probability()
    print("p range:", p.min(), "-", p.max())

    classified = map.classify()
    print("Free:", (classified == 0).sum(),
          "Occupied:", (classified == 1).sum(),
          "Unknown:", (classified == -1).sum())

    grid = map.binarize_map()
    print("Binary obstacles:", grid.sum(), "of", grid.size)

    dilated_grid = inflation.dilatation(config.ROBOT_SAFE_RADIUS, grid, map.get_resolution())

    # podgląd: klasyfikacja, binaryzacja i dylatacja
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(classified, cmap='gray_r', origin='lower')
    axes[0].set_title("Classified (-1/0/1)")

    axes[1].imshow(grid, cmap='gray_r', origin='lower')
    axes[1].set_title("Binary")

    axes[2].imshow(dilated_grid, cmap='gray_r', origin='lower')
    axes[2].set_title(f"Dilated (r = {config.ROBOT_SAFE_RADIUS} m)")

    plt.tight_layout()
    plt.show()
    
    if config.RUN_TUNING:
        run_tuning(dilated_grid, profile["params"], scaling=profile["scaling"])
    else:
        if config.PROFILE == "maze":
            start, goal = (19, 19), (83, 83)      # labirynt 
        elif config.PROFILE == "arena":
            start, goal = (160, 180), (240, 210)
        params, L_min = scaled_params(start, goal, profile["params"], profile["scaling"])
        planner = ACOPlanner(dilated_grid, start=start, goal=goal, **params)

        best = planner.run()

        if best is None:
            print("No ant reached the goal")
        else:

            REF = 226.5      # najkrótsza trasa start → wyjście 

            print(f"J = {planner.best_cost:.2f}, shortest = {REF:.1f}, "
                  f"J/shortest = {planner.best_cost / REF:.3f}")

            rows, cols = zip(*best)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(dilated_grid, cmap='gray_r', origin='lower')
            axes[0].plot(cols, rows, 'r-', linewidth=2)
            axes[0].plot(*planner.start[::-1], 'go')
            axes[0].plot(*planner.goal[::-1], 'bo')
            axes[0].set_title(f"Best path, J = {planner.best_cost:.2f}")

            axes[1].imshow(planner.tau, cmap='hot', origin='lower')
            axes[1].set_title("Pheromone τ")

            axes[2].plot(planner.history)
            axes[2].axhline(L_min, color='gray', linestyle='--', label="L_min")
            axes[2].set_xlabel("iteracja")
            axes[2].set_ylabel("best cost")
            axes[2].set_title("Convergence")
            axes[2].legend()

            plt.tight_layout()
            plt.show()
if __name__ == "__main__":
    main()