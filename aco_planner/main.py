from map_data import MapData
import matplotlib.pyplot as plt
import config
import numpy as np
import inflation 

def main():
    map = MapData(config.YAML_FILE_PATH)

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

    # podgląd: klasyfikacja, binaryzacja i dylatacja obok siebie
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(classified, cmap='gray_r', origin='lower')
    axes[0].set_title("Classified (-1/0/1)")

    axes[1].imshow(grid, cmap='gray_r', origin='lower')
    axes[1].set_title("Binary")

    axes[2].imshow(dilated_grid, cmap='gray_r', origin='lower')
    axes[2].set_title(f"Dilated (r = {config.ROBOT_SAFE_RADIUS} m)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()