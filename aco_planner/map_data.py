import yaml
import numpy as np
import PIL.Image as Image
from pathlib import Path

class MapData:
    def __init__(self, yaml_file_path: str):
        self.yaml_file_path = Path(yaml_file_path)  
        self.yaml_data = self.load_yaml()
        self.pgm_file_path = self.resolve_image_path() 
        self.map_data = self.load_map()

    def load_yaml(self):
        with open(self.yaml_file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def resolve_image_path(self):
        image_name = self.yaml_data['image']
        path = self.yaml_file_path.parent / image_name
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono obrazu mapy: {path}")
        return path

    def load_map(self):
        img = Image.open(self.pgm_file_path)
        return np.flipud(np.array(img))

    def get_resolution(self) -> float:
        return self.yaml_data['resolution']

    def get_origin(self) -> list[float]:
        return self.yaml_data['origin']

    def occupancy_probability(self) -> np.ndarray:
        negate = self.yaml_data['negate']
        img = self.map_data 

        if negate == 0:
            p = (255 - img) / 255
        else:
            p = img / 255

        return p

    def classify(self) -> np.ndarray:
        p = self.occupancy_probability()         
        occupied_thresh = self.yaml_data['occupied_thresh']
        free_thresh = self.yaml_data['free_thresh']

        classified_grid = np.full(p.shape, -1, dtype=np.int8)
        classified_grid[p < free_thresh] = 0
        classified_grid[p > occupied_thresh] = 1

        return classified_grid

    def binarize_map(self, unknown_as_obstacle: bool = True) -> np.ndarray:
        classified_grid = self.classify()
        unknown_value = 1 if unknown_as_obstacle else 0
        return np.where(classified_grid == -1, unknown_value, classified_grid)
