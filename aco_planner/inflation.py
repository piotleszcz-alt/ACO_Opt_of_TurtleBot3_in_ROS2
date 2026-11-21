import numpy as np
from scipy.ndimage import binary_dilation
import math

def dilatation(radius: float, bin_map: np.ndarray, resolution: float) -> np.ndarray:

    radius_px = math.ceil(round(radius/resolution,6))
    coords = np.arange(-radius_px, radius_px + 1)
    x,y = np.meshgrid(coords,coords)
    mask = x**2 + y**2 <= radius_px**2

    result = binary_dilation(bin_map,mask).astype(np.uint8)

    return result
