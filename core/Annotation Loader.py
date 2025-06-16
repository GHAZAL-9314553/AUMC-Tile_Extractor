import numpy as np
import os
from typing import Union
from PIL import Image
import json


def load_annotation_mask(path: str, shape: Union[tuple, None] = None) -> np.ndarray:
    """
    Load annotation mask from a .png or .json file.
    - PNG mask: binary or labeled mask.
    - JSON mask: assumed to contain polygon coordinates (simplified).
    """
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".png":
        mask = np.array(Image.open(path).convert("L"))
        return mask > 0  # Binary mask

    elif ext == ".json":
        if shape is None:
            raise ValueError("You must provide 'shape' (H, W) when using JSON mask.")
        return mask_from_json(path, shape)

    else:
        raise ValueError(f"Unsupported annotation format: {ext}")


def mask_from_json(json_path: str, shape: tuple) -> np.ndarray:
    from shapely.geometry import Polygon
    from rasterio.features import rasterize

    with open(json_path, 'r') as f:
        data = json.load(f)

    polygons = []
    for obj in data.get("shapes", []):
        if obj.get("type", "polygon") == "polygon":
            coords = obj["points"]
            polygons.append(Polygon(coords))

    mask = rasterize(
        [(poly, 1) for poly in polygons],
        out_shape=shape,
        fill=0,
        default_value=1,
        dtype='uint8'
    )
    return mask > 0
