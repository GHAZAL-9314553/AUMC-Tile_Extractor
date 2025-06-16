from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
import os

from masking_utils import BaseMasker
from tile_reader import BaseWSIReader


class BasePatchExtractor(ABC):
    """Abstract base class for patch extraction from WSIs."""

    def __init__(
        self,
        reader: BaseWSIReader,
        masker: Optional[BaseMasker] = None,
        patch_size: int = 256,
        stride: Optional[int] = None,
        level: int = 0,
    ):
        self.reader = reader
        self.masker = masker
        self.patch_size = patch_size
        self.stride = stride or patch_size  # default to non-overlapping
        self.level = level

    @abstractmethod
    def extract(self, save_dir: str) -> None:
        pass


class EfficientPatchExtractor(BasePatchExtractor):
    def extract(self, save_dir: str) -> None:
        width, height = self.reader.get_dimensions()
        level_factor = 2 ** self.level
        pw, ph = self.patch_size, self.patch_size
        sw, sh = self.stride, self.stride

        os.makedirs(save_dir, exist_ok=True)
        
        for y in range(0, height, sh * level_factor):
            for x in range(0, width, sw * level_factor):
                tile = self.reader.read_region(x, y, self.level, pw, ph)
                if tile.shape[0] != ph or tile.shape[1] != pw:
                    continue

                if self.masker:
                    mask = self.masker.get_mask(tile)
                    if not np.any(mask):
                        continue

                patch_path = os.path.join(save_dir, f"tile_x{x}_y{y}.png")
                self._save_patch(tile, patch_path)

    def _save_patch(self, patch: np.ndarray, path: str) -> None:
        from PIL import Image
        Image.fromarray(patch).save(path)