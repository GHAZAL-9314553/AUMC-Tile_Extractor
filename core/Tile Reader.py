from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Union

try:
    import cucim
    from cucim.clara import CudaImage
    CUCIM_AVAILABLE = True
except ImportError:
    CUCIM_AVAILABLE = False


class BaseWSIReader(ABC):
    """Abstract base class for WSI (Whole Slide Image) readers."""

    def __init__(self, wsi_path: str):
        self.wsi_path = wsi_path

    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_downsample_levels(self) -> int:
        pass

    @abstractmethod
    def read_region(
        self, x: int, y: int, level: int, width: int, height: int
    ) -> np.ndarray:
        pass


class CuCIMWSIReader(BaseWSIReader):
    """CuCIM-based implementation of a WSI reader."""

    def __init__(self, wsi_path: str):
        if not CUCIM_AVAILABLE:
            raise ImportError("CuCIM is not installed or failed to import.")

        super().__init__(wsi_path)
        self.slide = cucim.CuImage(wsi_path)

    def get_dimensions(self) -> Tuple[int, int]:
        return tuple(self.slide.size)

    def get_downsample_levels(self) -> int:
        return len(self.slide.resolutions['level_dimensions'])

    def read_region(
        self, x: int, y: int, level: int, width: int, height: int
    ) -> np.ndarray:
        tile = self.slide.read_region((x, y), level, (width, height))
        return np.array(tile)[:, :, :3]  # Ensure RGB only (no alpha)


# Future Readers (OpenSlide, ASAP, etc.) can subclass BaseWSIReader
# Example:
# class OpenSlideReader(BaseWSIReader):
#     ...
