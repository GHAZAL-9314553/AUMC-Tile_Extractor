import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from histolab.filters.image_filters import BluePenFilter, GreenPenFilter, RedPenFilter


class BaseMasker(ABC):
    @abstractmethod
    def get_mask(self, tile: np.ndarray) -> np.ndarray:
        pass


class OtsuMasker(BaseMasker):
    def get_mask(self, tile: np.ndarray) -> np.ndarray:
        gray = rgb2gray(tile)
        thresh = threshold_otsu(gray)
        return gray < thresh


class PenFilterMasker(BaseMasker):
    def __init__(self):
        self.filters = [RedPenFilter(), GreenPenFilter(), BluePenFilter()]

    def get_mask(self, tile: np.ndarray) -> np.ndarray:
        for f in self.filters:
            tile = f(tile)
        return np.ones(tile.shape[:2], dtype=bool)


class AnnotationMasker(BaseMasker):
    def __init__(self, annotation_mask: np.ndarray):
        self.annotation_mask = annotation_mask

    def get_mask(self, tile: np.ndarray) -> np.ndarray:
        return self.annotation_mask.astype(bool)


class UnifiedMasker(BaseMasker):
    def __init__(
        self,
        use_annotation: bool = False,
        use_otsu: bool = True,
        use_pen_filter: bool = True,
        annotation_mask: Optional[np.ndarray] = None
    ):
        self.use_annotation = use_annotation
        self.use_otsu = use_otsu
        self.use_pen_filter = use_pen_filter

        self.annotation_masker = AnnotationMasker(annotation_mask) if use_annotation and annotation_mask is not None else None
        self.otsu_masker = OtsuMasker() if use_otsu else None
        self.pen_masker = PenFilterMasker() if use_pen_filter else None

    def get_mask(self, tile: np.ndarray) -> np.ndarray:
        mask = np.ones(tile.shape[:2], dtype=bool)

        if self.otsu_masker:
            mask &= self.otsu_masker.get_mask(tile)

        if self.pen_masker:
            mask &= self.pen_masker.get_mask(tile)

        if self.annotation_masker:
            mask &= self.annotation_masker.get_mask(tile)

        return mask
