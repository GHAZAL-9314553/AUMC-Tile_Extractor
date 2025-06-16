from abc import ABC, abstractmethod
import numpy as np
import os
from typing import Optional


class BaseSaver(ABC):
    @abstractmethod
    def save(self, patch: np.ndarray, path: str):
        pass


class PngSaver(BaseSaver):
    def save(self, patch: np.ndarray, path: str):
        from PIL import Image
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(patch).save(path)


class HDF5Saver(BaseSaver):
    def __init__(self, hdf5_path: str, dataset_name: str = "patches"):
        import h5py
        self.hdf5_path = hdf5_path
        self.dataset_name = dataset_name
        self._initialized = False
        self.h5file = h5py.File(hdf5_path, "w")
        self.index = 0

    def save(self, patch: np.ndarray, path: Optional[str] = None):
        if not self._initialized:
            self.dataset = self.h5file.create_dataset(
                self.dataset_name,
                shape=(0,) + patch.shape,
                maxshape=(None,) + patch.shape,
                dtype=patch.dtype,
                chunks=True
            )
            self._initialized = True

        self.dataset.resize((self.index + 1,) + patch.shape)
        self.dataset[self.index] = patch
        self.index += 1

    def close(self):
        self.h5file.close()
