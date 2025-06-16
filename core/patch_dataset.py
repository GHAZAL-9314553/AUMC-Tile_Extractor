import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch


class PatchDataset(Dataset):
    def __init__(self, patch_dir, transform=None, format='png'):
        """
        Args:
            patch_dir (str): Directory containing patches
            transform (callable, optional): Transform to apply to patches
            format (str): 'png', 'npy', or 'hdf5'
        """
        self.patch_dir = patch_dir
        self.transform = transform
        self.format = format.lower()
        self.file_list = sorted([
            f for f in os.listdir(patch_dir)
            if f.endswith(self.format)
        ])

        if len(self.file_list) == 0:
            raise ValueError(f"No {self.format} files found in {patch_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        path = os.path.join(self.patch_dir, fname)

        if self.format == 'png':
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        elif self.format == 'npy':
            arr = np.load(path)
            image = torch.from_numpy(arr).float()
            if image.ndim == 2:
                image = image.unsqueeze(0)
            elif image.ndim == 3 and image.shape[2] in (1, 3):
                image = image.permute(2, 0, 1)

        else:
            raise NotImplementedError(f"Format {self.format} not supported")

        return image