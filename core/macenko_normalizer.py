import numpy as np
import cv2
import json
from sklearn.decomposition import PCA

class MacenkoNormalizer:
    def __init__(self, alpha=1, beta=0.15, light_intensity=255):
        self.alpha = alpha
        self.beta = beta
        self.light_intensity = light_intensity
        self.stain_matrix = None
        self.max_sat = None

    def fit(self, target_image: np.ndarray):
        target_image = self._standardize_brightness(target_image)
        od = self._rgb_to_od(target_image)
        od = od[~np.any(od < self.beta, axis=1)]

        _, eigvecs = np.linalg.eigh(np.cov(od.T))
        eigvecs = eigvecs[:, [1, 2]]  # Take top 2 eigenvectors

        if eigvecs[0, 0] < 0:
            eigvecs[:, 0] *= -1
        if eigvecs[0, 1] < 0:
            eigvecs[:, 1] *= -1

        projected = np.dot(od, eigvecs)
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        min_phi = np.percentile(angles, self.alpha)
        max_phi = np.percentile(angles, 100 - self.alpha)

        v1 = np.dot(eigvecs, [np.cos(min_phi), np.sin(min_phi)])
        v2 = np.dot(eigvecs, [np.cos(max_phi), np.sin(max_phi)])

        if v1[0] > v2[0]:
            self.stain_matrix = np.stack([v1, v2], axis=1)


            
        else:
            self.stain_matrix = np.stack([v2, v1], axis=1)

        concentrations = np.linalg.lstsq(self.stain_matrix, od.T, rcond=None)[0]
        self.max_sat = np.percentile(concentrations, 99, axis=1, keepdims=True)

    def transform(self, image: np.ndarray) -> np.ndarray:
        image = self._standardize_brightness(image)
        od = self._rgb_to_od(image)
        concentrations = np.linalg.lstsq(self.stain_matrix, od.T, rcond=None)[0]
        concentrations = concentrations / self.max_sat
        od_reconstructed = np.dot(self.stain_matrix, concentrations)
        recon_rgb = self._od_to_rgb(od_reconstructed.T)
        return recon_rgb.reshape(image.shape)

    def save_vector(self, path: str):
        data = {
            "stain_vectors": self.stain_matrix.tolist(),
            "max_sat": self.max_sat.tolist()
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_vector(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.stain_matrix = np.array(data["stain_vectors"])
        self.max_sat = np.array(data["max_sat"])

    def _rgb_to_od(self, rgb):
        rgb = rgb.astype(np.float32)
        rgb[rgb == 0] = 1  # prevent division by 0
        return -np.log(rgb / self.light_intensity)

    def _od_to_rgb(self, od):
        return np.clip(self.light_intensity * np.exp(-od), 0, 255).astype(np.uint8)

    def _standardize_brightness(self, img):
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_lab[:, :, 0] = 128
        return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)