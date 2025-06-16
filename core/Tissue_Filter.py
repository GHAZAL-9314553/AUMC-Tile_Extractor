import torch
import numpy as np
from torchvision import transforms


class TissueClassifier:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def is_tissue(self, tile: np.ndarray, threshold: float = 0.5) -> bool:
        input_tensor = self.transform(tile).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = torch.sigmoid(self.model(input_tensor)).item()
        return output >= threshold
