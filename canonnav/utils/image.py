"""Image normalization shared by training and inference."""
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_image(rgb01: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB in [0, 1] -> (3, H, W) float32, ImageNet mean/std normalized."""
    return ((rgb01 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
