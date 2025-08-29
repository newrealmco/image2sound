from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans

@dataclass
class ImageFeatures:
    brightness: float
    contrast: float
    edge_density: float
    palette_rgb: list  # list[tuple[int,int,int]]

def extract_features(path: Path, k_palette: int = 5) -> ImageFeatures:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    brightness = float(gray.mean())
    contrast = float(gray.std())
    edges = cv2.Canny((gray * 255).astype(np.uint8), 100, 200)
    edge_density = float(edges.mean()) / 255.0

    flat = arr.reshape(-1, 3)
    km = KMeans(n_clusters=k_palette, n_init="auto", random_state=0).fit(flat)
    centers = (km.cluster_centers_ * 255).astype(int)
    palette = [tuple(map(int, c)) for c in centers]

    return ImageFeatures(brightness, contrast, edge_density, palette)
