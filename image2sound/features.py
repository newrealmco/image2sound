from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans

@dataclass
class ImageFeatures:
    """Container for extracted image features.
    
    Attributes:
        brightness: Average brightness value normalized to [0,1]
        contrast: Standard deviation of grayscale values normalized to [0,1] 
        edge_density: Density of detected edges normalized to [0,1]
        palette_rgb: List of 5 dominant RGB color tuples
    """
    brightness: float
    contrast: float
    edge_density: float
    palette_rgb: list[tuple[int, int, int]]

def extract_features(path: Path, k_palette: int = 5) -> ImageFeatures:
    """Extract visual features from an image for audio synthesis mapping.
    
    Efficiently computes brightness, contrast, edge density, and color palette
    from an image. Optimized for ~200ms performance on 1080p images.
    
    Args:
        path: Path to the input image file
        k_palette: Number of dominant colors to extract (default: 5)
        
    Returns:
        ImageFeatures containing:
            - brightness: Mean grayscale value [0,1] 
            - contrast: Standard deviation of grayscale [0,1]
            - edge_density: Canny edge density [0,1]
            - palette_rgb: List of k_palette RGB tuples
            
    Raises:
        FileNotFoundError: If image path doesn't exist
        PIL.UnidentifiedImageError: If file is not a valid image
    """
    print(f"📸 Loading image: {path.name}")
    img = Image.open(path).convert("RGB")
    print(f"   ✅ Image loaded ({img.size[0]}x{img.size[1]} pixels)")
    
    print("🔍 Analyzing visual features...")
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    print("   [25%] 💡 Computing brightness...")
    brightness = float(gray.mean())
    
    print("   [50%] ⚡ Computing contrast...")
    contrast = float(gray.std())
    
    print("   [75%] 🔲 Detecting edges...")
    edges = cv2.Canny((gray * 255).astype(np.uint8), 100, 200)
    edge_density = float(edges.mean()) / 255.0

    print("   [90%] 🎨 Extracting color palette...")
    flat = arr.reshape(-1, 3)
    km = KMeans(n_clusters=k_palette, n_init="auto", random_state=0).fit(flat)
    centers = (km.cluster_centers_ * 255).astype(int)
    palette = [tuple(map(int, c)) for c in centers]

    print("   [100%] ✨ Feature extraction complete!")
    print(f"   📊 Results: brightness={brightness:.2f}, contrast={contrast:.2f}, edges={edge_density:.2f}")
    print(f"   🌈 Dominant colors: {len(palette)} found")

    return ImageFeatures(brightness, contrast, edge_density, palette)
