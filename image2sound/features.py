from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from .utils import get_file_seed

@dataclass
class ImageFeatures:
    """Container for extracted image features.
    
    Attributes:
        brightness: Average brightness value normalized to [0,1]
        contrast: Standard deviation of grayscale values normalized to [0,1] 
        edge_density: Density of detected edges normalized to [0,1]
        palette_rgb: List of 5 dominant RGB color tuples
        palette_variance: Variance of k-means color centers [0,1]
        texture_energy: Average Gabor filter energy across orientations [0,1]  
        cx: Brightness center of mass x-coordinate normalized to [0,1]
        cy: Brightness center of mass y-coordinate normalized to [0,1]
        seed: 32-bit deterministic seed from file hash
    """
    brightness: float
    contrast: float
    edge_density: float
    palette_rgb: list[tuple[int, int, int]]
    palette_variance: float
    texture_energy: float
    cx: float
    cy: float
    seed: int


def gabor_energy(gray: np.ndarray) -> float:
    """Compute average Gabor filter energy across multiple orientations.
    
    Fast texture analysis using OpenCV's Gabor filters at 0°, 45°, 90°, 135°.
    
    Args:
        gray: Grayscale image array [0,1]
        
    Returns:
        Average filter energy normalized to [0,1]
    """
    # Convert to uint8 for OpenCV
    gray_uint8 = (gray * 255).astype(np.uint8)
    
    # Gabor filter parameters for texture detection
    ksize = 31  # Kernel size
    sigma = 4   # Standard deviation 
    lambd = 10  # Wavelength
    gamma = 0.5 # Aspect ratio
    psi = 0     # Phase offset
    
    total_energy = 0.0
    orientations = [0, 45, 90, 135]  # degrees
    
    for theta_deg in orientations:
        theta = np.deg2rad(theta_deg)
        
        # Create Gabor kernel
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        
        # Apply filter and compute energy
        filtered = cv2.filter2D(gray_uint8, cv2.CV_8UC3, kernel)
        energy = float(np.mean(filtered ** 2))
        total_energy += energy
    
    # Average across orientations and normalize
    avg_energy = total_energy / len(orientations)
    # Normalize to [0,1] range (empirically determined scaling)
    return min(1.0, avg_energy / 65025.0)  # 255^2 for max uint8 energy


def brightness_center(gray: np.ndarray) -> tuple[float, float]:
    """Compute brightness-weighted center of mass.
    
    Fast vectorized computation of where the brightness is concentrated.
    
    Args:
        gray: Grayscale image array [0,1]
        
    Returns:
        (cx, cy) coordinates normalized to [0,1]
    """
    h, w = gray.shape
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[0:h, 0:w]
    
    # Compute weighted center of mass
    total_brightness = np.sum(gray)
    if total_brightness > 0:
        cx = float(np.sum(gray * x_coords) / total_brightness / w)
        cy = float(np.sum(gray * y_coords) / total_brightness / h)
    else:
        # If image is completely dark, center is at middle
        cx, cy = 0.5, 0.5
    
    return cx, cy


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
    
    print("   [60%] 🔲 Detecting edges...")
    edges = cv2.Canny((gray * 255).astype(np.uint8), 100, 200)
    edge_density = float(edges.mean()) / 255.0

    print("   [70%] 🎨 Extracting color palette...")
    flat = arr.reshape(-1, 3)
    km = KMeans(n_clusters=k_palette, n_init="auto", random_state=0).fit(flat)
    centers = (km.cluster_centers_ * 255).astype(int)
    palette = [tuple(map(int, c)) for c in centers]
    
    # Compute palette variance (spread of color centers)
    palette_variance = float(np.var(km.cluster_centers_))
    
    print("   [80%] 🌀 Computing texture energy...")
    texture_energy = gabor_energy(gray)
    
    print("   [90%] 📍 Finding brightness center of mass...")
    cx, cy = brightness_center(gray)
    
    print("   [95%] 🔢 Generating deterministic seed...")
    seed = get_file_seed(path)

    print("   [100%] ✨ Feature extraction complete!")
    print(f"   📊 Basic: brightness={brightness:.2f}, contrast={contrast:.2f}, edges={edge_density:.2f}")
    print(f"   🌈 Palette: {len(palette)} colors, variance={palette_variance:.3f}")
    print(f"   🌀 Texture energy: {texture_energy:.3f}")
    print(f"   📍 Center of mass: ({cx:.2f}, {cy:.2f})")
    print(f"   🎲 Deterministic seed: {seed}")

    return ImageFeatures(
        brightness=brightness,
        contrast=contrast, 
        edge_density=edge_density,
        palette_rgb=palette,
        palette_variance=palette_variance,
        texture_energy=texture_energy,
        cx=cx,
        cy=cy,
        seed=seed
    )
