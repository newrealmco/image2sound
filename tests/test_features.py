from pathlib import Path
from PIL import Image
from image2sound.features import extract_features

def test_extract_features_smoke(tmp_path: Path):
    """Test extract_features with a small solid-color image."""
    # Create a small solid-color image
    img_path = tmp_path / "test_img.png"
    Image.new("RGB", (10, 10), color=(200, 50, 50)).save(img_path)
    
    # Extract features
    features = extract_features(img_path)
    
    # Assert basic feature ranges
    assert 0 <= features.brightness <= 1
    assert 0 <= features.contrast <= 1  
    assert 0 <= features.edge_density <= 1
    
    # Assert new feature ranges
    assert 0 <= features.palette_variance <= 1
    assert 0 <= features.texture_energy <= 1
    assert 0 <= features.cx <= 1
    assert 0 <= features.cy <= 1
    
    # Assert seed is a valid integer
    assert isinstance(features.seed, int)
    assert 0 <= features.seed <= 4294967295  # 32-bit unsigned int max
    
    # Assert palette length and structure
    assert len(features.palette_rgb) == 5
    assert isinstance(features.palette_rgb, list)
    for rgb in features.palette_rgb:
        assert isinstance(rgb, tuple)
        assert len(rgb) == 3
        for channel in rgb:
            assert isinstance(channel, int)
            assert 0 <= channel <= 255


def test_deterministic_seed(tmp_path: Path):
    """Test that the same image produces the same seed."""
    # Create identical images
    img_path1 = tmp_path / "test1.png" 
    img_path2 = tmp_path / "test2.png"
    img = Image.new("RGB", (20, 20), color=(100, 150, 200))
    img.save(img_path1)
    img.save(img_path2)
    
    # Extract features from both
    features1 = extract_features(img_path1)
    features2 = extract_features(img_path2)
    
    # Seeds should be identical for identical image content
    assert features1.seed == features2.seed
    
    # But different file paths should still give same seed for same content
    assert features1.brightness == features2.brightness
