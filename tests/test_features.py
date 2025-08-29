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
    
    # Assert value ranges
    assert 0 <= features.brightness <= 1
    assert 0 <= features.contrast <= 1  
    assert 0 <= features.edge_density <= 1
    
    # Assert palette length
    assert len(features.palette_rgb) == 5
    
    # Additional validation for solid color image
    assert isinstance(features.palette_rgb, list)
    for rgb in features.palette_rgb:
        assert isinstance(rgb, tuple)
        assert len(rgb) == 3
        for channel in rgb:
            assert isinstance(channel, int)
            assert 0 <= channel <= 255
