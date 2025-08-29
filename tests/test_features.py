from pathlib import Path
from image2sound.features import extract_features

def test_extract_features_smoke(tmp_path: Path):
    # generate a tiny image
    p = tmp_path / "img.png"
    from PIL import Image
    Image.new("RGB", (10,10), color=(200, 50, 50)).save(p)
    feats = extract_features(p)
    assert 0 <= feats.brightness <= 1
    assert 0 <= feats.contrast <= 1
    assert 0 <= feats.edge_density <= 1
    assert len(feats.palette_rgb) == 5
