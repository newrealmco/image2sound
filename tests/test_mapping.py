from image2sound.mapping import map_features_to_music, _rgb_to_hue
from image2sound.features import ImageFeatures

def test_rgb_to_hue():
    """Test _rgb_to_hue returns values in [0,360) range."""
    # Test pure colors
    assert _rgb_to_hue((255, 0, 0)) == 0.0  # Red
    assert abs(_rgb_to_hue((0, 255, 0)) - 120.0) < 1e-10  # Green
    assert abs(_rgb_to_hue((0, 0, 255)) - 240.0) < 1e-10  # Blue
    
    # Test grayscale (should return 0)
    assert _rgb_to_hue((128, 128, 128)) == 0.0
    assert _rgb_to_hue((0, 0, 0)) == 0.0
    assert _rgb_to_hue((255, 255, 255)) == 0.0

def test_mapping_ranges():
    """Test that all mapped parameters are within expected ranges."""
    feats = ImageFeatures(brightness=0.7, contrast=0.2, edge_density=0.1,
                          palette_rgb=[(255,0,0)]*5)
    p = map_features_to_music(feats, style="ambient", target_duration=5.0)
    
    # Test ranges
    assert 60 <= p.bpm <= 160
    assert 0 <= p.intensity <= 1
    assert p.duration == 5.0
    
    # Test root is valid key
    valid_roots = {"C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"}
    assert p.root in valid_roots
    
    # Test scale format
    assert p.scale.endswith("_major") or p.scale.endswith("_minor")
    assert p.scale.startswith(p.root)

def test_duration_passthrough():
    """Test that target_duration is passed through correctly."""
    feats = ImageFeatures(brightness=0.5, contrast=0.3, edge_density=0.2,
                          palette_rgb=[(100,150,200)]*5)
    
    for duration in [10.0, 30.0, 60.0]:
        p = map_features_to_music(feats, target_duration=duration)
        assert p.duration == duration

def test_style_effects():
    """Test that styles produce expected BPM differences and other effects."""
    # Use same features for all styles to compare effects
    feats = ImageFeatures(brightness=0.6, contrast=0.4, edge_density=0.3,
                          palette_rgb=[(128,128,128)]*5)
    
    neutral = map_features_to_music(feats, style="neutral")
    ambient = map_features_to_music(feats, style="ambient") 
    cinematic = map_features_to_music(feats, style="cinematic")
    rock = map_features_to_music(feats, style="rock")
    
    # Ambient should have lower BPM than neutral
    assert ambient.bpm < neutral.bpm
    assert ambient.bpm >= 60  # Min BPM for ambient
    assert ambient.scale.endswith("_major")  # Force major
    assert ambient.instruments == ["pad","lead","bass"]
    
    # Cinematic should have higher BPM than neutral
    assert cinematic.bpm > neutral.bpm  
    assert cinematic.bpm <= 150  # Max BPM for cinematic
    assert cinematic.instruments == ["pad","lead","bass"]
    
    # Rock should have highest BPM
    assert rock.bpm > neutral.bpm
    assert rock.bpm <= 160  # Max BPM for rock
    assert rock.scale.endswith("_minor")  # Force minor
    assert rock.instruments == ["piano","lead","drums"]

def test_brightness_to_bpm_mapping():
    """Test BPM mapping from brightness values."""
    # Test min brightness
    feats_dark = ImageFeatures(brightness=0.0, contrast=0.5, edge_density=0.5,
                               palette_rgb=[(0,0,0)]*5)
    p_dark = map_features_to_music(feats_dark, style="neutral")
    assert p_dark.bpm == 80  # Min BPM
    
    # Test max brightness  
    feats_bright = ImageFeatures(brightness=1.0, contrast=0.5, edge_density=0.5,
                                 palette_rgb=[(255,255,255)]*5)
    p_bright = map_features_to_music(feats_bright, style="neutral")
    assert p_bright.bpm == 140  # Max BPM

def test_brightness_to_scale_mapping():
    """Test scale selection based on brightness."""
    # Bright image should use major scale
    feats_bright = ImageFeatures(brightness=0.8, contrast=0.5, edge_density=0.5,
                                 palette_rgb=[(200,200,200)]*5)
    p_bright = map_features_to_music(feats_bright, style="neutral")
    assert p_bright.scale.endswith("_major")
    
    # Dark image should use minor scale
    feats_dark = ImageFeatures(brightness=0.2, contrast=0.5, edge_density=0.5,
                               palette_rgb=[(50,50,50)]*5)
    p_dark = map_features_to_music(feats_dark, style="neutral")
    assert p_dark.scale.endswith("_minor")
