from image2sound.mapping import map_features_to_music
from image2sound.features import ImageFeatures

def test_mapping_ranges():
    feats = ImageFeatures(brightness=0.7, contrast=0.2, edge_density=0.1,
                          palette_rgb=[(255,0,0)]*5)
    p = map_features_to_music(feats, style="ambient", target_duration=5.0)
    assert 60 <= p.bpm <= 160
    assert p.duration == 5.0
    assert p.root in {"C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"}
