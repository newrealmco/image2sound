from dataclasses import dataclass
from .features import ImageFeatures

@dataclass
class MusicParams:
    """Container for musical parameters derived from image features.
    
    Attributes:
        bpm: Beats per minute (80-160 range)
        scale: Scale name in format "{root}_{major|minor}"
        root: Root note (C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B)
        instruments: List of instrument names for synthesis
        intensity: Musical intensity/dynamics [0,1]
        duration: Target duration in seconds
    """
    bpm: int
    scale: str
    root: str
    instruments: list[str]
    intensity: float
    duration: float

_HUES_TO_KEYS = ["C","G","D","A","E","B","F#","C#","Ab","Eb","Bb","F"]

def _rgb_to_hue(rgb: tuple[int,int,int]) -> float:
    """Convert RGB tuple to HSV hue value.
    
    Args:
        rgb: RGB color tuple with values in [0,255]
        
    Returns:
        Hue value in degrees [0,360)
    """
    r, g, b = [x/255 for x in rgb]
    mx, mn = max(r,g,b), min(r,g,b)
    if mx == mn: return 0.0
    if mx == r:  h = (60 * ((g-b)/(mx-mn)) + 360) % 360
    elif mx == g: h = (60 * ((b-r)/(mx-mn)) + 120) % 360
    else:         h = (60 * ((r-g)/(mx-mn)) + 240) % 360
    return h

def map_features_to_music(feats: ImageFeatures, style: str = "neutral", target_duration: float = 20.0) -> MusicParams:
    """Map image features to musical parameters for audio synthesis.
    
    Converts visual characteristics into musical elements using algorithmic mapping:
    - Dominant color hue → musical key (12-tone system)
    - Brightness → BPM (80-140) and major/minor scale selection
    - Edge density + contrast → musical intensity
    
    Args:
        feats: Extracted image features
        style: Musical style ("neutral", "ambient", "cinematic", "rock")
        target_duration: Desired audio length in seconds
        
    Returns:
        MusicParams with mapped musical characteristics
        
    Style Effects:
        - ambient: Lower BPM (min 60), force major scale, soft instruments
        - cinematic: Higher BPM (max 150), orchestral feel
        - rock: Highest BPM (max 160), force minor scale, rhythmic instruments
    """
    root_rgb = max(feats.palette_rgb, key=lambda c: sum(c))
    hue = _rgb_to_hue(root_rgb)
    key_ix = int(hue // 30) % 12
    root = _HUES_TO_KEYS[key_ix]

    bpm = int(80 + (140 - 80) * max(0.0, min(1.0, feats.brightness)))
    intensity = float(min(1.0, 0.5 * feats.edge_density + 0.5 * feats.contrast))
    scale = "major" if feats.brightness >= 0.5 else "minor"

    if style == "ambient":
        bpm = max(60, bpm - 20); scale = "major"
    elif style == "cinematic":
        bpm = min(150, bpm + 10)
    elif style == "rock":
        bpm = min(160, bpm + 20); scale = "minor"

    instruments = ["pad","lead","bass"] if style in ("ambient","cinematic") else ["piano","lead","drums"]
    return MusicParams(bpm=bpm, scale=f"{root}_{scale}", root=root,
                       instruments=instruments, intensity=intensity, duration=target_duration)
