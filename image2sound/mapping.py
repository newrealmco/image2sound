from dataclasses import dataclass
from .features import ImageFeatures

@dataclass
class MusicParams:
    bpm: int
    scale: str
    root: str
    instruments: list[str]
    intensity: float
    duration: float

_HUES_TO_KEYS = ["C","G","D","A","E","B","F#","C#","Ab","Eb","Bb","F"]

def _rgb_to_hue(rgb: tuple[int,int,int]) -> float:
    r, g, b = [x/255 for x in rgb]
    mx, mn = max(r,g,b), min(r,g,b)
    if mx == mn: return 0.0
    if mx == r:  h = (60 * ((g-b)/(mx-mn)) + 360) % 360
    elif mx == g: h = (60 * ((b-r)/(mx-mn)) + 120) % 360
    else:         h = (60 * ((r-g)/(mx-mn)) + 240) % 360
    return h

def map_features_to_music(feats: ImageFeatures, style="neutral", target_duration=20.0) -> MusicParams:
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
