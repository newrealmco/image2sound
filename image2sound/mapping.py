from dataclasses import dataclass
import numpy as np
from .features import ImageFeatures, ColorCluster


@dataclass
class VoiceSpec:
    """Voice specification derived from color cluster properties.
    
    Attributes:
        instrument: Instrument type (pluck, bell, marimba, pad_glass, pad_warm, lead_clean, brass_short)
        mode_bias: Preference for passing tones [-1.0, 1.0] where -1=avoid, +1=prefer
        pan: Stereo pan position [-1.0, 1.0] where -1=left, 0=center, 1=right
        gain: Volume level [0.0, 1.0]
        octave: Octave offset from base register [-2, +2]
        brightness: Filter brightness/cutoff [0.0, 1.0] where 1.0=open filter
        activity: Note density multiplier [0.1, 2.0] where 1.0=normal density
        color: Source color cluster for reference
    """
    instrument: str
    mode_bias: float
    pan: float
    gain: float
    octave: int
    brightness: float
    activity: float
    color: ColorCluster

# Musical modes with semitone patterns
MODES = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],      # Major scale
    "dorian": [0, 2, 3, 5, 7, 9, 10],      # Minor with raised 6th
    "phrygian": [0, 1, 3, 5, 7, 8, 10],    # Minor with lowered 2nd
    "lydian": [0, 2, 4, 6, 7, 9, 11],      # Major with raised 4th
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],  # Major with lowered 7th
    "aeolian": [0, 2, 3, 5, 7, 8, 10],     # Natural minor
    "harm_minor": [0, 2, 3, 5, 7, 8, 11],  # Harmonic minor
}

# Time signatures (numerator, denominator)
METERS = [(4, 4), (3, 4), (6, 8), (5, 4)]

# Chord progressions by modal character
PROGRESSIONS = {
    "bright": [  # For ionian, lydian
        ["I", "V", "vi", "IV"],
        ["I", "vi", "IV", "V"],
        ["I", "IV", "V", "I"],
    ],
    "modal": [   # For dorian, mixolydian
        ["i", "bVII", "IV", "i"],
        ["i", "IV", "bVII", "i"],
        ["i", "bVI", "bVII", "i"],
    ],
    "dark": [    # For phrygian, harmonic minor, aeolian
        ["i", "bII", "bVI", "V"],
        ["i", "iv", "V", "i"],
        ["i", "VI", "III", "VII"],
    ],
}

@dataclass
class MusicParams:
    """Container for musical parameters derived from image features.
    
    Attributes:
        bpm: Beats per minute (80-160 range)
        scale: Scale name in format "{root}_{major|minor}" (legacy)
        root: Root note (C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B)
        instruments: List of instrument names for synthesis (legacy)
        intensity: Musical intensity/dynamics [0,1]
        duration: Target duration in seconds
        mode: Musical mode (ionian, dorian, phrygian, lydian, mixolydian, aeolian, harm_minor)
        meter: Time signature as (numerator, denominator)
        progression: List of chord symbols (e.g., ["I", "V", "vi", "IV"])
        pan_lead: Lead instrument stereo pan [-0.6, 0.6] (legacy)
        lead_offset: Lead melody transposition in semitones [-5, +5] (legacy)
        voices: List of VoiceSpec objects, one per color cluster
    """
    bpm: int
    scale: str
    root: str
    instruments: list[str]  # Legacy
    intensity: float
    duration: float
    mode: str
    meter: tuple[int, int]
    progression: list[str]
    pan_lead: float  # Legacy
    lead_offset: int  # Legacy
    voices: list[VoiceSpec]
    has_complement: bool  # Whether palette contains complementary colors
    chord_enrichment_level: int  # 0=basic, 1=7/add9, 2=#11/6
    texture_energy: float  # Texture energy from image analysis [0,1]

_HUES_TO_KEYS = ["C","G","D","A","E","B","F#","C#","Ab","Eb","Bb","F"]

# Instrument mapping based on hue ranges
INSTRUMENT_MAP = {
    # Red-Orange (0-60°): Warm, energetic instruments
    (0, 60): ["brass_short", "pluck", "lead_clean"],
    # Yellow-Green (60-150°): Bright, natural instruments  
    (60, 150): ["bell", "marimba", "pluck"],
    # Cyan-Blue (150-240°): Cool, ethereal instruments
    (150, 240): ["pad_glass", "bell", "lead_clean"],
    # Purple-Magenta (240-360°): Deep, mysterious instruments
    (240, 360): ["pad_warm", "brass_short", "marimba"]
}

def voice_spec_from_color(color: ColorCluster, rng: np.random.Generator) -> VoiceSpec:
    """Map color cluster properties to voice specifications.
    
    Uses hue to select instrument family, saturation/value for brightness/activity,
    proportion for gain, and spatial position for panning.
    
    Args:
        color: ColorCluster with HSV and spatial properties
        rng: Seeded random number generator for consistent choices
        
    Returns:
        VoiceSpec with instrument and performance parameters
    """
    # Map hue to instrument type
    hue = color.hue
    instrument_choices = None
    for (hue_min, hue_max), instruments in INSTRUMENT_MAP.items():
        if hue_min <= hue < hue_max:
            instrument_choices = instruments
            break
    
    if instrument_choices is None:
        instrument_choices = ["pluck"]  # Fallback
    
    # Select instrument with slight randomization
    instrument = rng.choice(instrument_choices)
    
    # Mode bias: high saturation prefers passing tones, low saturation avoids them
    mode_bias = (color.sat - 0.5) * 1.5  # Map [0,1] to [-0.75, 0.75]
    mode_bias = np.clip(mode_bias, -1.0, 1.0)
    
    # Pan from spatial position with some spread
    pan = (color.cx - 0.5) * 1.6  # Map [0,1] to [-0.8, 0.8]
    pan = np.clip(pan, -1.0, 1.0)
    
    # Gain from proportion (louder for more prominent colors)
    gain = 0.3 + 0.7 * color.prop  # Map proportion to [0.3, 1.0]
    gain = np.clip(gain, 0.0, 1.0)
    
    # Octave from vertical position (higher = higher pitch)
    octave_raw = (0.5 - color.cy) * 4  # Map [0,1] to [2, -2]
    octave = int(np.clip(np.round(octave_raw), -2, 2))
    
    # Brightness from HSV value (brighter colors = brighter timbre)
    brightness = color.val  # Already [0,1]
    
    # Activity from saturation (more saturated = more active)
    activity = 0.5 + color.sat * 1.5  # Map [0,1] to [0.5, 2.0]
    activity = np.clip(activity, 0.1, 2.0)
    
    return VoiceSpec(
        instrument=instrument,
        mode_bias=mode_bias,
        pan=pan,
        gain=gain,
        octave=octave,
        brightness=brightness,
        activity=activity,
        color=color
    )


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


def _select_mode(brightness: float, palette_variance: float, rng: np.random.Generator) -> str:
    """Select musical mode based on brightness and palette variance."""
    if brightness > 0.7:
        weights = {"ionian": 0.4, "lydian": 0.3, "mixolydian": 0.2, "dorian": 0.1}
    elif brightness < 0.3:
        if palette_variance > 0.5:
            weights = {"phrygian": 0.3, "harm_minor": 0.3, "aeolian": 0.2, "dorian": 0.2}
        else:
            weights = {"aeolian": 0.4, "dorian": 0.3, "phrygian": 0.2, "harm_minor": 0.1}
    else:
        weights = {"dorian": 0.25, "mixolydian": 0.25, "aeolian": 0.2, "ionian": 0.15, "phrygian": 0.1, "lydian": 0.05}
    
    modes = list(weights.keys())
    mode_weights = list(weights.values())
    return rng.choice(modes, p=mode_weights)


def _select_meter(texture_energy: float, rng: np.random.Generator) -> tuple[int, int]:
    """Select time signature based on texture energy."""
    if texture_energy > 0.8:
        return rng.choice([(6, 8), (5, 4)], p=[0.7, 0.3])
    elif texture_energy > 0.5:
        return (6, 8)
    elif texture_energy < 0.2:
        return rng.choice([(3, 4), (4, 4)], p=[0.6, 0.4])
    else:
        return (4, 4)


def _detect_complementary_colors(colors: list, threshold_prop: float = 0.1) -> bool:
    """Detect if palette contains complementary color pairs.
    
    Args:
        colors: List of ColorCluster objects
        threshold_prop: Minimum proportion for colors to be considered
        
    Returns:
        True if any pair of colors is ~180°±20° apart with sufficient proportion
    """
    significant_colors = [c for c in colors if c.prop > threshold_prop]
    
    for i in range(len(significant_colors)):
        for j in range(i + 1, len(significant_colors)):
            hue1, hue2 = significant_colors[i].hue, significant_colors[j].hue
            
            # Calculate hue difference (handle wraparound)
            diff = abs(hue1 - hue2)
            if diff > 180:
                diff = 360 - diff
            
            # Check if colors are complementary (~180°±20°)
            if 160 <= diff <= 200:
                return True
    
    return False


def _determine_chord_enrichment(palette_variance: float) -> int:
    """Determine chord enrichment level based on palette variance.
    
    Args:
        palette_variance: Variance in the color palette
        
    Returns:
        0 for basic chords, 1 for 7/add9, 2 for #11/6
    """
    T1 = 0.4  # Threshold for 7/add9 enrichment
    T2 = 0.7  # Threshold for #11/6 enrichment
    
    if palette_variance > T2:
        return 2  # Allow #11/6
    elif palette_variance > T1:
        return 1  # Allow 7/add9
    else:
        return 0  # Basic chords only


def _select_progression(mode: str, rng: np.random.Generator) -> list[str]:
    """Select chord progression based on mode character."""
    if mode in ["ionian", "lydian"]:
        category = "bright"
    elif mode in ["dorian", "mixolydian"]:
        category = "modal"
    else:  # phrygian, aeolian, harm_minor
        category = "dark"
    
    return list(rng.choice(PROGRESSIONS[category]))


def map_features_to_music(feats: ImageFeatures, style: str = "neutral", target_duration: float = 20.0) -> MusicParams:
    """Map image features to musical parameters for audio synthesis."""
    print(f"🎵 Mapping image to music ({style} style)...")
    
    # Create seeded RNG from image for deterministic choices
    rng = np.random.default_rng(feats.seed)
    
    print("   [10%] 🎤 Creating voice specifications from color clusters...")
    voices = []
    for i, color in enumerate(feats.colors):
        voice = voice_spec_from_color(color, rng)
        voices.append(voice)
        print(f"      Voice {i+1}: {voice.instrument} (gain={voice.gain:.2f}, pan={voice.pan:+.2f}, octave={voice.octave:+d})")
    
    print("   [15%] 🌈 Finding musical key from dominant color...")
    root_rgb = max(feats.palette_rgb, key=lambda c: sum(c))
    hue = _rgb_to_hue(root_rgb)
    key_ix = int(hue // 30) % 12
    root = _HUES_TO_KEYS[key_ix]
    print(f"   🎹 Dominant color RGB{root_rgb} → hue {hue:.0f}° → key of {root}")

    print("   [25%] 🎭 Selecting musical mode...")
    mode = _select_mode(feats.brightness, feats.palette_variance, rng)
    print(f"   🎼 Brightness {feats.brightness:.2f} + variance {feats.palette_variance:.3f} → {mode}")
    
    print("   [35%] ⏱️ Choosing time signature...")
    meter = _select_meter(feats.texture_energy, rng)
    print(f"   🥁 Texture energy {feats.texture_energy:.3f} → {meter[0]}/{meter[1]} time")
    
    print("   [40%] 🎨 Analyzing color relationships...")
    has_complement = _detect_complementary_colors(feats.colors)
    chord_enrichment_level = _determine_chord_enrichment(feats.palette_variance)
    
    complement_msg = "complementary colors detected" if has_complement else "no complementary pairs"
    enrichment_levels = ["basic triads", "7th/add9 chords", "#11/6th extensions"]
    print(f"   🌈 Color analysis: {complement_msg}, chord enrichment: {enrichment_levels[chord_enrichment_level]}")
    
    print("   [45%] 🎵 Selecting chord progression...")
    progression = _select_progression(mode, rng)
    print(f"   🎹 Mode {mode} → progression {' - '.join(progression)}")

    print("   [55%] ⚡ Converting brightness to tempo...")
    bpm = int(80 + (140 - 80) * max(0.0, min(1.0, feats.brightness)))
    print(f"   🎼 Brightness {feats.brightness:.2f} → {bpm} BPM")
    
    print("   [65%] 🔥 Computing musical intensity...")
    intensity = float(min(1.0, 0.5 * feats.edge_density + 0.5 * feats.contrast))
    print(f"   💫 Edges + contrast → intensity {intensity:.2f}")
    
    print("   [75%] 📍 Mapping brightness center to spatial parameters...")
    # Map center of mass to pan and lead offset  
    pan_lead = float((feats.cx - 0.5) * 1.2)  # [-0.6, 0.6]
    pan_lead = max(-0.6, min(0.6, pan_lead))
    
    lead_offset = int((feats.cy - 0.5) * 10)  # [-5, +5] 
    lead_offset = max(-5, min(5, lead_offset))
    
    print(f"   🎚️  Center ({feats.cx:.2f}, {feats.cy:.2f}) → pan {pan_lead:.2f}, offset {lead_offset:+d}")

    print(f"   [85%] 🎭 Applying {style} style effects...")
    # Create legacy scale field and apply style modifications
    scale = "major" if mode in ["ionian", "lydian", "mixolydian"] else "minor"
    
    if style == "ambient":
        old_bpm = bpm
        bpm = max(60, bpm - 20)
        scale = "major"  # Force major for ambient
        print(f"   🌅 Ambient: {old_bpm} → {bpm} BPM, forced major scale")
    elif style == "cinematic":
        old_bpm = bpm
        bpm = min(150, bpm + 10)
        print(f"   🎬 Cinematic: {old_bpm} → {bpm} BPM boost")
    elif style == "rock":
        old_bpm = bpm
        bpm = min(160, bpm + 20)
        scale = "minor"  # Force minor for rock
        print(f"   🤘 Rock: {old_bpm} → {bpm} BPM, forced minor scale")
    else:
        print(f"   ⚖️  Neutral: keeping original mappings")

    instruments = ["pad","lead","bass"] if style in ("ambient","cinematic") else ["piano","lead","drums"]
    
    print(f"   [100%] ✨ Musical mapping complete!")
    print(f"   🎵 Core: {root} {mode}, {meter[0]}/{meter[1]}, {bpm} BPM")
    print(f"   🎹 Progression: {' - '.join(progression)}")
    print(f"   🎚️  Spatial: pan {pan_lead:+.2f}, offset {lead_offset:+d}")
    print(f"   🎺 Instruments: {', '.join(instruments)}")
    
    return MusicParams(
        bpm=bpm,
        scale=f"{root}_{scale}",  # Legacy field
        root=root,
        instruments=instruments,  # Legacy
        intensity=intensity,
        duration=target_duration,
        mode=mode,
        meter=meter,
        progression=progression,
        pan_lead=pan_lead,  # Legacy
        lead_offset=lead_offset,  # Legacy
        voices=voices,
        has_complement=has_complement,
        chord_enrichment_level=chord_enrichment_level,
        texture_energy=feats.texture_energy
    )