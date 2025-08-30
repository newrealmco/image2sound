"""Image to sound conversion through algorithmic sonification."""

__version__ = "0.1.0"

# Lazy loading to keep CLI --help fast while maintaining public API
def __getattr__(name: str):
    """Lazy load module components to avoid heavy imports during CLI --help."""
    if name == "ImageFeatures":
        from .features import ImageFeatures
        return ImageFeatures
    elif name == "extract_features":
        from .features import extract_features
        return extract_features
    elif name == "MusicParams":
        from .mapping import MusicParams
        return MusicParams
    elif name == "map_features_to_music":
        from .mapping import map_features_to_music
        return map_features_to_music
    elif name == "Note":
        from .compose import Note
        return Note
    elif name == "compose_track":
        from .compose import compose_track
        return compose_track
    elif name == "render_wav":
        from .synth import render_wav
        return render_wav
    elif name == "rng_from_file":
        from .utils import rng_from_file
        return rng_from_file
    elif name == "get_file_seed":
        from .utils import get_file_seed
        return get_file_seed
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "ImageFeatures",
    "extract_features", 
    "MusicParams",
    "map_features_to_music",
    "Note",
    "compose_track",
    "render_wav",
    "rng_from_file",
    "get_file_seed",
]