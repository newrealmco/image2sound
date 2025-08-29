"""Image to sound conversion through algorithmic sonification."""

from .features import ImageFeatures, extract_features
from .mapping import MusicParams, map_features_to_music
from .compose import Note, compose_track
from .synth import render_wav
from .utils import rng_from_file, get_file_seed

__version__ = "0.1.0"

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