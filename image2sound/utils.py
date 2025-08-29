"""Utility functions for image2sound package."""

import hashlib
from pathlib import Path
import numpy as np


def rng_from_file(path: Path) -> np.random.Generator:
    """Create a seeded random number generator from file contents.
    
    Uses SHA-256 hash of the file bytes to derive a deterministic 32-bit seed,
    ensuring the same image always produces the same random sequence.
    
    Args:
        path: Path to the file to hash
        
    Returns:
        NumPy random generator seeded with file hash
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If the file cannot be read
    """
    # Read file bytes and compute SHA-256 hash
    file_bytes = path.read_bytes()
    hash_digest = hashlib.sha256(file_bytes).digest()
    
    # Extract first 4 bytes and convert to 32-bit unsigned integer
    seed = int.from_bytes(hash_digest[:4], byteorder='big', signed=False)
    
    # Return seeded generator
    return np.random.default_rng(seed)


def get_file_seed(path: Path) -> int:
    """Get the 32-bit seed derived from file hash.
    
    Args:
        path: Path to the file to hash
        
    Returns:
        32-bit unsigned integer seed
    """
    file_bytes = path.read_bytes()
    hash_digest = hashlib.sha256(file_bytes).digest()
    return int.from_bytes(hash_digest[:4], byteorder='big', signed=False)