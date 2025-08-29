import numpy as np, soundfile as sf
from typing import List
from .compose import Note

def _midi_to_freq(m: int) -> float:
    """Convert MIDI note number to frequency in Hz.
    
    Args:
        m: MIDI note number (69 = A4 = 440Hz)
        
    Returns:
        Frequency in Hz
    """
    return 440.0 * 2 ** ((m - 69) / 12)

def render_wav(notes: List[Note], sr: int, out_path) -> None:
    """Render a list of notes to a WAV audio file using basic synthesis.
    
    Synthesizes musical notes using different approaches by track type:
    - Tonal tracks: Sine wave + second harmonic with ADSR envelope
      * Attack: 10ms linear rise
      * Decay: Exponential with tau ≈ 0.4s
      * Mixed at ~-6dB (0.2 gain)
    - Drums: Filtered noise burst with quick linear decay
    
    Args:
        notes: List of Note objects to synthesize
        sr: Sample rate in Hz (e.g., 44100)
        out_path: Output file path for WAV file
        
    Output:
        Writes 32-bit float WAV file with simple limiter at 0.98 peak
    """
    # Calculate total duration with 0.5s buffer
    dur = max(n.start + n.dur for n in notes) + 0.5
    y = np.zeros(int(sr * dur), dtype=np.float32)

    for n in notes:
        start = int(sr * n.start)
        length = int(sr * n.dur)
        if length <= 0: 
            continue

        if n.track == "drums":
            # Drums: noise burst with quick linear decay
            env = np.linspace(1.0, 0.0, length, dtype=np.float32)
            sig = (np.random.randn(length).astype(np.float32) * 0.25 * env * n.vel)
        else:
            # Tonal: sine + 0.3*sin(2f) with ADSR envelope
            f = _midi_to_freq(n.midi)
            t = np.arange(length) / sr
            
            # ADSR: attack=10ms, decay tau~0.4s
            env = np.minimum(1.0, t / 0.01) * np.exp(-t / 0.4)
            
            # Sine wave + second harmonic, mixed at ~-6dB
            sig = (np.sin(2*np.pi*f*t) + 0.3*np.sin(2*np.pi*2*f*t)) * env * n.vel * 0.2
            sig = sig.astype(np.float32)
            
        # Sum into buffer
        y[start:start+length] += sig

    # Simple limiter to 0.98 peak
    mx = float(np.max(np.abs(y)) or 1.0)
    y = (y / mx * 0.98).astype(np.float32)
    
    # Write WAV file
    sf.write(str(out_path), y, sr)
