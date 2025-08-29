import numpy as np, soundfile as sf
from typing import List
from .compose import Note

def _midi_to_freq(m): return 440.0 * 2 ** ((m - 69) / 12)

def render_wav(notes: List[Note], sr: int, out_path):
    dur = max(n.start + n.dur for n in notes) + 0.5
    y = np.zeros(int(sr * dur), dtype=np.float32)

    for n in notes:
        start = int(sr * n.start)
        length = int(sr * n.dur)
        if length <= 0: continue

        if n.track == "drums":
            env = np.linspace(1.0, 0.0, length, dtype=np.float32)
            sig = (np.random.randn(length).astype(np.float32) * 0.25 * env * n.vel)
        else:
            f = _midi_to_freq(n.midi)
            t = np.arange(length) / sr
            env = np.minimum(1.0, t / 0.01) * np.exp(-t / 0.4)
            sig = (np.sin(2*np.pi*f*t) + 0.3*np.sin(2*np.pi*2*f*t)) * env * n.vel * 0.2
            sig = sig.astype(np.float32)
        y[start:start+length] += sig

    mx = float(np.max(np.abs(y)) or 1.0)
    y = (y / mx * 0.98).astype(np.float32)
    sf.write(str(out_path), y, sr)
