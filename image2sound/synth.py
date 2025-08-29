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
    print(f"🎚️  Preparing audio synthesis...")
    
    # Calculate total duration with 0.5s buffer
    dur = max(n.start + n.dur for n in notes) + 0.5
    y = np.zeros(int(sr * dur), dtype=np.float32)
    
    print(f"   📏 Audio buffer: {dur:.2f}s at {sr}Hz ({len(y):,} samples)")
    
    # Group notes by track for progress reporting
    tracks = {}
    for n in notes:
        if n.track not in tracks:
            tracks[n.track] = []
        tracks[n.track].append(n)
    
    print(f"   🎵 Synthesizing {len(notes)} notes across {len(tracks)} tracks...")
    
    total_notes = len(notes)
    processed = 0
    
    for n in notes:
        start = int(sr * n.start)
        length = int(sr * n.dur)
        if length <= 0: 
            processed += 1
            continue

        # Show progress every 25% of notes
        if total_notes >= 4 and processed % (total_notes // 4) == 0 and processed > 0:
            progress = int((processed / total_notes) * 100)
            print(f"   [{progress}%] 🎼 Synthesizing note {processed}/{total_notes}...")

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
        processed += 1
    
    print(f"   [90%] 🔊 Applying audio limiter...")
    # Simple limiter to 0.98 peak
    mx = float(np.max(np.abs(y)) or 1.0)
    y = (y / mx * 0.98).astype(np.float32)
    print(f"   📊 Peak level: {mx:.3f} → 0.98 (normalized)")
    
    print(f"   [95%] 💾 Writing WAV file...")
    # Write WAV file
    sf.write(str(out_path), y, sr)
    
    # File size for user feedback
    import os
    file_size = os.path.getsize(out_path)
    size_mb = file_size / (1024 * 1024)
    
    print(f"   [100%] ✅ Audio synthesis complete!")
    print(f"   🎵 Track breakdown:")
    for track_name, track_notes in tracks.items():
        print(f"   🎸 {track_name}: {len(track_notes)} notes synthesized")
    print(f"   📁 Output: {out_path} ({size_mb:.1f}MB)")
