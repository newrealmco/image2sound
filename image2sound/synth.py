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


def _equal_power_pan(pan: float) -> tuple[float, float]:
    """Calculate equal-power stereo pan gains.
    
    Args:
        pan: Pan position [-1.0, 1.0] where -1=left, 0=center, 1=right
        
    Returns:
        (left_gain, right_gain) tuple with equal power panning
    """
    pan = np.clip(pan, -1.0, 1.0)
    # Equal power panning: -3dB at center
    angle = (pan + 1.0) * np.pi / 4.0  # Map [-1,1] to [0, π/2]
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)
    return left_gain, right_gain


def _apply_1pole_lpf(signal: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    """Apply 1-pole lowpass filter to soften saw waves.
    
    Args:
        signal: Input signal
        cutoff: Cutoff frequency in Hz
        sr: Sample rate
        
    Returns:
        Filtered signal
    """
    # Simple 1-pole LPF: y[n] = a*x[n] + (1-a)*y[n-1]
    a = 1.0 - np.exp(-2.0 * np.pi * cutoff / sr)
    y = np.zeros_like(signal)
    for i in range(len(signal)):
        y[i] = a * signal[i] + (1 - a) * (y[i-1] if i > 0 else 0)
    return y


def _circular_delay(signal: np.ndarray, delay_samples: int, feedback: float) -> np.ndarray:
    """Apply simple circular buffer delay effect.
    
    Args:
        signal: Input signal
        delay_samples: Delay time in samples
        feedback: Feedback amount [0.0, 1.0]
        
    Returns:
        Signal with delay effect
    """
    if delay_samples <= 0:
        return signal
        
    # Create circular delay buffer
    buffer = np.zeros(delay_samples)
    output = np.zeros_like(signal)
    
    for i in range(len(signal)):
        # Read from delay buffer
        delayed = buffer[i % delay_samples]
        
        # Mix dry + wet
        output[i] = signal[i] + 0.3 * delayed
        
        # Write to delay buffer with feedback
        buffer[i % delay_samples] = signal[i] + feedback * delayed
        
    return output


def render_wav(notes: List[Note], sr: int, out_path) -> None:
    """Render a list of notes to a stereo WAV audio file with enhanced synthesis.
    
    Synthesizes musical notes using track-specific timbres:
    - Lead: Sine + triangle with vibrato and delay
    - Chords/Pads: Softened saw wave with slower attack
    - Bass: Triangle wave with shorter decay  
    - Drums: Noise burst with pitch envelope for kicks
    
    Supports stereo panning with equal-power law (-3dB at center).
    
    Args:
        notes: List of Note objects to synthesize
        sr: Sample rate in Hz (e.g., 44100)
        out_path: Output file path for WAV file
        
    Output:
        Writes stereo 32-bit float WAV file with limiter at 0.98 peak
    """
    print(f"🎚️  Preparing audio synthesis...")
    
    # Calculate total duration with 0.5s buffer
    dur = max(n.start + n.dur for n in notes) + 0.5
    # Stereo buffer: [samples, 2]
    y = np.zeros((int(sr * dur), 2), dtype=np.float32)
    
    print(f"   📏 Stereo buffer: {dur:.2f}s at {sr}Hz ({len(y):,} samples)")
    
    # Group notes by track for progress reporting
    tracks = {}
    for n in notes:
        if n.track not in tracks:
            tracks[n.track] = []
        tracks[n.track].append(n)
    
    print(f"   🎵 Synthesizing {len(notes)} notes across {len(tracks)} tracks...")
    
    total_notes = len(notes)
    processed = 0
    
    # Group notes by track for lead delay processing
    lead_notes = []
    
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

        # Generate signal based on track type
        if n.track == "drums":
            # Drums: noise burst with pitch envelope for kicks
            env = np.linspace(1.0, 0.0, length, dtype=np.float32)
            sig = np.random.randn(length).astype(np.float32) * 0.25 * env * n.vel
            
            # Add pitch envelope for kick (MIDI 36)
            if n.midi == 36:
                t = np.arange(length) / sr
                pitch_env = np.exp(-t * 30)  # Quick pitch drop
                freq = 60 * pitch_env + 40   # 60Hz dropping to 40Hz
                kick_tone = np.sin(2 * np.pi * freq * t) * 0.3
                sig += kick_tone.astype(np.float32)
                
        elif n.track == "lead":
            # Lead: sine + triangle with vibrato
            f = _midi_to_freq(n.midi)
            t = np.arange(length) / sr
            
            # Vibrato: ~5Hz, ±10 cents
            vibrato = np.sin(2 * np.pi * 5.0 * t) * 0.01  # ±1% frequency
            f_vibrato = f * (1.0 + vibrato)
            
            # ADSR: attack=10ms, decay tau~0.4s
            env = np.minimum(1.0, t / 0.01) * np.exp(-t / 0.4)
            
            # Sine + small triangle
            sine = np.sin(2 * np.pi * f_vibrato * t)
            triangle = 2.0 * np.abs(2 * (f_vibrato * t - np.floor(f_vibrato * t + 0.5))) - 1.0
            sig = (sine + 0.2 * triangle) * env * n.vel * 0.15
            sig = sig.astype(np.float32)
            
            # Store for delay processing
            lead_notes.append((start, length, sig.copy()))
            
        elif n.track in ["chords", "pad"]:
            # Chords/pads: softened saw with slower attack
            f = _midi_to_freq(n.midi)
            t = np.arange(length) / sr
            
            # Slower attack: 80-120ms
            attack_time = 0.1  # 100ms
            env = np.minimum(1.0, t / attack_time) * np.exp(-t / 0.6)
            
            # Saw wave
            saw = 2 * (f * t - np.floor(f * t + 0.5))
            
            # Soften with 1-pole LPF
            sig = _apply_1pole_lpf(saw, f * 4, sr) * env * n.vel * 0.12
            sig = sig.astype(np.float32)
            
        elif n.track == "bass":
            # Bass: triangle with shorter decay
            f = _midi_to_freq(n.midi)
            t = np.arange(length) / sr
            
            # Shorter decay for bass
            env = np.minimum(1.0, t / 0.01) * np.exp(-t / 0.2)
            
            # Triangle wave
            triangle = 2.0 * np.abs(2 * (f * t - np.floor(f * t + 0.5))) - 1.0
            sig = triangle * env * n.vel * 0.18
            sig = sig.astype(np.float32)
            
        else:
            # Default tonal
            f = _midi_to_freq(n.midi)
            t = np.arange(length) / sr
            env = np.minimum(1.0, t / 0.01) * np.exp(-t / 0.4)
            sig = np.sin(2*np.pi*f*t) * env * n.vel * 0.15
            sig = sig.astype(np.float32)
        
        # Apply stereo panning
        pan = getattr(n, 'pan', 0.0)  # Default to center if no pan
        left_gain, right_gain = _equal_power_pan(pan)
        
        # Mix into stereo buffer
        y[start:start+length, 0] += sig * left_gain   # Left channel
        y[start:start+length, 1] += sig * right_gain  # Right channel
        
        processed += 1
    
    # Apply delay to lead notes
    if lead_notes:
        print(f"   🎸 Applying delay to {len(lead_notes)} lead notes...")
        # Calculate 1/8 note delay time (assuming 4/4 time)
        eighth_note = 60.0 / 120.0 / 2.0  # Default to 120 BPM, 1/8 note
        delay_samples = int(sr * eighth_note)
        
        for start, length, sig in lead_notes:
            # Apply delay
            delayed_sig = _circular_delay(sig, delay_samples, 0.2)
            
            # Get pan for this lead note (assuming center)
            left_gain, right_gain = _equal_power_pan(0.0)
            
            # Add delayed signal back to buffer
            y[start:start+length, 0] += (delayed_sig - sig) * left_gain * 0.3
            y[start:start+length, 1] += (delayed_sig - sig) * right_gain * 0.3
    
    print(f"   [90%] 🔊 Applying stereo limiter...")
    # Simple limiter to 0.98 peak across both channels
    mx = float(np.max(np.abs(y)) or 1.0)
    y = (y / mx * 0.98).astype(np.float32)
    print(f"   📊 Peak level: {mx:.3f} → 0.98 (normalized)")
    
    print(f"   [95%] 💾 Writing stereo WAV file...")
    # Write stereo WAV file
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