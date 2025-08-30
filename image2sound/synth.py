import numpy as np
import wave
from typing import List
from .compose import Note

# Try to import soundfile, fallback to built-in wave module
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

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


def _apply_simple_reverb(signal: np.ndarray, reverb_amount: float, sr: int) -> np.ndarray:
    """Apply simple reverb using multiple delay lines.
    
    Args:
        signal: Input signal
        reverb_amount: Reverb amount [0,1] 
        sr: Sample rate
        
    Returns:
        Signal with reverb applied
    """
    if reverb_amount <= 0.01:
        return signal
    
    # Simple reverb with multiple delays
    delay_times = [0.03, 0.05, 0.07, 0.11]  # Delay times in seconds
    delay_gains = [0.3, 0.25, 0.2, 0.15]    # Decay factors
    
    reverb_signal = np.zeros_like(signal)
    
    for delay_time, gain in zip(delay_times, delay_gains):
        delay_samples = int(delay_time * sr)
        if delay_samples < len(signal):
            # Create delay buffer
            delayed = np.zeros_like(signal)
            delayed[delay_samples:] = signal[:-delay_samples] * gain * reverb_amount
            reverb_signal += delayed
    
    return signal + reverb_signal


def _apply_stereo_widener(left: np.ndarray, right: np.ndarray, width_amount: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply ultralight stereo widening using delayed mid/side processing.
    
    Args:
        left: Left channel signal
        right: Right channel signal  
        width_amount: Widening amount [0,1]
        sr: Sample rate
        
    Returns:
        (widened_left, widened_right) tuple
    """
    if width_amount <= 0.01:
        return left, right
    
    # Convert to mid/side
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    
    # Apply tiny delay to side channel (1-3ms for subtle effect)
    delay_ms = 1.5  # Very small delay for widening
    delay_samples = int(delay_ms * sr / 1000)
    
    if delay_samples > 0 and delay_samples < len(side):
        # Delayed and inverted side for widening effect
        delayed_side = np.zeros_like(side)
        delayed_side[delay_samples:] = -side[:-delay_samples]  # Inverted
        
        # Mix original and delayed side based on width amount
        enhanced_side = side + delayed_side * width_amount * 0.3
        
        # Convert back to left/right
        widened_left = mid + enhanced_side
        widened_right = mid - enhanced_side
        
        return widened_left, widened_right
    
    return left, right


def _synthesize_instrument(instrument: str, f: float, t: np.ndarray, brightness: float, saturation: float = 0.5, value: float = 0.5) -> np.ndarray:
    """Synthesize audio signal with color-driven spectral control.
    
    Args:
        instrument: Instrument type (pluck, bell, marimba, pad_glass, pad_warm, lead_clean, brass_short)
        f: Fundamental frequency in Hz
        t: Time array for the note duration
        brightness: Filter brightness [0,1] where 1.0 = open filter (legacy)
        saturation: Color saturation [0,1] controls harmonic content
        value: Color value/brightness [0,1] controls filtering and reverb
        
    Returns:
        Raw audio signal (before envelope)
    """
    if instrument == "pluck":
        # Plucked string: fundamental + saturation-controlled harmonics
        fundamental = np.sin(2 * np.pi * f * t)
        signal = fundamental
        
        # Add 3rd and 5th harmonics based on saturation
        if saturation > 0.3:
            third_harmonic = np.sin(2 * np.pi * f * 3 * t) * (saturation - 0.3) * 0.4
            signal += third_harmonic
        if saturation > 0.6:
            fifth_harmonic = np.sin(2 * np.pi * f * 5 * t) * (saturation - 0.6) * 0.3
            signal += fifth_harmonic
        
    elif instrument == "bell":
        # Bell-like: fundamental + progressive harmonics based on saturation
        fundamental = np.sin(2 * np.pi * f * t)
        second_harmonic = np.sin(2 * np.pi * f * 2 * t) * 0.5  # Always present
        signal = fundamental + second_harmonic
        
        # Add higher harmonics based on saturation
        if saturation > 0.2:
            third_harmonic = np.sin(2 * np.pi * f * 3 * t) * saturation * 0.4
            signal += third_harmonic
        if saturation > 0.5:
            fifth_harmonic = np.sin(2 * np.pi * f * 5 * t) * (saturation - 0.5) * 0.3
            signal += fifth_harmonic
        
    elif instrument == "marimba":
        # Marimba: triangle base + saturation-controlled even harmonics
        triangle = 2.0 * np.abs(2 * (f * t - np.floor(f * t + 0.5))) - 1.0
        signal = triangle
        
        # Add even harmonics based on saturation
        if saturation > 0.1:
            second_harmonic = np.sin(2 * np.pi * f * 2 * t) * saturation * 0.5
            signal += second_harmonic
        if saturation > 0.4:
            fourth_harmonic = np.sin(2 * np.pi * f * 4 * t) * (saturation - 0.4) * 0.3
            signal += fourth_harmonic
        
    elif instrument == "pad_glass":
        # Glass pad: detuned oscillators + saturation-controlled harmonics
        fundamental = np.sin(2 * np.pi * f * t)
        detune_up = np.sin(2 * np.pi * f * 1.005 * t) * 0.7    # Slight detune up
        detune_down = np.sin(2 * np.pi * f * 0.995 * t) * 0.5  # Slight detune down
        signal = fundamental + detune_up + detune_down
        
        # Add shimmer harmonics based on saturation (higher frequencies for glass-like timbre)
        if saturation > 0.3:
            shimmer = np.sin(2 * np.pi * f * 7 * t) * (saturation - 0.3) * 0.2
            signal += shimmer
        if saturation > 0.7:
            high_shimmer = np.sin(2 * np.pi * f * 11 * t) * (saturation - 0.7) * 0.1
            signal += high_shimmer
        
    elif instrument == "pad_warm":
        # Warm pad: saw wave + saturation-controlled odd harmonics
        saw = 2 * (f * t - np.floor(f * t + 0.5))
        signal = saw * 0.6  # Reduce base level to accommodate harmonics
        
        # Add warm odd harmonics based on saturation
        if saturation > 0.2:
            third_harmonic = np.sin(2 * np.pi * f * 3 * t) * saturation * 0.4
            signal += third_harmonic
        if saturation > 0.5:
            fifth_harmonic = np.sin(2 * np.pi * f * 5 * t) * (saturation - 0.5) * 0.3
            signal += fifth_harmonic
        if saturation > 0.8:
            seventh_harmonic = np.sin(2 * np.pi * f * 7 * t) * (saturation - 0.8) * 0.2
            signal += seventh_harmonic
        
    elif instrument == "lead_clean":
        # Clean lead: sine base + saturation-controlled harmonics for edge
        fundamental = np.sin(2 * np.pi * f * t)
        signal = fundamental
        
        # Add slight square-like harmonics based on saturation
        if saturation > 0.1:
            third_harmonic = np.sin(2 * np.pi * f * 3 * t) * saturation * 0.3
            signal += third_harmonic
        if saturation > 0.4:
            fifth_harmonic = np.sin(2 * np.pi * f * 5 * t) * (saturation - 0.4) * 0.2
            signal += fifth_harmonic
        if saturation > 0.7:
            seventh_harmonic = np.sin(2 * np.pi * f * 7 * t) * (saturation - 0.7) * 0.15
            signal += seventh_harmonic
        
    elif instrument == "brass_short":
        # Brass: saw base + saturation-controlled odd harmonics for brassy bite
        saw = 2 * (f * t - np.floor(f * t + 0.5))
        signal = saw * 0.7  # Reduce base to accommodate harmonics
        
        # Add brass-like odd harmonics based on saturation
        if saturation > 0.1:
            third_harmonic = np.sin(2 * np.pi * f * 3 * t) * saturation * 0.5
            signal += third_harmonic
        if saturation > 0.4:
            fifth_harmonic = np.sin(2 * np.pi * f * 5 * t) * (saturation - 0.4) * 0.4
            signal += fifth_harmonic
        if saturation > 0.7:
            seventh_harmonic = np.sin(2 * np.pi * f * 7 * t) * (saturation - 0.7) * 0.3
            signal += seventh_harmonic
        
    else:
        # Fallback: simple sine
        signal = np.sin(2 * np.pi * f * t)
    
    # Apply value-controlled filtering to all instruments
    # Calculate cutoff frequency based on color value (brightness)
    base_cutoff = f * 1.5  # Start at 1.5x fundamental (more conservative)
    max_cutoff = min(12000, f * 12)  # Higher max for brighter sounds
    cutoff = base_cutoff + (max_cutoff - base_cutoff) * value
    
    # Apply filtering - more aggressive for harmonically rich instruments
    filter_intensity = 1.0
    if instrument in ["pad_warm", "brass_short"]:
        filter_intensity = 0.8  # More filtering for saw-based instruments
    elif instrument in ["pad_glass", "bell"]:
        filter_intensity = 0.3  # Light filtering for pure tones
    else:
        filter_intensity = 0.6  # Medium filtering for others
    
    # Mix filtered and unfiltered signal based on filter intensity
    filtered_signal = _apply_1pole_lpf(signal, cutoff, 44100)
    signal = signal * (1 - filter_intensity) + filtered_signal * filter_intensity
    
    return signal


def _get_envelope(instrument: str, length: int, sr: int) -> np.ndarray:
    """Generate ADSR envelope appropriate for the instrument type.
    
    Args:
        instrument: Instrument type
        length: Length of envelope in samples
        sr: Sample rate
        
    Returns:
        Envelope array [0,1]
    """
    t = np.arange(length) / sr
    
    if instrument == "pluck":
        # Quick attack, exponential decay
        attack_time = 0.01  # 10ms
        env = np.minimum(1.0, t / attack_time) * np.exp(-t / 0.3)
        
    elif instrument == "bell":
        # Medium attack, long decay
        attack_time = 0.05  # 50ms
        env = np.minimum(1.0, t / attack_time) * np.exp(-t / 0.8)
        
    elif instrument == "marimba":
        # Quick attack, medium decay
        attack_time = 0.02  # 20ms
        env = np.minimum(1.0, t / attack_time) * np.exp(-t / 0.4)
        
    elif instrument in ["pad_glass", "pad_warm"]:
        # Slow attack, sustained
        attack_time = 0.15  # 150ms
        decay_time = 0.8
        env = np.minimum(1.0, t / attack_time) * np.exp(-t / decay_time)
        
    elif instrument == "lead_clean":
        # Medium attack, controlled decay
        attack_time = 0.03  # 30ms
        env = np.minimum(1.0, t / attack_time) * np.exp(-t / 0.5)
        
    elif instrument == "brass_short":
        # Quick attack, quick decay (staccato)
        attack_time = 0.02  # 20ms
        env = np.minimum(1.0, t / attack_time) * np.exp(-t / 0.25)
        
    else:
        # Default envelope
        attack_time = 0.01
        env = np.minimum(1.0, t / attack_time) * np.exp(-t / 0.4)
    
    return env


def write_wav_pcm16(path: str, y: np.ndarray, sr: int) -> None:
    """Write audio data to WAV file using built-in wave module.
    
    Args:
        path: Output file path
        y: Audio data array, shape (samples,) for mono or (samples, 2) for stereo
        sr: Sample rate in Hz
    """
    # Handle mono vs stereo
    if y.ndim == 1:
        # Mono
        nchannels = 1
        frames = y
    elif y.ndim == 2 and y.shape[1] == 2:
        # Stereo - interleave channels
        nchannels = 2
        frames = y  # Already in correct format
    else:
        raise ValueError(f"Unsupported audio shape: {y.shape}. Expected (samples,) or (samples, 2)")
    
    # Clip to [-1, 1] range
    frames = np.clip(frames, -1.0, 1.0)
    
    # Convert to 16-bit PCM
    frames_int16 = (frames * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(2)  # 2 bytes = 16 bits
        wf.setframerate(sr)
        
        if nchannels == 1:
            # Mono - write directly
            wf.writeframes(frames_int16.tobytes())
        else:
            # Stereo - interleave and write
            interleaved = frames_int16.flatten()
            wf.writeframes(interleaved.tobytes())


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
    """Render a list of notes to a stereo WAV audio file with voice-based synthesis.
    
    Synthesizes musical notes using voice-specific instruments:
    - pluck: Sine + triangle with quick decay
    - bell: Sine with harmonic series (bell-like overtones)
    - marimba: Triangle with even harmonics
    - pad_glass: Pure sine waves with slight detuning
    - pad_warm: Filtered saw wave with warmth
    - lead_clean: Sine + small square wave
    - brass_short: Saw with emphasized odd harmonics
    
    Each instrument has appropriate ADSR envelope and brightness-controlled filtering.
    
    Args:
        notes: List of Note objects with voice-specific track names
        sr: Sample rate in Hz (e.g., 44100)
        out_path: Output file path for WAV file
        
    Output:
        Writes stereo WAV file with limiter at 0.98 peak:
        - 32-bit float if soundfile is available
        - 16-bit PCM if using built-in wave module fallback
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

        # Initialize color properties for all notes
        color_value = 0.5  # Default value for non-voice tracks
        is_pad = False
        
        # Parse track name to extract instrument and voice color properties
        if n.track.startswith("voice_"):
            # Format: "voice_X_instrumentname" 
            parts = n.track.split("_")
            if len(parts) >= 3:
                instrument = parts[2]
                voice_id = int(parts[1])  # Extract voice ID for color mapping
                is_pad = "pad" in instrument
            else:
                instrument = "pluck"  # Fallback
                voice_id = 0
            
            # Derive color properties from note characteristics
            # Since we don't have direct access to VoiceSpec here, we'll use heuristics
            # Higher velocity often correlates with higher saturation
            saturation = min(1.0, n.vel * 1.5)  # Scale velocity to saturation
            
            # Use MIDI note position to derive brightness (higher notes = brighter)
            # Middle C (60) as baseline
            color_value = 0.3 + 0.7 * min(1.0, max(0.0, (n.midi - 48) / 48.0))  # Map MIDI 48-96 to value 0.3-1.0
            
            # Legacy brightness for backward compatibility
            brightness = n.vel  # vel already represents gain, use as brightness too
            
            # Generate instrument-specific signal
            f = _midi_to_freq(n.midi)
            t = np.arange(length) / sr
            
            # Synthesize the instrument with color control
            sig = _synthesize_instrument(instrument, f, t, brightness, saturation, color_value)
            
            # Apply appropriate envelope
            env = _get_envelope(instrument, length, sr)
            
            # Apply envelope and velocity
            sig = sig * env * n.vel * 0.12
            
            # Apply reverb based on color value (brighter = more reverb)
            reverb_amount = color_value * 0.3  # Scale to 0-30% reverb
            sig = _apply_simple_reverb(sig, reverb_amount, sr)
            
            sig = sig.astype(np.float32)
            
        elif n.track.startswith("transition_"):
            # Transition effects
            t = np.arange(length) / sr
            
            if n.track == "transition_swell":
                # Cymbal swell: noise burst with LPF sweep
                noise = np.random.randn(length).astype(np.float32)
                
                # Create LPF sweep from high to low frequency
                sweep_start_freq = 8000  # Start at 8kHz
                sweep_end_freq = 200     # End at 200Hz
                sweep_freqs = sweep_start_freq * np.exp(-t * 3)  # Exponential sweep down
                
                # Apply time-varying LPF (approximated)
                sig = noise * 0.3 * n.vel
                # Apply gentle envelope for swell effect
                swell_env = np.minimum(1.0, t * 3) * np.exp(-t * 0.5)
                sig = sig * swell_env
                
            elif n.track == "transition_fill":
                # Drum fills: enhanced drum hits
                if n.midi == 36:  # Kick
                    # Enhanced kick with pitch sweep
                    pitch_env = np.exp(-t * 40)
                    freq = 80 * pitch_env + 40
                    kick = np.sin(2 * np.pi * freq * t) * 0.4
                    noise = np.random.randn(length) * 0.1
                    sig = (kick + noise) * n.vel
                elif n.midi == 38:  # Snare
                    # Snare with noise and tone
                    tone = np.sin(2 * np.pi * 250 * t) * 0.3
                    noise = np.random.randn(length) * 0.4
                    sig = (tone + noise) * n.vel
                else:  # Hi-hat (42)
                    # Hi-hat as filtered noise
                    noise = np.random.randn(length) * 0.2
                    sig = _apply_1pole_lpf(noise, 8000, sr) * n.vel
                
                # Quick decay for drum fills
                fill_env = np.exp(-t * 10)
                sig = sig * fill_env
                
            else:
                # Unknown transition type, fallback to noise
                sig = np.random.randn(length).astype(np.float32) * 0.1 * n.vel
                
            sig = sig.astype(np.float32)
            
        elif n.track == "drums":
            # Legacy drums support
            env = np.linspace(1.0, 0.0, length, dtype=np.float32)
            sig = np.random.randn(length).astype(np.float32) * 0.25 * env * n.vel
            
            # Add pitch envelope for kick (MIDI 36)
            if n.midi == 36:
                t = np.arange(length) / sr
                pitch_env = np.exp(-t * 30)  # Quick pitch drop
                freq = 60 * pitch_env + 40   # 60Hz dropping to 40Hz
                kick_tone = np.sin(2 * np.pi * freq * t) * 0.3
                sig += kick_tone.astype(np.float32)
        
        else:
            # Legacy track support (lead, chords, bass, etc.)
            f = _midi_to_freq(n.midi)
            t = np.arange(length) / sr
            
            if n.track == "lead":
                # Vibrato for lead
                vibrato = np.sin(2 * np.pi * 5.0 * t) * 0.01
                f_vibrato = f * (1.0 + vibrato)
                sig = np.sin(2 * np.pi * f_vibrato * t)
                env = np.minimum(1.0, t / 0.01) * np.exp(-t / 0.4)
                # Store for delay processing
                lead_notes.append((start, length, (sig * env * n.vel * 0.15).copy()))
            else:
                # Simple sine for other legacy tracks
                sig = np.sin(2*np.pi*f*t)
                env = np.minimum(1.0, t / 0.01) * np.exp(-t / 0.4)
            
            sig = sig * env * n.vel * 0.15
            sig = sig.astype(np.float32)
        
        # Apply stereo panning
        pan = getattr(n, 'pan', 0.0)  # Default to center if no pan
        left_gain, right_gain = _equal_power_pan(pan)
        
        # Create initial stereo signals
        left_sig = sig * left_gain
        right_sig = sig * right_gain
        
        # Apply stereo widening to pad instruments when brightness is high
        if is_pad and color_value > 0.6:
            # Only widen pads when color_value (brightness) > 0.6 for airy feel
            width_amount = (color_value - 0.6) * 2.5  # Map 0.6-1.0 to 0.0-1.0
            left_sig, right_sig = _apply_stereo_widener(left_sig, right_sig, width_amount, sr)
        
        # Mix into stereo buffer
        y[start:start+length, 0] += left_sig   # Left channel
        y[start:start+length, 1] += right_sig  # Right channel
        
        processed += 1
    
    # Apply delay to legacy lead notes (if any)
    if lead_notes:
        print(f"   🎸 Applying delay to {len(lead_notes)} legacy lead notes...")
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
    
    # Use soundfile if available (32-bit float), otherwise use built-in wave (16-bit PCM)
    if HAS_SOUNDFILE:
        # Write 32-bit float stereo WAV file with soundfile
        sf.write(str(out_path), y, sr)
        print(f"   🎵 Format: 32-bit float WAV via soundfile")
    else:
        # Write 16-bit PCM stereo WAV file with built-in wave module
        write_wav_pcm16(str(out_path), y, sr)
        print(f"   🎵 Format: 16-bit PCM WAV via built-in wave module")
    
    # File size for user feedback
    import os
    file_size = os.path.getsize(out_path)
    size_mb = file_size / (1024 * 1024)
    
    print(f"   [100%] ✅ Audio synthesis complete!")
    print(f"   🎵 Track breakdown:")
    for track_name, track_notes in tracks.items():
        print(f"   🎸 {track_name}: {len(track_notes)} notes synthesized")
    print(f"   📁 Output: {out_path} ({size_mb:.1f}MB)")