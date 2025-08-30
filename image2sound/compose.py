from dataclasses import dataclass
from typing import List
import math
import numpy as np
from .mapping import MusicParams, VoiceSpec, MODES

@dataclass
class Note:
    """A musical note with timing, pitch, and performance attributes.
    
    Attributes:
        start: Start time in seconds
        dur: Duration in seconds
        midi: MIDI note number (60 = middle C)
        vel: Velocity/volume [0,1]
        track: Track name for grouping notes
        pan: Stereo pan position [-1.0, 1.0] where -1=left, 0=center, 1=right
    """
    start: float
    dur: float
    midi: int
    vel: float
    track: str
    pan: float = 0.0

def _scale_midi(root: str, mode: str) -> list[int]:
    """Build a scale from root note using the specified mode.
    
    Args:
        root: Root note name (C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B)
        mode: Musical mode name from MODES dictionary
        
    Returns:
        List of MIDI note numbers in the scale, centered around middle C (60)
    """
    names = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
    root_ix = names.index(root)
    pattern = MODES.get(mode, MODES["ionian"])  # Fallback to ionian (major)
    return [60 + ((root_ix + semitone) % 12) for semitone in pattern]


def _apply_mode_bias(scale_notes: list[int], mode_bias: float, beat: int, rng: np.random.Generator) -> int:
    """Select a note from the scale with mode bias for passing tones.
    
    Args:
        scale_notes: List of MIDI notes in the current scale
        mode_bias: Bias for passing tones [-1.0, 1.0] where -1=avoid, +1=prefer
        beat: Current beat number for pattern selection
        rng: Random number generator for choices
        
    Returns:
        MIDI note number
    """
    # Base pattern: cycle through scale degrees
    base_note = scale_notes[beat % len(scale_notes)]
    
    # Apply mode bias for passing tones (chromatic approach)
    if abs(mode_bias) > 0.3 and rng.random() < abs(mode_bias) * 0.5:
        if mode_bias > 0:  # Prefer passing tones
            # Add chromatic approach (+1 or -1 semitone)
            offset = rng.choice([-1, 1])
            return base_note + offset
        else:  # Avoid passing tones, stick to scale
            return base_note
    
    return base_note


def _compose_voice_track(voice: VoiceSpec, params: MusicParams, voice_id: int, rng: np.random.Generator) -> List[Note]:
    """Compose a track for a specific voice based on its specifications.
    
    Args:
        voice: Voice specification with instrument and performance parameters
        params: Global musical parameters (key, BPM, etc.)
        voice_id: Unique identifier for this voice
        rng: Seeded random number generator
        
    Returns:
        List of Note objects for this voice
    """
    scale_notes = _scale_midi(params.root, params.mode)
    spb = 60.0 / params.bpm  # Seconds per beat
    total_beats = int(params.duration / spb)
    
    # Apply activity scaling to note density
    beat_interval = max(1, int(1.0 / voice.activity))  # How often to place notes
    
    # Calculate base register with octave offset
    base_register = 60 + (voice.octave * 12)  # Middle C + octave shifts
    
    notes = []
    track_name = f"voice_{voice_id}_{voice.instrument}"
    
    beat = 0
    while beat < total_beats:
        t = beat * spb
        
        # Select note with mode bias
        midi_note = _apply_mode_bias(scale_notes, voice.mode_bias, beat, rng)
        
        # Transpose to voice's register
        midi_note = midi_note - 60 + base_register
        
        # Ensure note is in valid MIDI range
        midi_note = np.clip(midi_note, 21, 108)  # Piano range
        
        # Note duration varies with activity
        base_duration = spb * (0.8 + 0.4 * rng.random())  # 80-120% of beat
        if voice.activity > 1.5:  # High activity = shorter notes
            base_duration *= 0.7
        elif voice.activity < 0.7:  # Low activity = longer notes
            base_duration *= 1.4
            
        # Velocity scales with voice gain
        velocity = voice.gain * (0.7 + 0.3 * rng.random())  # Add slight variation
        
        notes.append(Note(
            start=t,
            dur=base_duration,
            midi=int(midi_note),
            vel=velocity,
            track=track_name,
            pan=voice.pan
        ))
        
        # Advance beat based on activity
        beat += beat_interval
    
    return notes

def compose_track(p: MusicParams) -> List[Note]:
    """Compose a multi-voice musical arrangement from parameters.
    
    Creates one track per voice, with each voice having its own instrument,
    register, activity level, and musical behavior based on color properties.
    
    Args:
        p: Musical parameters including voices, BPM, key, duration, etc.
        
    Returns:
        List of Note objects sorted by start time
    """
    print(f"🎼 Composing multi-voice arrangement...")
    print(f"   🎵 Key: {p.root} {p.mode}, {p.bpm} BPM, {p.duration:.1f}s duration")
    print(f"   🎤 Voices: {len(p.voices)} color-derived instruments")
    
    # Create seeded RNG for composition choices
    rng = np.random.default_rng(hash(p.root + p.mode + str(p.bpm)) & 0xFFFFFFFF)
    
    spb = 60.0 / p.bpm  # seconds per beat
    beats = int(p.duration / spb)
    
    print(f"   📏 Timing: {spb:.3f}s per beat, {beats} total beats")
    print(f"   🎹 Scale: {p.root} {p.mode} ({MODES[p.mode]})")

    all_notes: List[Note] = []
    voice_note_counts = []

    # Compose track for each voice
    for i, voice in enumerate(p.voices):
        print(f"   [{int((i+1)/len(p.voices)*80)}%] 🎵 Composing voice {i+1}: {voice.instrument}...")
        
        voice_notes = _compose_voice_track(voice, p, i+1, rng)
        all_notes.extend(voice_notes)
        voice_note_counts.append(len(voice_notes))
        
        # Show voice details
        color = voice.color
        print(f"      🎨 Color: RGB{color.rgb} (prop={color.prop:.2f}, pos=({color.cx:.2f},{color.cy:.2f}))")
        print(f"      🎶 Notes: {len(voice_notes)}, gain={voice.gain:.2f}, pan={voice.pan:+.2f}, octave={voice.octave:+d}")

    print(f"   [100%] ✨ Composition complete!")
    print(f"   📊 Generated {len(all_notes)} total notes across {len(p.voices)} voices:")
    for i, (voice, count) in enumerate(zip(p.voices, voice_note_counts)):
        print(f"   🎵 Voice {i+1} ({voice.instrument}): {count} notes")
    
    # Sort by start time
    sorted_notes = sorted(all_notes, key=lambda n: n.start)
    print(f"   🎼 Notes arranged chronologically")
    return sorted_notes
