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


@dataclass
class Section:
    """A musical section with timing and voice activity.
    
    Attributes:
        name: Section name (A, B, A', Tutti)
        start_beat: Starting beat of the section
        end_beat: Ending beat of the section
        active_voices: List of voice indices that are active in this section
        density_multiplier: Overall density scaling for the section [0.1, 1.0]
        transition: Whether this section includes a transition effect
    """
    name: str
    start_beat: int
    end_beat: int
    active_voices: list[int]
    density_multiplier: float = 1.0
    transition: bool = False


def _create_section_structure(voices: list[VoiceSpec], total_beats: int, meter: tuple[int, int]) -> list[Section]:
    """Create A/B/A'/Tutti section structure based on top colors.
    
    Args:
        voices: List of voice specifications (sorted by color proportion)
        total_beats: Total number of beats in the composition
        meter: Time signature as (numerator, denominator)
        
    Returns:
        List of Section objects defining the structure
    """
    beats_per_bar = meter[0]
    total_bars = (total_beats + beats_per_bar - 1) // beats_per_bar  # Round up to nearest bar
    # Ensure minimum 2 bars for sectional structure, but don't force 4 bars for very short pieces
    total_bars = max(2, total_bars)
    total_beats = total_bars * beats_per_bar  # Align to bar boundaries
    
    # Determine section lengths based on total duration
    if total_bars <= 8:
        # Short piece: A-B-Tutti
        if total_bars <= 3:
            # Very short: just A and B, no Tutti
            a_bars = (total_bars + 1) // 2  # Round up
            b_bars = total_bars - a_bars
            sections_plan = [("A", a_bars), ("B", b_bars)]
        else:
            # Regular short piece: A-B-Tutti
            a_bars = max(1, total_bars // 3)
            b_bars = max(1, total_bars // 3)
            tutti_bars = total_bars - a_bars - b_bars
            sections_plan = [("A", a_bars), ("B", b_bars), ("Tutti", tutti_bars)]
    else:
        # Longer piece: A-B-A'-Tutti
        a_bars = max(2, total_bars // 4)
        b_bars = max(2, total_bars // 4)
        a_prime_bars = max(2, total_bars // 4)
        tutti_bars = total_bars - a_bars - b_bars - a_prime_bars
        sections_plan = [("A", a_bars), ("B", b_bars), ("A'", a_prime_bars), ("Tutti", tutti_bars)]
    
    # Create sections
    sections = []
    current_beat = 0
    
    for i, (section_name, bars) in enumerate(sections_plan):
        start_beat = current_beat
        end_beat = current_beat + (bars * beats_per_bar)
        
        if section_name == "A":
            # Section A: Primary voice (highest proportion) + light pad support
            active_voices = [0]  # Primary voice
            if len(voices) > 3:  # Add light pad if we have enough voices
                active_voices.append(len(voices) - 1)  # Last voice as pad
            density_multiplier = 1.0
            
        elif section_name == "B":
            # Section B: Secondary voice (second highest proportion)
            active_voices = [1] if len(voices) > 1 else [0]
            if len(voices) > 4:  # Add different pad support
                active_voices.append(len(voices) - 2)  # Second-to-last voice as pad
            density_multiplier = 1.0
            
        elif section_name == "A'":
            # Section A': Return to primary voice with variation
            active_voices = [0]  # Primary voice
            if len(voices) > 2:
                active_voices.append(2)  # Third voice for variation
            density_multiplier = 1.2  # Slightly more active
            
        else:  # "Tutti"
            # Tutti: All voices but at reduced density to avoid clutter
            active_voices = list(range(len(voices)))
            density_multiplier = 0.6  # Reduced density
        
        # Add transition effect between sections (except for the last section)
        transition = i < len(sections_plan) - 1
        
        sections.append(Section(
            name=section_name,
            start_beat=start_beat,
            end_beat=end_beat,
            active_voices=active_voices,
            density_multiplier=density_multiplier,
            transition=transition
        ))
        
        current_beat = end_beat
    
    return sections, total_beats


def _create_transition_notes(start_time: float, duration: float, rng: np.random.Generator) -> List[Note]:
    """Create transition effect notes (cymbal swell or fill).
    
    Args:
        start_time: Start time of the transition in seconds
        duration: Duration of the transition effect
        rng: Random number generator
        
    Returns:
        List of transition effect notes
    """
    notes = []
    
    # Choose transition type
    transition_type = rng.choice(["swell", "fill"])
    
    if transition_type == "swell":
        # Cymbal swell: noise burst with LPF sweep
        notes.append(Note(
            start=start_time,
            dur=duration,
            midi=49,  # Crash cymbal MIDI note
            vel=0.3,
            track="transition_swell",
            pan=0.0
        ))
    else:
        # Fill: Quick drum hits
        fill_hits = max(2, int(duration * 4))  # Roughly 4 hits per second
        for i in range(fill_hits):
            hit_time = start_time + (i * duration / fill_hits)
            midi_note = rng.choice([36, 38, 42])  # Kick, snare, hi-hat
            notes.append(Note(
                start=hit_time,
                dur=0.1,
                midi=midi_note,
                vel=0.4,
                track="transition_fill",
                pan=rng.uniform(-0.3, 0.3)
            ))
    
    return notes


def _compose_voice_track(voice: VoiceSpec, params: MusicParams, voice_id: int, sections: list[Section], rng: np.random.Generator) -> List[Note]:
    """Compose a track for a specific voice within sectional structure.
    
    Args:
        voice: Voice specification with instrument and performance parameters
        params: Global musical parameters (key, BPM, etc.)
        voice_id: Unique identifier for this voice
        sections: List of sections defining when this voice is active
        rng: Seeded random number generator
        
    Returns:
        List of Note objects for this voice
    """
    scale_notes = _scale_midi(params.root, params.mode)
    spb = 60.0 / params.bpm  # Seconds per beat
    
    # Calculate base register with octave offset
    base_register = 60 + (voice.octave * 12)  # Middle C + octave shifts
    
    notes = []
    track_name = f"voice_{voice_id}_{voice.instrument}"
    
    # Compose notes only for sections where this voice is active
    for section in sections:
        if voice_id not in section.active_voices:
            continue  # Skip inactive sections for this voice
        
        # Apply section density multiplier to voice activity
        effective_activity = voice.activity * section.density_multiplier
        beat_interval = max(1, int(1.0 / effective_activity))
        
        # Add section-specific variation
        if section.name == "A'":
            # A' section: add variation by shifting register slightly
            register_variation = 2  # +2 semitones
        else:
            register_variation = 0
        
        # Compose notes for this section
        beat = section.start_beat
        while beat < section.end_beat:
            t = beat * spb
            
            # Select note with mode bias
            midi_note = _apply_mode_bias(scale_notes, voice.mode_bias, beat, rng)
            
            # Transpose to voice's register with variation
            midi_note = midi_note - 60 + base_register + register_variation
            
            # Ensure note is in valid MIDI range
            midi_note = np.clip(midi_note, 21, 108)  # Piano range
            
            # Note duration varies with activity
            base_duration = spb * (0.8 + 0.4 * rng.random())  # 80-120% of beat
            if effective_activity > 1.5:  # High activity = shorter notes
                base_duration *= 0.7
            elif effective_activity < 0.7:  # Low activity = longer notes
                base_duration *= 1.4
            
            # Velocity scales with voice gain and section
            base_velocity = voice.gain * (0.7 + 0.3 * rng.random())
            if section.name == "Tutti":
                base_velocity *= 0.8  # Slightly quieter in tutti to avoid clutter
            
            notes.append(Note(
                start=t,
                dur=base_duration,
                midi=int(midi_note),
                vel=base_velocity,
                track=track_name,
                pan=voice.pan
            ))
            
            # Advance beat based on activity
            beat += beat_interval
    
    return notes

def compose_track(p: MusicParams) -> List[Note]:
    """Compose a sectional multi-voice musical arrangement from parameters.
    
    Creates a structured piece with A/B/A'/Tutti sections based on dominant colors.
    Each section features different combinations of voices to create musical narrative.
    
    Args:
        p: Musical parameters including voices, BPM, key, duration, etc.
        
    Returns:
        List of Note objects sorted by start time
    """
    print(f"🎼 Composing sectional arrangement...")
    print(f"   🎵 Key: {p.root} {p.mode}, {p.bpm} BPM, {p.duration:.1f}s duration")
    print(f"   🎤 Voices: {len(p.voices)} color-derived instruments")
    
    # Create seeded RNG for composition choices
    rng = np.random.default_rng(hash(p.root + p.mode + str(p.bpm)) & 0xFFFFFFFF)
    
    spb = 60.0 / p.bpm  # seconds per beat
    initial_beats = int(p.duration / spb)
    
    print(f"   📊 Analyzing dominant colors for section structure...")
    # Voices are already sorted by color proportion (descending)
    top_colors = p.voices[:min(3, len(p.voices))]  # Top 2-3 colors
    for i, voice in enumerate(top_colors):
        print(f"      Color {i+1}: RGB{voice.color.rgb} (prop={voice.color.prop:.2f}) → {voice.instrument}")
    
    print(f"   🔍 Creating section structure...")
    sections, total_beats = _create_section_structure(p.voices, initial_beats, p.meter)
    
    # Update duration to align with bar boundaries
    actual_duration = total_beats * spb
    print(f"   📏 Timing: {spb:.3f}s per beat, {total_beats} beats ({actual_duration:.1f}s duration)")
    print(f"   🎹 Scale: {p.root} {p.mode} ({MODES[p.mode]})")
    
    # Display section structure
    for section in sections:
        active_instruments = [p.voices[i].instrument for i in section.active_voices]
        beats_duration = section.end_beat - section.start_beat
        bars = beats_duration // p.meter[0]
        print(f"   🎵 {section.name}: bars {section.start_beat//p.meter[0]+1}-{section.end_beat//p.meter[0]} ({bars} bars) - {active_instruments}")

    all_notes: List[Note] = []
    voice_note_counts = []

    # Compose track for each voice within sectional structure
    for i, voice in enumerate(p.voices):
        print(f"   [{int((i+1)/len(p.voices)*70)}%] 🎵 Composing voice {i+1}: {voice.instrument}...")
        
        voice_notes = _compose_voice_track(voice, p, i, sections, rng)
        all_notes.extend(voice_notes)
        voice_note_counts.append(len(voice_notes))
        
        # Show voice activity
        active_sections = [s.name for s in sections if i in s.active_voices]
        color = voice.color
        print(f"      🎨 Color: RGB{color.rgb} (prop={color.prop:.2f}) - Active in: {', '.join(active_sections)}")
        print(f"      🎶 Notes: {len(voice_notes)}, gain={voice.gain:.2f}, pan={voice.pan:+.2f}, octave={voice.octave:+d}")

    # Add transition effects
    print(f"   [80%] 🎵 Adding transition effects...")
    transition_count = 0
    for section in sections:
        if section.transition:
            # Add transition at the end of this section
            transition_start = section.end_beat * spb - 0.5  # Start 0.5s before section end
            transition_duration = 1.0  # 1 second transition
            transition_notes = _create_transition_notes(transition_start, transition_duration, rng)
            all_notes.extend(transition_notes)
            transition_count += len(transition_notes)
    
    print(f"      Added {transition_count} transition effect notes")

    print(f"   [100%] ✨ Composition complete!")
    print(f"   📊 Generated {len(all_notes)} total notes across {len(p.voices)} voices + transitions:")
    for i, (voice, count) in enumerate(zip(p.voices, voice_note_counts)):
        print(f"   🎵 Voice {i+1} ({voice.instrument}): {count} notes")
    
    # Sort by start time
    sorted_notes = sorted(all_notes, key=lambda n: n.start)
    print(f"   🎼 Notes arranged chronologically across sections")
    return sorted_notes
