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


# Drum pattern definitions by meter - each pattern is (beat_positions, density_level, name)
DRUM_PATTERNS = {
    (4, 4): [  # 4/4 time patterns
        ([0, 2], 0, "basic_kick_snare"),           # Kick on 1,3 - minimal
        ([0, 1, 2, 3], 1, "four_on_floor"),       # All beats - low density
        ([0, 2, 2.5, 3.5], 2, "backbeat"),        # Kick, snare, off-beat hits
        ([0, 1.5, 2, 2.75, 3.5], 3, "shuffle"),   # Shuffle feel - medium
        ([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], 4, "eighth_notes"),  # 8th note hi-hats
        ([0, 0.33, 0.67, 1, 1.33, 1.67, 2, 2.33, 2.67, 3, 3.33, 3.67], 5, "triplet_feel"),  # Triplets
        ([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75], 6, "sixteenth_notes"),  # 16th notes
        ([0, 0.125, 0.375, 0.5, 0.625, 1, 1.125, 1.375, 1.5, 1.625, 2, 2.125, 2.375, 2.5, 2.625, 3, 3.125, 3.375, 3.5, 3.625], 7, "complex_syncopation")  # Very dense
    ],
    (3, 4): [  # 3/4 time patterns  
        ([0], 0, "waltz_minimal"),                 # Just downbeat
        ([0, 2], 1, "waltz_basic"),                # 1 and 3
        ([0, 1, 2], 2, "waltz_full"),              # All beats
        ([0, 1, 1.5, 2, 2.5], 3, "waltz_swing"),  # With subdivisions
        ([0, 0.5, 1, 1.5, 2, 2.5], 4, "waltz_compound"),  # 8th note feel
        ([0, 0.33, 0.67, 1, 1.33, 1.67, 2, 2.33, 2.67], 5, "waltz_triplet"),  # Triplet subdivisions
        ([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75], 6, "waltz_sixteenth")  # Dense
    ],
    (6, 8): [  # 6/8 time patterns
        ([0], 0, "6_8_minimal"),                   # Downbeat only
        ([0, 3], 1, "6_8_basic"),                  # Strong beats
        ([0, 1.5, 3, 4.5], 2, "6_8_compound"),    # Compound feel
        ([0, 1, 2, 3, 4, 5], 3, "6_8_full"),      # All 8th notes
        ([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5], 4, "6_8_detailed"),  # 16th note subdivisions
        ([0, 0.33, 1, 1.5, 2, 3, 3.33, 4, 4.5, 5], 5, "6_8_swing"),  # Swing subdivision
        ([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75], 6, "6_8_complex")  # Very dense
    ],
    (5, 4): [  # 5/4 time patterns
        ([0], 0, "5_4_minimal"),                   # Downbeat only
        ([0, 2], 1, "5_4_basic"),                  # 1 and 3
        ([0, 2, 3], 2, "5_4_asymmetric"),         # Asymmetric grouping
        ([0, 1, 2, 3, 4], 3, "5_4_full"),         # All beats
        ([0, 0.5, 1, 2, 2.5, 3, 4, 4.5], 4, "5_4_compound"),  # With subdivisions
        ([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5], 5, "5_4_dense"),  # 8th note feel
        ([0, 0.25, 0.75, 1, 1.5, 2, 2.25, 2.75, 3, 3.5, 4, 4.25, 4.75], 6, "5_4_syncopated")  # Complex
    ]
}


def _select_drum_pattern(meter: tuple[int, int], texture_energy: float, max_voice_saturation: float, rng: np.random.Generator) -> tuple[list[float], str]:
    """Select drum pattern based on texture energy and voice saturation.
    
    Args:
        meter: Time signature (numerator, denominator)
        texture_energy: Texture energy from image analysis [0,1]
        max_voice_saturation: Maximum saturation among all voices [0,1]
        rng: Random number generator
        
    Returns:
        (beat_positions, pattern_name) tuple
    """
    # Compute composite density score
    # texture_energy contributes to rhythmic complexity
    # max_voice_saturation adds color-driven rhythmic intensity
    density_score = (texture_energy * 0.6 + max_voice_saturation * 0.4)
    
    # Get patterns for this meter (convert to tuple if needed)
    meter_tuple = tuple(meter) if not isinstance(meter, tuple) else meter
    patterns = DRUM_PATTERNS.get(meter_tuple, DRUM_PATTERNS[(4, 4)])  # Fallback to 4/4
    
    # Map density score to pattern index (0-7 range)
    pattern_index = int(density_score * (len(patterns) - 1))
    pattern_index = np.clip(pattern_index, 0, len(patterns) - 1)
    
    # Add slight randomization (±1 level)
    if rng.random() < 0.3:  # 30% chance to vary
        variation = rng.choice([-1, 1])
        pattern_index = np.clip(pattern_index + variation, 0, len(patterns) - 1)
    
    beat_positions, density_level, pattern_name = patterns[pattern_index]
    return beat_positions, pattern_name


def _determine_voice_style(voice: VoiceSpec) -> str:
    """Determine voice playing style based on saturation.
    
    Args:
        voice: Voice specification with color properties
        
    Returns:
        "arpeggio" for high saturation, "sustained" for low saturation
    """
    saturation_threshold = 0.6
    return "arpeggio" if voice.color.sat > saturation_threshold else "sustained"


def _humanize_timing(original_time: float, rng: np.random.Generator) -> float:
    """Apply micro-timing humanization.
    
    Args:
        original_time: Original note start time in seconds
        rng: Random number generator
        
    Returns:
        Humanized timing with ±10-20ms variation
    """
    # Random humanization between -20ms to +20ms
    humanization = rng.uniform(-0.020, 0.020)  # ±20ms
    return max(0.0, original_time + humanization)


def _chord_to_midi(chord_symbol: str, root_note: str, mode: str, enrichment_level: int = 0, rng: np.random.Generator = None) -> list[int]:
    """Convert chord symbol to MIDI note numbers.
    
    Args:
        chord_symbol: Roman numeral chord symbol (I, V, vi, etc.)
        root_note: Root note of the key (C, D, etc.)
        mode: Musical mode (ionian, aeolian, etc.)
        enrichment_level: 0=basic, 1=7/add9, 2=#11/6
        
    Returns:
        List of MIDI note numbers for the chord
    """
    # Map root note to semitone offset
    root_offsets = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
                   "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
    key_root = root_offsets[root_note]
    
    # Get scale pattern
    scale_pattern = MODES.get(mode, MODES["ionian"])
    
    # Map Roman numerals to scale degrees (0-indexed)
    roman_map = {
        "I": 0, "i": 0, "II": 1, "ii": 1, "bII": 1,
        "III": 2, "iii": 2, "bIII": 2, "IV": 3, "iv": 3,
        "V": 4, "v": 4, "bV": 4, "VI": 5, "vi": 5, "bVI": 5,
        "VII": 6, "vii": 6, "bVII": 6
    }
    
    # Extract chord root degree
    base_symbol = chord_symbol.rstrip("°+-#9b9#11")
    degree = roman_map.get(base_symbol, 0)
    
    # Build basic triad
    chord_root = (key_root + scale_pattern[degree % len(scale_pattern)]) % 12
    third = (chord_root + 4 if chord_symbol.isupper() else chord_root + 3) % 12  # Major/minor third
    fifth = (chord_root + 7) % 12
    
    chord_notes = [60 + chord_root, 60 + third, 60 + fifth]  # Middle C octave
    
    # Add enrichments based on level
    if enrichment_level >= 1 and rng is not None:
        # Add 7th or add9
        if "7" in chord_symbol or rng.random() < 0.4:
            seventh = (chord_root + (11 if chord_symbol.isupper() else 10)) % 12
            chord_notes.append(60 + seventh)
        elif "add9" in chord_symbol or rng.random() < 0.3:
            ninth = (chord_root + 2) % 12
            chord_notes.append(60 + ninth + 12)  # Octave up
    
    if enrichment_level >= 2 and rng is not None:
        # Add #11 or 6th extensions
        if "#11" in chord_symbol or rng.random() < 0.2:
            sharp_eleven = (chord_root + 6) % 12
            chord_notes.append(60 + sharp_eleven + 12)
        elif "6" in chord_symbol or rng.random() < 0.3:
            sixth = (chord_root + 9) % 12
            chord_notes.append(60 + sixth)
    
    return sorted(chord_notes)


def _create_altered_v_chord(root_note: str, mode: str, rng: np.random.Generator) -> list[int]:
    """Create an altered V chord (b9/#9/#11) for complementary color sections.
    
    Args:
        root_note: Root note of the key
        mode: Musical mode
        rng: Random number generator
        
    Returns:
        List of MIDI notes for altered V chord
    """
    root_offsets = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
                   "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
    key_root = root_offsets[root_note]
    
    # V chord root is a fifth above the key root
    v_root = (key_root + 7) % 12
    
    # Basic dominant 7th chord
    chord_notes = [
        60 + v_root,        # Root
        60 + (v_root + 4) % 12,  # Major third
        60 + (v_root + 7) % 12,  # Perfect fifth
        60 + (v_root + 10) % 12  # Minor seventh
    ]
    
    # Add alterations
    alterations = ["b9", "#9", "#11"]
    chosen_alt = rng.choice(alterations)
    
    if chosen_alt == "b9":
        chord_notes.append(60 + (v_root + 1) % 12 + 12)  # b9 octave up
    elif chosen_alt == "#9":
        chord_notes.append(60 + (v_root + 3) % 12 + 12)  # #9 octave up
    else:  # "#11"
        chord_notes.append(60 + (v_root + 6) % 12 + 12)  # #11 octave up
    
    return sorted(chord_notes)


def _voice_lead_to_nearest(from_chord: list[int], to_chord: list[int]) -> list[int]:
    """Apply smooth voice-leading by choosing nearest chord tones.
    
    Args:
        from_chord: Previous chord MIDI notes
        to_chord: Target chord MIDI notes
        
    Returns:
        Re-voiced target chord with smooth voice-leading
    """
    if not from_chord:
        return to_chord
    
    # For each note in the target chord, find the best octave (±12 semitones)
    voiced_chord = []
    for target_note in to_chord:
        best_note = target_note
        min_distance = float('inf')
        
        # Try different octaves
        for octave_shift in [-12, 0, 12]:
            candidate = target_note + octave_shift
            if 48 <= candidate <= 84:  # Reasonable range
                # Find minimum distance to any note in from_chord
                distance = min(abs(candidate - from_note) for from_note in from_chord)
                if distance < min_distance:
                    min_distance = distance
                    best_note = candidate
        
        voiced_chord.append(best_note)
    
    return sorted(voiced_chord)


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
    
    # Derive the voice's signature color motif
    voice_motif = _derive_color_motif(voice, scale_notes, rng)
    motif_occurrence_count = 0  # Track how many times motif has been played for mutations
    
    # Compose notes only for sections where this voice is active
    for section in sections:
        if voice_id not in section.active_voices:
            continue  # Skip inactive sections for this voice
        
        # Apply section density multiplier to voice activity
        effective_activity = voice.activity * section.density_multiplier
        beat_interval = max(1, int(1.0 / effective_activity))
        
        # Determine voice playing style based on color saturation
        voice_style = _determine_voice_style(voice)
        
        # Add section-specific variation
        if section.name == "A'":
            # A' section: add variation by shifting register slightly
            register_variation = 2  # +2 semitones
        else:
            register_variation = 0
        
        # Play color call-sign motif at the start of the section
        section_start_time = section.start_beat * spb
        
        # Apply mutations to motif based on occurrence count
        current_motif = voice_motif.copy()  # Start with original
        if motif_occurrence_count > 0:
            # Choose mutation type based on occurrence count
            if motif_occurrence_count % 2 == 1:
                current_motif = _mutate_motif(current_motif, "invert", scale_notes)
            else:
                current_motif = _mutate_motif(current_motif, "transpose_5th", scale_notes)
        
        # Get current chord for motif fitting (use first chord of section)
        beats_per_bar = params.meter[0]
        bar_index = section.start_beat // beats_per_bar
        chord_index = bar_index % len(params.progression)
        chord_symbol = params.progression[chord_index]
        current_chord = _chord_to_midi(chord_symbol, params.root, params.mode, 0, rng)
        
        # Fit motif to current chord
        fitted_motif = _fit_motif_to_chord(current_motif, current_chord, scale_notes)
        
        # Play the 3-note motif
        motif_note_duration = spb * 0.4  # Each motif note is 40% of a beat
        for i, motif_midi in enumerate(fitted_motif):
            motif_time = section_start_time + (i * motif_note_duration)
            humanized_motif_time = _humanize_timing(motif_time, rng)
            
            # Transpose to voice register
            motif_midi_transposed = motif_midi - 60 + base_register + register_variation
            motif_midi_transposed = np.clip(motif_midi_transposed, 21, 108)
            
            # Motif notes have distinctive velocity and duration
            motif_velocity = voice.gain * 0.9  # Prominent but not overpowering
            
            notes.append(Note(
                start=humanized_motif_time,
                dur=motif_note_duration * 1.2,  # Slightly overlapping for legato
                midi=int(motif_midi_transposed),
                vel=motif_velocity,
                track=track_name,
                pan=voice.pan
            ))
        
        motif_occurrence_count += 1
        
        # Compose regular notes for this section (start after motif)
        beat = section.start_beat + 2  # Start 2 beats after motif (give it space)
        while beat < section.end_beat:
            t = beat * spb
            
            # Apply humanization to timing
            humanized_time = _humanize_timing(t, rng)
            
            if voice_style == "arpeggio" and effective_activity > 0.8:
                # High saturation = arpeggio style: play multiple notes in quick succession
                chord_notes = []
                base_note = _apply_mode_bias(scale_notes, voice.mode_bias, beat, rng)
                
                # Create simple arpeggio (1-3-5 pattern)
                arp_intervals = [0, 2, 4]  # Root, third, fifth in scale degrees
                arp_duration = spb / len(arp_intervals)  # Divide beat among arpeggio notes
                
                for i, interval in enumerate(arp_intervals):
                    scale_idx = (scale_notes.index(base_note) + interval) % len(scale_notes)
                    arp_note = scale_notes[scale_idx]
                    
                    # Transpose to voice's register with variation
                    midi_note = arp_note - 60 + base_register + register_variation
                    midi_note = np.clip(midi_note, 21, 108)
                    
                    # Stagger arpeggio notes slightly
                    arp_time = humanized_time + (i * arp_duration)
                    
                    # Shorter duration for arpeggio notes
                    note_duration = arp_duration * 0.9
                    
                    # Velocity with slight variation for each arpeggio note
                    arp_velocity = voice.gain * (0.6 + 0.2 * rng.random())
                    if section.name == "Tutti":
                        arp_velocity *= 0.8
                    
                    notes.append(Note(
                        start=arp_time,
                        dur=note_duration,
                        midi=int(midi_note),
                        vel=arp_velocity,
                        track=track_name,
                        pan=voice.pan
                    ))
                
            else:
                # Sustained style: longer, held notes (low saturation)
                midi_note = _apply_mode_bias(scale_notes, voice.mode_bias, beat, rng)
                
                # Transpose to voice's register with variation
                midi_note = midi_note - 60 + base_register + register_variation
                midi_note = np.clip(midi_note, 21, 108)
                
                # Note duration varies with activity and style
                if voice_style == "sustained":
                    base_duration = spb * (1.2 + 0.8 * rng.random())  # Longer sustained notes
                else:
                    base_duration = spb * (0.8 + 0.4 * rng.random())  # Standard duration
                
                if effective_activity > 1.5:  # High activity = shorter notes
                    base_duration *= 0.7
                elif effective_activity < 0.7:  # Low activity = longer notes
                    base_duration *= 1.4
                
                # Velocity scales with voice gain and section
                base_velocity = voice.gain * (0.7 + 0.3 * rng.random())
                if section.name == "Tutti":
                    base_velocity *= 0.8  # Slightly quieter in tutti to avoid clutter
                
                notes.append(Note(
                    start=humanized_time,
                    dur=base_duration,
                    midi=int(midi_note),
                    vel=base_velocity,
                    track=track_name,
                    pan=voice.pan
                ))
            
            # Advance beat based on activity
            beat += beat_interval
    
    return notes


def _derive_color_motif(voice: VoiceSpec, scale_notes: list[int], rng: np.random.Generator) -> list[int]:
    """Derive a 3-note motif from voice color properties.
    
    Args:
        voice: Voice specification with color properties
        scale_notes: Scale notes available for the motif
        rng: Random number generator seeded by voice color
        
    Returns:
        List of 3 MIDI notes representing the color call-sign motif
    """
    # Map hue to starting scale degree (0-360° → 0-6 scale degrees)
    hue_bucket = int(voice.color.hue / 360.0 * len(scale_notes)) % len(scale_notes)
    starting_degree = hue_bucket
    
    # Create seeded RNG for consistent intervals based on voice color
    color_seed = int(voice.color.hue * 1000 + voice.color.sat * 100 + voice.color.val * 10) & 0xFFFFFFFF
    motif_rng = np.random.default_rng(color_seed)
    
    # Define possible intervals (scale degrees)
    interval_choices = [1, 2, 3, -1, -2]  # Step up/down within scale
    
    # Generate two intervals to create 3-note motif
    interval1 = motif_rng.choice(interval_choices)
    interval2 = motif_rng.choice(interval_choices)
    
    # Build motif as scale degrees
    motif_degrees = [
        starting_degree,
        (starting_degree + interval1) % len(scale_notes),
        (starting_degree + interval1 + interval2) % len(scale_notes)
    ]
    
    # Convert to MIDI notes
    motif_notes = [scale_notes[degree] for degree in motif_degrees]
    return motif_notes


def _mutate_motif(original_motif: list[int], mutation_type: str, scale_notes: list[int]) -> list[int]:
    """Apply mutation to a motif (inversion or transposition by 5th).
    
    Args:
        original_motif: Original 3-note motif
        mutation_type: "invert" or "transpose_5th"
        scale_notes: Available scale notes
        
    Returns:
        Mutated motif
    """
    if mutation_type == "invert":
        # Invert intervals around first note
        root_note = original_motif[0]
        intervals = [note - root_note for note in original_motif]
        inverted_intervals = [-interval for interval in intervals]
        inverted_motif = [root_note + interval for interval in inverted_intervals]
        
        # Ensure notes stay in reasonable range and map to scale
        adjusted_motif = []
        for note in inverted_motif:
            # Find closest scale note
            closest_scale_note = min(scale_notes, key=lambda x: abs(x - note))
            adjusted_motif.append(closest_scale_note)
        return adjusted_motif
        
    elif mutation_type == "transpose_5th":
        # Transpose by a perfect 5th (7 semitones)
        transposed = [note + 7 for note in original_motif]
        
        # Map to scale notes and ensure reasonable range
        adjusted_motif = []
        for note in transposed:
            # Find closest scale note
            closest_scale_note = min(scale_notes, key=lambda x: abs(x - note))
            # Keep in reasonable range
            while closest_scale_note > 84:  # Too high
                closest_scale_note -= 12
            while closest_scale_note < 36:  # Too low
                closest_scale_note += 12
            adjusted_motif.append(closest_scale_note)
        return adjusted_motif
    
    return original_motif


def _fit_motif_to_chord(motif: list[int], chord_notes: list[int], scale_notes: list[int]) -> list[int]:
    """Adjust motif notes to fit the current chord when possible.
    
    Args:
        motif: Original motif notes
        chord_notes: Current chord notes (MIDI)
        scale_notes: Available scale notes
        
    Returns:
        Chord-fitted motif
    """
    if not chord_notes:
        return motif
    
    fitted_motif = []
    
    for note in motif:
        # Check if note is already a chord tone (within octave)
        note_class = note % 12
        is_chord_tone = any((chord_note % 12) == note_class for chord_note in chord_notes)
        
        if is_chord_tone:
            # Note fits chord, keep it
            fitted_motif.append(note)
        else:
            # Find nearest chord tone
            chord_tones_extended = []
            for chord_note in chord_notes:
                # Add chord tones in multiple octaves around the motif note
                for octave in [-12, 0, 12]:
                    candidate = chord_note + octave
                    if 36 <= candidate <= 84:  # Reasonable range
                        chord_tones_extended.append(candidate)
            
            if chord_tones_extended:
                nearest_chord_tone = min(chord_tones_extended, key=lambda x: abs(x - note))
                fitted_motif.append(nearest_chord_tone)
            else:
                # Fallback: use original note
                fitted_motif.append(note)
    
    return fitted_motif


def _compose_chord_track(params: MusicParams, sections: list[Section], rng: np.random.Generator) -> List[Note]:
    """Compose chord progression track with color-aware enrichments and altered V chords.
    
    Args:
        params: Musical parameters with chord enrichment settings
        sections: List of sections for timing
        rng: Random number generator
        
    Returns:
        List of chord Notes
    """
    spb = 60.0 / params.bpm
    chord_notes = []
    
    # Track previous chord for voice-leading
    prev_chord_midi = []
    
    for section in sections:
        # Determine if this section should have an altered V chord
        use_altered_v = (params.has_complement and 
                        rng.random() < 0.6 and  # 60% chance per section
                        section.name != "Tutti")  # Not in tutti to avoid clutter
        
        section_beats = section.end_beat - section.start_beat
        beats_per_bar = params.meter[0]
        bars_in_section = section_beats // beats_per_bar
        
        for bar in range(bars_in_section):
            bar_start_beat = section.start_beat + (bar * beats_per_bar)
            bar_start_time = bar_start_beat * spb
            bar_duration = beats_per_bar * spb
            
            # Select chord from progression
            chord_index = bar % len(params.progression)
            chord_symbol = params.progression[chord_index]
            
            # Check if this is the last bar in the section and we want altered V
            if use_altered_v and bar == bars_in_section - 1:
                # Insert altered V chord in second half of bar, resolving to I
                half_bar_time = bar_duration / 2
                
                # First half: original chord
                chord_midi = _chord_to_midi(chord_symbol, params.root, params.mode, 
                                          params.chord_enrichment_level, rng)
                chord_midi = _voice_lead_to_nearest(prev_chord_midi, chord_midi)
                
                for midi_note in chord_midi:
                    chord_notes.append(Note(
                        start=bar_start_time,
                        dur=half_bar_time,
                        midi=midi_note,
                        vel=0.5,
                        track="chords",
                        pan=0.0
                    ))
                
                # Second half: altered V chord
                altered_v_midi = _create_altered_v_chord(params.root, params.mode, rng)
                altered_v_midi = _voice_lead_to_nearest(chord_midi, altered_v_midi)
                
                for midi_note in altered_v_midi:
                    chord_notes.append(Note(
                        start=bar_start_time + half_bar_time,
                        dur=half_bar_time,
                        midi=midi_note,
                        vel=0.6,  # Slightly louder for emphasis
                        track="chords",
                        pan=0.0
                    ))
                
                prev_chord_midi = altered_v_midi
                
            else:
                # Normal chord
                chord_midi = _chord_to_midi(chord_symbol, params.root, params.mode,
                                          params.chord_enrichment_level, rng)
                chord_midi = _voice_lead_to_nearest(prev_chord_midi, chord_midi)
                
                for midi_note in chord_midi:
                    chord_notes.append(Note(
                        start=bar_start_time,
                        dur=bar_duration,
                        midi=midi_note,
                        vel=0.5,
                        track="chords",
                        pan=0.0
                    ))
                
                prev_chord_midi = chord_midi
    
    return chord_notes


def _compose_drum_track(params: MusicParams, sections: list[Section], texture_energy: float, rng: np.random.Generator) -> List[Note]:
    """Compose rhythm track using color/texture-driven drum patterns.
    
    Args:
        params: Musical parameters
        sections: List of sections for timing
        texture_energy: Texture energy from image analysis
        rng: Random number generator
        
    Returns:
        List of drum Notes
    """
    if not params.voices:
        return []
    
    # Find maximum saturation among all voices for rhythm intensity
    max_voice_saturation = max(voice.color.sat for voice in params.voices)
    
    # Select drum pattern based on texture and color saturation
    beat_positions, pattern_name = _select_drum_pattern(params.meter, texture_energy, max_voice_saturation, rng)
    
    spb = 60.0 / params.bpm  # seconds per beat
    beats_per_bar = params.meter[0]
    drum_notes = []
    
    # Generate drum hits for each section
    for section in sections:
        section_beats = section.end_beat - section.start_beat
        bars_in_section = section_beats // beats_per_bar
        
        for bar in range(bars_in_section):
            bar_start_beat = section.start_beat + (bar * beats_per_bar)
            
            for beat_pos in beat_positions:
                if beat_pos < beats_per_bar:  # Ensure beat is within bar
                    absolute_beat = bar_start_beat + beat_pos
                    drum_time = absolute_beat * spb
                    
                    # Apply humanization to timing
                    humanized_time = _humanize_timing(drum_time, rng)
                    
                    # Vary drum sounds based on beat position and intensity
                    if beat_pos == 0:  # Downbeat
                        midi_note = 36  # Kick drum
                        velocity = 0.7 + 0.2 * max_voice_saturation  # Louder for saturated colors
                    elif int(beat_pos * 2) % 2 == 0:  # On strong subdivisions
                        midi_note = 38 if rng.random() < 0.6 else 42  # Snare or hi-hat
                        velocity = 0.5 + 0.2 * max_voice_saturation
                    else:  # Weak subdivisions
                        midi_note = 42  # Hi-hat
                        velocity = 0.3 + 0.2 * max_voice_saturation
                    
                    # Add slight velocity variation for humanization
                    velocity += rng.uniform(-0.1, 0.1)
                    velocity = np.clip(velocity, 0.1, 1.0)
                    
                    drum_notes.append(Note(
                        start=humanized_time,
                        dur=0.1,  # Short drum hits
                        midi=midi_note,
                        vel=velocity,
                        track="drums_rhythm",
                        pan=rng.uniform(-0.2, 0.2)  # Slight stereo spread
                    ))
    
    return drum_notes


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

    # Add chord progression track
    print(f"   [70%] 🎹 Composing chord progression...")
    chord_notes = _compose_chord_track(p, sections, rng)
    all_notes.extend(chord_notes)
    
    enrichment_desc = ["basic triads", "7th/add9 chords", "#11/6th extensions"][p.chord_enrichment_level]
    complement_desc = "with altered V chords" if p.has_complement else "standard progression"
    print(f"      🎵 {len(chord_notes)} chord notes using {enrichment_desc} ({complement_desc})")

    # Add rhythm/drum track
    print(f"   [75%] 🥁 Composing rhythm track...")
    max_voice_saturation = max(voice.color.sat for voice in p.voices) if p.voices else 0.0
    beat_positions, pattern_name = _select_drum_pattern(p.meter, p.texture_energy, max_voice_saturation, rng)
    drum_notes = _compose_drum_track(p, sections, p.texture_energy, rng)
    all_notes.extend(drum_notes)
    
    density_score = p.texture_energy * 0.6 + max_voice_saturation * 0.4
    print(f"      🥁 {len(drum_notes)} drum hits using '{pattern_name}' pattern (density={density_score:.2f})")

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
    print(f"   📊 Generated {len(all_notes)} total notes across {len(p.voices)} voices + chords + rhythm + transitions:")
    for i, (voice, count) in enumerate(zip(p.voices, voice_note_counts)):
        voice_style = _determine_voice_style(voice)
        sat_desc = f"sat={voice.color.sat:.2f}"
        print(f"   🎵 Voice {i+1} ({voice.instrument}): {count} notes [{voice_style} style, {sat_desc}]")
    print(f"   🎹 Chords: {len(chord_notes)} notes")
    print(f"   🥁 Rhythm: {len(drum_notes)} hits ({pattern_name})")
    
    # Sort by start time
    sorted_notes = sorted(all_notes, key=lambda n: n.start)
    print(f"   🎼 Notes arranged chronologically across sections")
    return sorted_notes
