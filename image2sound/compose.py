from dataclasses import dataclass
from typing import List
import math
from .mapping import MusicParams

@dataclass
class Note:
    """A musical note with timing, pitch, and performance attributes.
    
    Attributes:
        start: Start time in seconds
        dur: Duration in seconds
        midi: MIDI note number (60 = middle C)
        vel: Velocity/volume [0,1]
        track: Track name for grouping notes
    """
    start: float
    dur: float
    midi: int
    vel: float
    track: str

def _scale_midi(root: str, major: bool) -> list[int]:
    """Build a major or minor scale starting from root note.
    
    Args:
        root: Root note name (C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B)
        major: True for major scale, False for minor scale
        
    Returns:
        List of 7 MIDI note numbers in the scale, centered around middle C (60)
    """
    names = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
    root_ix = names.index(root)
    pattern = [0,2,4,5,7,9,11] if major else [0,2,3,5,7,8,10]
    return [60 + ((root_ix + i) % 12) for i in pattern]

def compose_track(p: MusicParams) -> List[Note]:
    """Compose a simple multi-track musical arrangement from parameters.
    
    Creates a basic 4/4 time arrangement with chords, lead melody, bass, and drums:
    - Chords: Triad (root, third, fifth) on beats 0, 4, 8... lasting 2 beats
    - Lead: Ascending scale pattern, one note per beat, octave up
    - Bass: Root note on even beats, octave down
    - Drums: Kick (36) and snare (38) alternating per beat
    
    Args:
        p: Musical parameters including BPM, scale, duration, etc.
        
    Returns:
        List of Note objects sorted by start time
    """
    print(f"🎼 Composing musical arrangement...")
    print(f"   🎵 Key: {p.scale}, {p.bpm} BPM, {p.duration:.1f}s duration")
    
    major = "major" in p.scale
    scale = _scale_midi(p.root, major)
    spb = 60.0 / p.bpm  # seconds per beat
    notes: List[Note] = []
    beats = int(p.duration / spb)
    
    print(f"   📏 Timing: {spb:.3f}s per beat, {beats} total beats")
    print(f"   🎹 Scale notes: {scale}")

    # Track counters for progress
    chord_count = 0
    lead_count = 0  
    bass_count = 0
    drum_count = 0

    for b in range(beats):
        t = b * spb
        
        # Show progress every 25% of beats
        if beats >= 4 and b % (beats // 4) == 0 and b > 0:
            progress = int((b / beats) * 100)
            print(f"   [{progress}%] 🎵 Composing beat {b}/{beats}...")
        
        # Chords: triad on beat %4==0, duration=2 beats
        if b % 4 == 0:
            for m in [scale[0], scale[2], scale[4]]:
                notes.append(Note(t, 2*spb, m, 0.5, "chords"))
                chord_count += 1
        
        # Lead: ascending scale pattern, +12 semitones (octave up)
        lead = scale[(b*2) % len(scale)] + 12
        notes.append(Note(t, spb, lead, 0.8, "lead"))
        lead_count += 1
        
        # Bass: root note on even beats, -12 semitones (octave down)
        if b % 2 == 0:
            notes.append(Note(t, spb, scale[0]-12, 0.7, "bass"))
            bass_count += 1
        
        # Drums: alternating kick (36) and snare (38) every beat
        notes.append(Note(t, 0.05, 36 if b % 2 == 0 else 38, 1.0, "drums"))
        drum_count += 1

    print(f"   [100%] ✨ Composition complete!")
    print(f"   📊 Generated {len(notes)} total notes:")
    print(f"   🎹 Chords: {chord_count} notes")
    print(f"   🎶 Lead melody: {lead_count} notes") 
    print(f"   🎸 Bass: {bass_count} notes")
    print(f"   🥁 Drums: {drum_count} notes")
    
    # Sort by start time
    sorted_notes = sorted(notes, key=lambda n: n.start)
    print(f"   🎼 Notes arranged chronologically")
    return sorted_notes
