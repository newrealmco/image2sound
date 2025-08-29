from dataclasses import dataclass
from typing import List
from .mapping import MusicParams

@dataclass
class Note:
    start: float
    dur: float
    midi: int
    vel: float
    track: str

def _scale_midi(root: str, major: bool) -> list[int]:
    names = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
    root_ix = names.index(root)
    pattern = [0,2,4,5,7,9,11] if major else [0,2,3,5,7,8,10]
    return [60 + ((root_ix + i) % 12) for i in pattern]

def compose_track(p: MusicParams) -> List[Note]:
    major = "major" in p.scale
    scale = _scale_midi(p.root, major)
    spb = 60.0 / p.bpm
    notes: List[Note] = []
    beats = int(p.duration / spb)

    for b in range(beats):
        t = b * spb
        if b % 4 == 0:
            for m in [scale[0], scale[2], scale[4]]:
                notes.append(Note(t, 2*spb, m, 0.5, "chords"))
        lead = scale[(b*2) % len(scale)] + 12
        notes.append(Note(t, spb, lead, 0.8, "lead"))
        if b % 2 == 0:
            notes.append(Note(t, spb, scale[0]-12, 0.7, "bass"))
        notes.append(Note(t, 0.05, 36 if b % 2 == 0 else 38, 1.0, "drums"))
    return notes
