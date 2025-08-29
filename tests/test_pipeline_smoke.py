from pathlib import Path
from PIL import Image
from image2sound.cli import main as cli_main
from image2sound.compose import compose_track
from image2sound.mapping import MusicParams
import subprocess, sys

def test_cli_smoke(tmp_path: Path):
    img = tmp_path / "img.png"
    Image.new("RGB", (64,64), color=(100, 180, 220)).save(img)
    out = tmp_path / "out.wav"
    cmd = [sys.executable, "-m", "image2sound.cli", str(img), "--out", str(out)]
    # run as module to exercise click command
    subprocess.check_call(cmd)
    assert out.exists() and out.stat().st_size > 1000

def test_compose_smoke():
    """Test compose_track with mock MusicParams for 5s duration at ~120 BPM."""
    # Mock MusicParams for 5 second duration, ~120 BPM
    params = MusicParams(
        bpm=120,
        scale="C_major",
        root="C", 
        instruments=["piano", "lead", "drums"],
        intensity=0.7,
        duration=5.0
    )
    
    # Compose the track
    notes = compose_track(params)
    
    # Assert notes not empty
    assert len(notes) > 0, "Composed track should contain notes"
    
    # Assert starts are non-decreasing (sorted by start time)
    start_times = [note.start for note in notes]
    assert start_times == sorted(start_times), "Note start times should be non-decreasing"
    
    # Assert last note end <= duration + 0.5s tolerance
    if notes:
        last_note_end = max(note.start + note.dur for note in notes)
        assert last_note_end <= params.duration + 0.5, f"Last note end {last_note_end} should be <= {params.duration + 0.5}"
    
    # Additional validation: verify we have different track types
    track_names = set(note.track for note in notes)
    expected_tracks = {"chords", "lead", "bass", "drums"}
    assert expected_tracks.issubset(track_names), f"Expected tracks {expected_tracks}, got {track_names}"
    
    # Verify chord notes appear on appropriate beats (every 4th beat)
    chord_notes = [n for n in notes if n.track == "chords"]
    assert len(chord_notes) > 0, "Should have chord notes"
    
    # Verify lead notes appear on every beat
    lead_notes = [n for n in notes if n.track == "lead"] 
    spb = 60.0 / params.bpm
    expected_beats = int(params.duration / spb)
    assert len(lead_notes) == expected_beats, f"Expected {expected_beats} lead notes, got {len(lead_notes)}"
