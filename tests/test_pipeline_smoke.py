from pathlib import Path
from PIL import Image
from image2sound.cli import main as cli_main
from image2sound.compose import compose_track
from image2sound.mapping import MusicParams, VoiceSpec
from image2sound.features import ColorCluster
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
    # Create mock color cluster
    mock_color = ColorCluster(
        rgb=(200, 100, 50),
        hue=30.0,
        sat=0.75,
        val=0.78,
        prop=1.0,
        cx=0.5,
        cy=0.5
    )
    
    # Create mock voice
    mock_voice = VoiceSpec(
        instrument="pluck",
        mode_bias=0.0,
        pan=0.0,
        gain=0.8,
        octave=0,
        brightness=0.7,
        activity=1.0,
        color=mock_color
    )
    
    # Mock MusicParams for 5 second duration, ~120 BPM
    params = MusicParams(
        bpm=120,
        scale="C_major",
        root="C", 
        instruments=["piano", "lead", "drums"],  # Legacy
        intensity=0.7,
        duration=5.0,
        mode="ionian",
        meter=(4, 4),
        progression=["I", "V", "vi", "IV"],
        pan_lead=0.0,  # Legacy
        lead_offset=0,  # Legacy
        voices=[mock_voice]
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
    
    # Additional validation: verify we have voice tracks
    track_names = set(note.track for note in notes)
    expected_voice_tracks = {f"voice_{i+1}_{voice.instrument}" for i, voice in enumerate(params.voices)}
    assert expected_voice_tracks.issubset(track_names), f"Expected voice tracks {expected_voice_tracks}, got {track_names}"
    
    # Verify we have notes for each voice
    for i, voice in enumerate(params.voices):
        voice_track = f"voice_{i+1}_{voice.instrument}"
        voice_notes = [n for n in notes if n.track == voice_track]
        assert len(voice_notes) > 0, f"Should have notes for {voice_track}"
    
    # Verify basic musical properties
    spb = 60.0 / params.bpm
    expected_max_beats = int(params.duration / spb) + 1  # Allow some flexibility
    assert len(notes) <= expected_max_beats * len(params.voices), f"Too many notes: {len(notes)} for {expected_max_beats} beats and {len(params.voices)} voices"
