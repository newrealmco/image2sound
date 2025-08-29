import click
from pathlib import Path
from .features import extract_features
from .mapping import map_features_to_music
from .compose import compose_track
from .synth import render_wav

@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--out", "-o", type=click.Path(), default="out.wav", 
              help="Output WAV file path")
@click.option("--style", type=click.Choice(["neutral","ambient","cinematic","rock"]), 
              default="neutral", help="Musical style for sonification")
@click.option("--duration", type=float, default=20.0, 
              help="Target duration in seconds")
def main(image_path: str, out: str, style: str, duration: float) -> None:
    """Convert an image to audio through algorithmic sonification.
    
    Extracts visual features (brightness, contrast, edges, colors) from an image
    and maps them to musical parameters (BPM, scale, key) to generate audio.
    
    Pipeline: extract_features → map_features_to_music → compose_track → render_wav
    
    Args:
        image_path: Path to input image file
        
    Options:
        --out: Output WAV file path (default: out.wav)
        --style: Musical style - neutral, ambient, cinematic, or rock
        --duration: Target audio duration in seconds (default: 20.0)
    """
    print("🎶Convert an image to audio🎶")
    feats = extract_features(Path(image_path))
    params = map_features_to_music(feats, style=style, target_duration=duration)
    notes = compose_track(params)
    render_wav(notes, sr=44100, out_path=Path(out))
    click.echo(f"✓ Wrote {out}")

if __name__ == "__main__":
    main()
