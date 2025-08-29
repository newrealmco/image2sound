import click
from pathlib import Path
from .features import extract_features
from .mapping import map_features_to_music
from .compose import compose_track
from .synth import render_wav

@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--out", "-o", type=click.Path(), default="out.wav")
@click.option("--style", type=click.Choice(["neutral","ambient","cinematic","rock"]), default="neutral")
@click.option("--duration", type=float, default=20.0, help="seconds")
def main(image_path, out, style, duration):
    feats = extract_features(Path(image_path))
    params = map_features_to_music(feats, style=style, target_duration=duration)
    notes = compose_track(params)
    render_wav(notes, sr=44100, out_path=Path(out))
    click.echo(f"✓ Wrote {out}")
