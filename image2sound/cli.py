import click
from pathlib import Path

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
    # Import heavy modules only when actually running, not for --help
    from .features import extract_features
    from .mapping import map_features_to_music
    from .compose import compose_track
    from .synth import render_wav
    
    print("=" * 60)
    print("🎨✨ IMAGE2SOUND: Algorithmic Sonification ✨🎵")
    print("=" * 60)
    print(f"🖼️  Input: {image_path}")
    print(f"🎼 Style: {style}")
    print(f"⏱️  Duration: {duration}s")
    print(f"📁 Output: {out}")
    print()
    
    print("🚀 Starting sonification pipeline...")
    print()
    
    # Step 1: Feature Extraction
    print("🔍 STEP 1/4: Visual Feature Extraction")
    print("-" * 40)
    feats = extract_features(Path(image_path))
    print()
    
    # Step 2: Musical Mapping  
    print("🎵 STEP 2/4: Musical Parameter Mapping")
    print("-" * 40)
    params = map_features_to_music(feats, style=style, target_duration=duration)
    print()
    
    # Step 3: Composition
    print("🎼 STEP 3/4: Musical Composition")
    print("-" * 40)
    notes = compose_track(params)
    print()
    
    # Step 4: Audio Synthesis
    print("🎚️  STEP 4/4: Audio Synthesis")
    print("-" * 40)
    render_wav(notes, sr=44100, out_path=Path(out))
    print()
    
    print("=" * 60)
    print("🎉 SONIFICATION COMPLETE! 🎉")
    print("=" * 60)
    print(f"🎵 Your image has been transformed into music!")
    print(f"📁 Audio saved to: {out}")
    print()
    print("📊 Musical Summary:")
    print(f"   🎹 Key: {params.root} {params.mode}")
    print(f"   🥁 Tempo: {params.bpm} BPM in {params.meter[0]}/{params.meter[1]} time")
    print(f"   🎼 Progression: {' → '.join(params.progression)}")
    print(f"   🎲 Seed: {feats.seed} (deterministic from image)")
    print()
    print(f"🎧 Ready to play - enjoy your sonic artwork! ✨")
    print("=" * 60)

if __name__ == "__main__":
    main()
