# image2sound
Turn images into short musical pieces via algorithmic sonification.

## Quickstart
```bash
pip install -e .
python -m image2sound.cli examples/demo.jpg -o out.wav --style ambient --duration 20
```

## How it works
- **Extract features**: Brightness, contrast, edge density, and 5-color palette from image
- **Map to music**: Hue → key, brightness → BPM/scale, contrast+edges → intensity
- **Compose arrangement**: 4/4 time with chords, lead melody, bass, and drums
- **Synthesize audio**: Sine waves with harmonics and ADSR, drums as noise bursts

## Styles
- **`neutral`**: Balanced mapping, piano/lead/drums
- **`ambient`**: Slower, major scale, soft pad/lead/bass instruments
- **`cinematic`**: Faster tempo, orchestral pad/lead/bass
- **`rock`**: Fastest, minor scale, piano/lead/drums with punch

## Examples
```bash
# Basic usage
python -m image2sound.cli photo.jpg

# Custom style and duration
python -m image2sound.cli landscape.png --style cinematic --duration 30

# Output to specific file
python -m image2sound.cli portrait.jpg -o music.wav --style rock --duration 15
```

## Install & test locally
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest -q
python -m image2sound.cli examples/demo.jpg -o out.wav --style ambient
open out.wav  # (macOS) listen
```
