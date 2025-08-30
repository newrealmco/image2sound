# image2sound
Turn images into short musical pieces via algorithmic sonification.

## Installation

### PyPI Installation
```bash
pip install image2sound-newrealm                # core only
pip install 'image2sound-newrealm[ui]'          # + Gradio UI
pip install 'image2sound-newrealm[opencv]'      # + OpenCV feature extraction
pip install 'image2sound-newrealm[audio]'       # + Soundfile output
pip install 'image2sound-newrealm[ml]'          # + scikit-learn clustering
pip install 'image2sound-newrealm[full]'        # all extras
```

### Development Setup (Current)
```bash
# Clone the repository
git clone https://github.com/newrealmco/image2sound.git
cd image2sound

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with UI support
pip install -e .[ui]
```

## Quickstart

### GUI (Recommended)
```bash
image2sound-ui
```
Opens a beautiful web interface in your browser where you can:
- Upload images with drag & drop
- Choose musical styles and settings  
- Watch real-time generation progress
- Play generated music instantly
- Download files and open in file manager

### Command Line
```bash
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

## Batch demo
Generate a comprehensive demo set with musical metadata for analysis:
```bash
# Run batch processing on all images in examples/
python scripts/batch_demo.py

# Creates files like: bright_gradient_ambient_C_ionian_95bpm_4-4_I-V-vi-IV.wav
# Outputs CSV with: file,image,style,bpm,key,mode,meter,progression,seed,brightness,contrast,edge_density
```

The batch script processes all images in `examples/` with each of the 4 styles (neutral, ambient, cinematic, rock), creating descriptive filenames that include the musical parameters and outputting detailed CSV metadata for analysis.

## Development & Testing
```bash
# Install with development dependencies
pip install -e .[ui] -r requirements-dev.txt

# Run tests
pytest -q

# Test CLI
python -m image2sound.cli examples/demo.jpg -o out.wav --style ambient

# Test UI (opens in browser)
image2sound-ui
```
