# Installation Guide

## Quick Install

### Using requirements.txt
```bash
# Install all production dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Using pyproject.toml (recommended)
```bash
# Install package with all dependencies
pip install -e .
```

## Development Setup

### For development work
```bash
# Install with development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .

# Run tests
pytest
```

## Verification

Test the installation:
```bash
# Create a test image and convert to audio
python -m image2sound.cli examples/demo.jpg --style ambient --duration 10
```

## Dependencies

### Core Requirements
- **Pillow** (≥10.0.0) - Image processing
- **numpy** (≥1.20.0) - Numerical computing
- **opencv-python** (≥4.5.0) - Computer vision for edge detection
- **scikit-learn** (≥1.0.0) - K-means clustering for color palette
- **soundfile** (≥0.10.0) - Audio file I/O
- **librosa** (≥0.9.0) - Audio processing utilities
- **click** (≥8.0.0) - CLI framework
- **pytest** (≥6.0.0) - Testing framework

### Python Version
Requires Python ≥3.10

## Troubleshooting

### Common Issues
1. **librosa installation fails**: Install system audio libraries first
   ```bash
   # macOS
   brew install portaudio
   
   # Ubuntu/Debian
   sudo apt-get install portaudio19-dev
   ```

2. **OpenCV import errors**: Try reinstalling opencv-python
   ```bash
   pip uninstall opencv-python
   pip install opencv-python
   ```

3. **Permission errors**: Use virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```