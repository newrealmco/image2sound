# image2sound
Turn images into short musical pieces via algorithmic sonification.

## Quickstart
```bash
pip install -e .
python -m image2sound.cli examples/demo.jpg -o out.wav --style ambient --duration 20
```

## Install & smoke-test locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest -q
python -m image2sound.cli examples/demo.jpg -o out.wav --style ambient
open out.wav  # (macOS) listen
```
