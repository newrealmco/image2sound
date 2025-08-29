from pathlib import Path
from PIL import Image
from image2sound.cli import main as cli_main
import subprocess, sys

def test_cli_smoke(tmp_path: Path):
    img = tmp_path / "img.png"
    Image.new("RGB", (64,64), color=(100, 180, 220)).save(img)
    out = tmp_path / "out.wav"
    cmd = [sys.executable, "-m", "image2sound.cli", str(img), "--out", str(out)]
    # run as module to exercise click command
    subprocess.check_call(cmd)
    assert out.exists() and out.stat().st_size > 1000
