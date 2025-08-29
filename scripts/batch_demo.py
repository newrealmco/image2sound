#!/usr/bin/env python3
"""Batch demo script for image2sound.

Processes all images in examples/ directory with different styles and generates
CSV metadata output for analysis and demonstration purposes.
"""

import sys
import csv
from pathlib import Path
from PIL import Image

# Add parent directory to path to import image2sound
sys.path.insert(0, str(Path(__file__).parent.parent))

from image2sound.features import extract_features
from image2sound.mapping import map_features_to_music
from image2sound.compose import compose_track
from image2sound.synth import render_wav


def create_demo_images():
    """Create a set of demo images with different visual characteristics."""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    print("🎨 Creating demo images...")
    
    # 1. Bright colorful gradient
    img = Image.new('RGB', (200, 200))
    pixels = []
    for y in range(200):
        for x in range(200):
            r = int(255 * (x / 200))
            g = int(255 * (y / 200))
            b = int(200)
            pixels.append((r, g, b))
    img.putdata(pixels)
    img.save(examples_dir / "bright_gradient.jpg")
    
    # 2. Dark moody image
    img = Image.new('RGB', (200, 200))
    pixels = []
    for y in range(200):
        for x in range(200):
            r = int(50 + 30 * (x / 200))
            g = int(20 + 40 * (y / 200))
            b = int(80 + 20 * ((x + y) / 400))
            pixels.append((r, g, b))
    img.putdata(pixels)
    img.save(examples_dir / "dark_moody.jpg")
    
    # 3. High contrast geometric
    img = Image.new('RGB', (200, 200))
    pixels = []
    for y in range(200):
        for x in range(200):
            if (x // 40 + y // 40) % 2 == 0:
                pixels.append((255, 255, 255))
            else:
                pixels.append((0, 0, 0))
    img.putdata(pixels)
    img.save(examples_dir / "geometric_contrast.jpg")
    
    # 4. Warm sunset colors
    img = Image.new('RGB', (200, 200))
    pixels = []
    for y in range(200):
        for x in range(200):
            r = int(255 * (1 - y / 400))
            g = int(150 * (1 - y / 300))
            b = int(50 + 100 * (y / 200))
            pixels.append((r, g, b))
    img.putdata(pixels)
    img.save(examples_dir / "warm_sunset.jpg")
    
    # 5. Cool ocean colors
    img = Image.new('RGB', (200, 200))
    pixels = []
    for y in range(200):
        for x in range(200):
            r = int(30 + 50 * (y / 200))
            g = int(100 + 100 * (x / 200))
            b = int(180 + 50 * (1 - y / 200))
            pixels.append((r, g, b))
    img.putdata(pixels)
    img.save(examples_dir / "cool_ocean.jpg")
    
    print(f"   ✅ Created 5 demo images in {examples_dir}/")


def safe_filename(text: str) -> str:
    """Convert text to safe filename by replacing problematic characters."""
    return text.replace("/", "-").replace(" ", "_").replace("#", "sharp")


def process_image(image_path: Path, style: str, out_dir: Path) -> dict:
    """Process a single image with given style and return metadata."""
    print(f"🎵 Processing {image_path.name} with {style} style...")
    
    # Extract features
    features = extract_features(image_path)
    
    # Map to musical parameters
    params = map_features_to_music(features, style=style, target_duration=10.0)
    
    # Compose track
    notes = compose_track(params)
    
    # Create filename with musical metadata
    meter_str = f"{params.meter[0]}-{params.meter[1]}"
    progression_str = "-".join(params.progression)[:20]  # Limit length
    safe_progression = safe_filename(progression_str)
    
    filename = (f"{image_path.stem}_{style}_{params.root}_{params.mode}_"
                f"{params.bpm}bpm_{meter_str}_{safe_progression}.wav")
    
    output_path = out_dir / filename
    
    # Render audio
    render_wav(notes, sr=44100, out_path=output_path)
    
    # Return metadata
    return {
        'file': str(output_path),
        'image': image_path.name,
        'style': style,
        'bpm': params.bpm,
        'key': params.root,
        'mode': params.mode,
        'meter': f"{params.meter[0]}/{params.meter[1]}",
        'progression': ' -> '.join(params.progression),
        'seed': features.seed,
        'brightness': round(features.brightness, 3),
        'contrast': round(features.contrast, 3),
        'edge_density': round(features.edge_density, 3)
    }


def main():
    """Main batch processing function."""
    print("🚀 Starting batch demo generation...")
    print("=" * 60)
    
    # Setup paths
    examples_dir = Path("examples")
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    
    # Create demo images if examples directory is sparse
    if len(list(examples_dir.glob("*.jpg"))) < 3:
        create_demo_images()
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(examples_dir.glob(f"*{ext}"))
        image_files.extend(examples_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("❌ No images found in examples/ directory")
        return
    
    print(f"📸 Found {len(image_files)} images: {[f.name for f in image_files]}")
    print()
    
    # Styles to process
    styles = ['neutral', 'ambient', 'cinematic', 'rock']
    
    # Process all combinations
    metadata_list = []
    total_combinations = len(image_files) * len(styles)
    processed = 0
    
    print(f"🎼 Processing {total_combinations} combinations...")
    print()
    
    # CSV header
    print("CSV Output:")
    print("file,image,style,bpm,key,mode,meter,progression,seed,brightness,contrast,edge_density")
    
    for image_path in sorted(image_files):
        for style in styles:
            try:
                metadata = process_image(image_path, style, out_dir)
                metadata_list.append(metadata)
                processed += 1
                
                # Print CSV line
                csv_line = (f"{metadata['file']},{metadata['image']},{metadata['style']},"
                           f"{metadata['bpm']},{metadata['key']},{metadata['mode']},"
                           f"{metadata['meter']},\"{metadata['progression']}\","
                           f"{metadata['seed']},{metadata['brightness']},"
                           f"{metadata['contrast']},{metadata['edge_density']}")
                print(csv_line)
                
                print(f"   ✅ [{processed}/{total_combinations}] {image_path.name} ({style}) → {metadata['key']} {metadata['mode']}, {metadata['bpm']} BPM")
                
            except Exception as e:
                print(f"   ❌ Error processing {image_path.name} ({style}): {e}")
                processed += 1
    
    print()
    print("=" * 60)
    print("🎉 Batch demo generation complete!")
    print(f"📊 Processed: {len(metadata_list)} successful renders")
    print(f"📁 Output directory: {out_dir.absolute()}")
    
    # Save CSV file
    csv_path = out_dir / "batch_demo_metadata.csv"
    if metadata_list:
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = metadata_list[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_list)
        print(f"📊 Metadata saved to: {csv_path}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()