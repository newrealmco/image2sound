"""Gradio web UI for image2sound with soothing design and comprehensive functionality."""

import os
import platform
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from .cli import extract_features, map_features_to_music, compose_track, render_wav


def reveal_in_filesystem(file_path: str) -> str:
    """Open the file's location in the system file manager.
    
    Args:
        file_path: Path to the file to reveal
        
    Returns:
        Status message indicating success or failure
    """
    if not os.path.exists(file_path):
        return f"❌ File not found: {file_path}"
    
    try:
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            # Try to select the file, fallback to opening directory
            try:
                subprocess.run(["open", "-R", file_path], check=True)
                return "✅ Opened in Finder"
            except subprocess.CalledProcessError:
                # Fallback to opening directory
                directory = os.path.dirname(file_path)
                subprocess.run(["open", directory], check=True)
                return "✅ Opened directory in Finder"
                
        elif system == "windows":
            # Use explorer to select the file
            subprocess.run(["explorer", "/select,", file_path], check=True)
            return "✅ Opened in Explorer"
            
        else:  # Linux and other Unix-like systems
            # Open the directory containing the file
            directory = os.path.dirname(file_path)
            subprocess.run(["xdg-open", directory], check=True)
            return "✅ Opened directory in file manager"
            
    except subprocess.CalledProcessError as e:
        return f"❌ Failed to open file manager: {e}"
    except FileNotFoundError:
        return "❌ File manager not found on this system"


def build_output_path(output_dir: str, filename_stem: str, image_path: str) -> Path:
    """Build the complete output path for the generated audio file.
    
    Args:
        output_dir: Directory to save the file in (empty for default Downloads)
        filename_stem: Base filename (without extension)
        image_path: Path to source image (for fallback naming)
        
    Returns:
        Complete path for the output .wav file
    """
    # Use provided directory or default to Downloads/image2sound
    if output_dir.strip():
        output_path = Path(output_dir.strip())
    else:
        # Default to Downloads/image2sound folder
        downloads_dir = Path.home() / "Downloads" / "image2sound"
        output_path = downloads_dir
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use provided filename or derive from image
    if filename_stem.strip():
        base_name = filename_stem.strip()
    else:
        base_name = Path(image_path).stem if image_path else "generated_music"
    
    # Ensure .wav extension
    return output_path / f"{base_name}.wav"


def generate_music(
    image_file: str,
    style: str,
    duration: float,
    seed_text: str,
    output_dir: str,
    filename_stem: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True)
) -> Tuple[str, Optional[str], str, str]:
    """Generate music from an image using the image2sound pipeline.
    
    Args:
        image_file: Path to uploaded image file
        style: Musical style ("neutral", "ambient", "cinematic", "rock")
        duration: Duration in seconds
        seed_text: Optional seed value as string
        output_dir: Output directory path
        filename_stem: Base filename for output
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (summary_text, audio_path, status_message, reveal_status)
    """
    if not image_file:
        return "❌ No image provided", None, "Please upload an image first.", ""
    
    if not os.path.exists(image_file):
        return "❌ Image file not found", None, "The uploaded image file could not be found.", ""
    
    try:
        progress(0.1, desc="🔍 Extracting image features...")
        
        # Parse seed if provided, otherwise use None for image-derived seed
        seed = None
        if seed_text.strip():
            try:
                seed = int(seed_text.strip())
            except ValueError:
                return "❌ Invalid seed", None, f"Seed must be an integer, got: {seed_text}", ""
        
        # Extract features from image
        from pathlib import Path
        features = extract_features(Path(image_file))
        
        # Apply seed if provided (otherwise use image-derived seed)
        if seed is not None:
            features.seed = seed
        
        progress(0.4, desc="🎵 Mapping features to music...")
        
        # Map to musical parameters
        music_params = map_features_to_music(features, style=style, target_duration=duration)
        
        progress(0.7, desc="🎼 Composing musical arrangement...")
        
        # Compose the track
        notes = compose_track(music_params)
        
        progress(0.9, desc="🔊 Rendering audio file...")
        
        # Build final output path
        final_output_path = build_output_path(output_dir, filename_stem, image_file)
        
        # Generate audio in a temporary file first (for Gradio compatibility)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Render to temporary WAV file
        render_wav(notes, 44100, temp_path)
        
        # Copy to final location if different from temp
        if str(final_output_path) != temp_path:
            try:
                shutil.copy2(temp_path, final_output_path)
            except Exception as e:
                # If copy fails, we'll still return the temp file for Gradio to serve
                pass
        
        progress(1.0, desc="✨ Generation complete!")
        
        # Build summary text
        summary_lines = [
            f"🎵 **Musical Summary**",
            f"**Key:** {music_params.root} {music_params.mode}",
            f"**BPM:** {music_params.bpm}",
            f"**Meter:** {music_params.meter[0]}/{music_params.meter[1]}",
            f"**Chord Progression:** {' → '.join(music_params.progression)}",
            f"**Duration:** {duration:.1f}s",
            f"**Voices:** {len(music_params.voices)} color-derived instruments",
            f"**Style:** {style.title()}",
        ]
        
        if music_params.has_complement:
            summary_lines.append("**Special:** Complementary colors detected - added altered V chords")
        
        summary_lines.extend([
            "",
            f"🎨 **Color Analysis:**"
        ])
        
        for i, voice in enumerate(music_params.voices[:3]):  # Show top 3 colors
            color = voice.color
            summary_lines.append(
                f"**Color {i+1}:** RGB{color.rgb} → {voice.instrument} "
                f"(prop={color.prop:.2f}, sat={color.sat:.2f})"
            )
        
        if len(music_params.voices) > 3:
            summary_lines.append(f"... and {len(music_params.voices) - 3} more colors")
        
        summary_text = "\n".join(summary_lines)
        
        # Success status with file locations
        temp_file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        
        if str(final_output_path) != temp_path and os.path.exists(final_output_path):
            status_message = f"✅ Generated successfully! Saved to: {final_output_path.name} ({temp_file_size_mb:.1f} MB)"
            # Store final path for reveal functionality
            reveal_path = str(final_output_path)
        else:
            status_message = f"✅ Generated successfully! Temporary file ({temp_file_size_mb:.1f} MB)"
            reveal_path = temp_path
        
        # Return temp path for Gradio to serve, store final path info in status
        return summary_text, temp_path, status_message, reveal_path
        
    except Exception as e:
        error_msg = f"❌ Generation failed: {str(e)}"
        progress(0.0, desc="Failed")
        return error_msg, None, error_msg, ""


def build_interface() -> gr.Blocks:
    """Build and return the Gradio interface without launching it.
    
    Returns:
        Configured Gradio Blocks interface
    """
    # Create custom theme with soft colors and teal accent
    theme = gr.themes.Soft(
        primary_hue="teal",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    )
    
    with gr.Blocks(
        theme=theme,
        title="image2sound - Visual Music Generation",
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .control-column { padding-right: 1rem; }
        .results-column { padding-left: 1rem; }
        .status-box { padding: 1rem; border-radius: 8px; margin: 1rem 0; }
        .summary-box { 
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-left: 4px solid #0d9488;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        """
    ) as demo:
        
        # Header
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    # 🎵 image2sound
                    ### Transform images into beautiful, unique music
                    
                    Upload an image and watch it become a musical composition. Each color, texture, 
                    and visual element is carefully translated into musical parameters to create 
                    a piece that captures the essence of your image.
                    """,
                    elem_classes=["main-header"]
                )
        
        # Main interface
        with gr.Row(equal_height=True):
            # Left column: Controls
            with gr.Column(scale=1, elem_classes=["control-column"]):
                gr.Markdown("### 🎨 **Image & Settings**")
                
                # Image upload
                image_input = gr.File(
                    label="Upload Image",
                    file_types=[".jpg", ".jpeg", ".png", ".webp"],
                    type="filepath"
                )
                
                # Style selection
                style_input = gr.Radio(
                    choices=["neutral", "ambient", "cinematic", "rock"],
                    value="ambient",
                    label="Musical Style",
                    info="Each style creates different moods and tempos"
                )
                
                # Duration slider
                duration_input = gr.Slider(
                    minimum=5,
                    maximum=60,
                    value=20,
                    step=1,
                    label="Duration (seconds)",
                    info="Length of the generated music"
                )
                
                # Advanced settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    seed_input = gr.Textbox(
                        label="Seed (optional)",
                        placeholder="Leave empty for image-derived seed",
                        info="Integer seed for reproducible results"
                    )
                    
                    output_dir_input = gr.Textbox(
                        label="Output Directory (optional)",
                        value="",
                        placeholder="Leave empty to use downloads folder",
                        info="Where to save generated music files permanently"
                    )
                    
                    filename_input = gr.Textbox(
                        label="Filename (without extension)",
                        placeholder="Leave empty to use image name",
                        info="Base name for the output .wav file"
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "🎵 Generate Music",
                    variant="primary",
                    size="lg"
                )
            
            # Right column: Progress and Results
            with gr.Column(scale=1, elem_classes=["results-column"]):
                gr.Markdown("### 🎼 **Results**")
                
                # Status display
                status_display = gr.Textbox(
                    label="Status",
                    value="Ready to generate music from your image",
                    interactive=False,
                    elem_classes=["status-box"]
                )
                
                # Summary display
                summary_display = gr.Markdown(
                    "Upload an image and click Generate to see musical analysis here.",
                    elem_classes=["summary-box"]
                )
                
                # Audio player
                audio_player = gr.Audio(
                    label="Generated Music",
                    visible=False
                )
                
                # File actions
                with gr.Row(visible=False) as file_actions:
                    reveal_btn = gr.Button(
                        "📁 Show in File Manager",
                        size="sm"
                    )
                
                # Reveal status
                reveal_status = gr.Textbox(
                    label="File Manager",
                    visible=False,
                    interactive=False
                )
        
        # State to store the current audio file path
        audio_path_state = gr.State(value="")
        
        # Wire up the generate button
        def on_generate(*args):
            """Handle generate button click with proper UI updates."""
            summary, audio_path, status, reveal_path = generate_music(*args)
            
            # Update UI based on whether generation was successful
            if audio_path:
                return (
                    summary,  # summary_display
                    status,   # status_display
                    audio_path,  # audio_player value
                    reveal_path,  # audio_path_state (for reveal functionality)
                    gr.update(visible=True),  # audio_player visibility
                    gr.update(visible=True),  # file_actions visibility
                    gr.update(visible=True),  # reveal_status visibility
                )
            else:
                return (
                    summary,  # summary_display
                    status,   # status_display
                    None,     # audio_player value
                    "",       # audio_path_state
                    gr.update(visible=False),  # audio_player visibility
                    gr.update(visible=False),  # file_actions visibility
                    gr.update(visible=False),  # reveal_status visibility
                )
        
        generate_btn.click(
            fn=on_generate,
            inputs=[
                image_input, style_input, duration_input, seed_input,
                output_dir_input, filename_input
            ],
            outputs=[
                summary_display, status_display, audio_player, 
                audio_path_state, audio_player, file_actions, reveal_status
            ]
        )
        
        # Wire up the reveal button
        def on_reveal(audio_path):
            """Handle reveal button click."""
            if audio_path:
                return reveal_in_filesystem(audio_path)
            return "❌ No file to reveal"
        
        reveal_btn.click(
            fn=on_reveal,
            inputs=[audio_path_state],
            outputs=[reveal_status]
        )
        
        # Footer
        with gr.Row():
            gr.Markdown(
                """
                ---
                *Built with [Gradio](https://gradio.app) • 
                Powered by image2sound • 
                Transform your visual world into music*
                """,
                elem_classes=["main-header"]
            )
    
    return demo


def main() -> None:
    """Launch the image2sound web UI."""
    print("🎵 Starting image2sound web interface...")
    
    try:
        # Build the interface
        demo = build_interface()
        
        print("🌐 Launching web interface...")
        print("   - The interface will open in your default browser")
        print("   - Use Ctrl+C to stop the server")
        
        # Launch with browser opening and default port selection
        demo.launch(
            inbrowser=True,
            prevent_thread_lock=False,  # Keep the main thread alive
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down image2sound UI...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to start UI: {e}")
        print("\nTroubleshooting:")
        print("- Make sure no other application is using port 7860")
        print("- Try running: lsof -ti:7860 | xargs kill -9")
        print("- Or manually specify a different port in the code")
        sys.exit(1)


if __name__ == "__main__":
    main()