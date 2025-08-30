"""Smoke tests for the Gradio UI module."""

import pytest


def test_ui_import():
    """Test that the UI module can be imported without errors."""
    try:
        import image2sound.ui
        assert hasattr(image2sound.ui, 'build_interface')
        assert hasattr(image2sound.ui, 'main')
    except ImportError as e:
        pytest.skip(f"UI dependencies not installed: {e}")


def test_build_interface():
    """Test that build_interface returns a Gradio Blocks object without launching."""
    try:
        from image2sound.ui import build_interface
        
        # Build the interface (should not launch or block)
        demo = build_interface()
        
        # Should return a Gradio Blocks object
        assert demo is not None
        assert hasattr(demo, 'launch')  # Should have launch method
        assert hasattr(demo, 'blocks')  # Should have blocks attribute
        
        # Should not be running/launched
        assert not hasattr(demo, 'server') or demo.server is None
        
    except ImportError as e:
        pytest.skip(f"UI dependencies not installed: {e}")


def test_reveal_in_filesystem():
    """Test file system reveal function with non-existent file."""
    try:
        from image2sound.ui import reveal_in_filesystem
        
        # Test with non-existent file
        result = reveal_in_filesystem("/non/existent/file.txt")
        assert "File not found" in result
        assert "❌" in result
        
    except ImportError as e:
        pytest.skip(f"UI dependencies not installed: {e}")


def test_build_output_path():
    """Test output path building functionality."""
    try:
        from image2sound.ui import build_output_path
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with explicit directory and filename
            output_path = build_output_path(temp_dir, "test_music", "/path/to/image.jpg")
            assert output_path.name == "test_music.wav"
            assert output_path.parent == Path(temp_dir)
            
            # Test with empty filename (should use image name)
            output_path = build_output_path(temp_dir, "", "/path/to/beautiful_image.png")
            assert output_path.name == "beautiful_image.wav"
            
            # Test with whitespace filename (should use image name)
            output_path = build_output_path(temp_dir, "   ", "/path/to/sunset.jpg")
            assert output_path.name == "sunset.wav"
            
            # Test with no image path (should use default)
            output_path = build_output_path(temp_dir, "", "")
            assert output_path.name == "generated_music.wav"
            
            # Test with empty directory (should use Downloads)
            output_path = build_output_path("", "test_music", "/path/to/image.jpg")
            assert output_path.name == "test_music.wav"
            assert "Downloads" in str(output_path)
            assert "image2sound" in str(output_path)
            
    except ImportError as e:
        pytest.skip(f"UI dependencies not installed: {e}")