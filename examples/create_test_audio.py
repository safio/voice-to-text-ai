"""
Create a test M4A file for testing the voice-to-text agent.

This script generates a simple audio file with a sine wave tone
that can be used for testing the voice-to-text agent.
"""
import os
import sys
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine

# Add the parent directory to the path so we can import the src package
sys.path.append(str(Path(__file__).parent.parent))


def create_test_audio(output_path, duration_ms=5000, sample_rate=44100):
    """
    Create a test audio file with a sine wave tone.
    
    Args:
        output_path (str): Path to save the output M4A file.
        duration_ms (int, optional): Duration in milliseconds. Defaults to 5000 (5 seconds).
        sample_rate (int, optional): Sample rate in Hz. Defaults to 44100.
    
    Returns:
        str: Path to the created file.
    """
    print(f"Creating test audio file: {output_path}")
    
    # Create a sine wave tone
    sine_wave = Sine(440)  # 440 Hz = A4 note
    audio = sine_wave.to_audio_segment(duration=duration_ms)
    
    # Export as M4A
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    audio.export(output_path, format="mp4", parameters=["-c:a", "aac"])
    
    print(f"Test audio file created: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Default output path
    default_output_path = Path(__file__).parent / "test_audio.m4a"
    
    # Get output path if provided
    output_path = sys.argv[1] if len(sys.argv) > 1 else default_output_path
    
    # Create the test audio file
    create_test_audio(output_path)
