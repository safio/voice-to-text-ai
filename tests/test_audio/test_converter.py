"""
Tests for the audio converter module.
"""
import os
import tempfile
from pathlib import Path
import pytest
from pydub import AudioSegment

from src.audio.converter import AudioConverter


class TestAudioConverter:
    """Test cases for the AudioConverter class."""
    
    def test_convert_m4a_to_wav_success(self):
        """Test successful conversion of M4A to WAV."""
        # Create a test M4A file
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Create a silent audio segment and save as M4A
            audio = AudioSegment.silent(duration=1000)  # 1 second of silence
            audio.export(tmp_path, format="m4a")
            
            # Initialize converter
            converter = AudioConverter()
            
            # Convert to WAV
            wav_path = converter.convert_m4a_to_wav(tmp_path)
            
            # Check that the WAV file exists
            assert os.path.exists(wav_path)
            assert wav_path.endswith(".wav")
            
            # Check that the WAV file is valid
            wav_audio = AudioSegment.from_file(wav_path, format="wav")
            assert len(wav_audio) == 1000  # Should be 1 second
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)
    
    def test_convert_m4a_to_wav_with_output_path(self):
        """Test conversion with a specified output path."""
        # Create a test M4A file
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Create output path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
            out_path = out_file.name
            os.unlink(out_path)  # Remove the file so converter can create it
        
        try:
            # Create a silent audio segment and save as M4A
            audio = AudioSegment.silent(duration=1000)
            audio.export(tmp_path, format="m4a")
            
            # Initialize converter
            converter = AudioConverter()
            
            # Convert to WAV with specified output path
            wav_path = converter.convert_m4a_to_wav(tmp_path, out_path)
            
            # Check that the WAV file exists at the specified path
            assert os.path.exists(wav_path)
            assert wav_path == out_path
            
            # Check that the WAV file is valid
            wav_audio = AudioSegment.from_file(wav_path, format="wav")
            assert len(wav_audio) == 1000  # Should be 1 second
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if os.path.exists(out_path):
                os.unlink(out_path)
    
    def test_convert_m4a_to_wav_file_not_found(self):
        """Test handling of non-existent input file."""
        # Initialize converter
        converter = AudioConverter()
        
        # Try to convert a non-existent file
        with pytest.raises(FileNotFoundError):
            converter.convert_m4a_to_wav("/path/to/nonexistent/file.m4a")
    
    def test_convert_m4a_to_wav_invalid_extension(self):
        """Test handling of input file with invalid extension."""
        # Create a test file with wrong extension
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Initialize converter
            converter = AudioConverter()
            
            # Try to convert a file with wrong extension
            with pytest.raises(ValueError):
                converter.convert_m4a_to_wav(tmp_path)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
