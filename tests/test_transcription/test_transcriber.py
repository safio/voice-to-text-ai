"""
Tests for the transcriber module.
"""
import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.transcription.transcriber import Transcriber
from src.transcription.models import TranscriptionOptions, TranscriptionResult, Segment


class TestTranscriber:
    """Test cases for the Transcriber class."""
    
    def test_init_with_default_options(self):
        """Test initialization with default options."""
        transcriber = Transcriber()
        assert transcriber.options is not None
        assert transcriber.options.model_size == "base"
        assert transcriber.options.language is None
        assert transcriber.model is None
    
    def test_init_with_custom_options(self):
        """Test initialization with custom options."""
        options = TranscriptionOptions(
            model_size="small",
            language="en",
            word_timestamps=True
        )
        transcriber = Transcriber(options=options)
        assert transcriber.options == options
        assert transcriber.options.model_size == "small"
        assert transcriber.options.language == "en"
        assert transcriber.options.word_timestamps is True
    
    @patch("whisper.load_model")
    def test_load_model(self, mock_load_model):
        """Test loading the Whisper model."""
        # Setup mock
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Initialize transcriber
        transcriber = Transcriber(
            options=TranscriptionOptions(model_size="tiny", device="cpu")
        )
        
        # Load model
        transcriber.load_model()
        
        # Check that the model was loaded with correct parameters
        mock_load_model.assert_called_once_with("tiny", device="cpu")
        assert transcriber.model == mock_model
        
        # Loading again should not call load_model again
        transcriber.load_model()
        mock_load_model.assert_called_once()
    
    @patch("whisper.load_model")
    def test_transcribe_file_not_found(self, mock_load_model):
        """Test handling of non-existent input file."""
        # Initialize transcriber
        transcriber = Transcriber()
        
        # Try to transcribe a non-existent file
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe("/path/to/nonexistent/file.wav")
        
        # Model should not be loaded
        mock_load_model.assert_not_called()
    
    @patch.object(Transcriber, "_transcribe_audio")
    @patch.object(Transcriber, "_is_long_audio")
    @patch("whisper.load_model")
    def test_transcribe_short_audio(self, mock_load_model, mock_is_long_audio, mock_transcribe_audio):
        """Test transcription of a short audio file."""
        # Setup mocks
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_is_long_audio.return_value = False
        
        # Create expected result
        expected_result = TranscriptionResult(
            text="Hello world",
            segments=[
                Segment(id=0, start=0.0, end=1.0, text="Hello", confidence=0.9),
                Segment(id=1, start=1.0, end=2.0, text="world", confidence=0.8)
            ],
            language="en",
            confidence=0.85,
            duration=2.0,
            metadata={}
        )
        mock_transcribe_audio.return_value = expected_result
        
        # Create a test WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Initialize transcriber
            transcriber = Transcriber()
            
            # Transcribe
            result = transcriber.transcribe(tmp_path)
            
            # Check that the model was loaded
            assert transcriber.model == mock_model
            
            # Check that the correct methods were called
            mock_is_long_audio.assert_called_once_with(Path(tmp_path))
            mock_transcribe_audio.assert_called_once_with(Path(tmp_path))
            
            # Check the result
            assert result == expected_result
            assert result.text == "Hello world"
            assert len(result.segments) == 2
            assert result.language == "en"
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @patch.object(Transcriber, "_transcribe_long_audio")
    @patch.object(Transcriber, "_is_long_audio")
    @patch("whisper.load_model")
    def test_transcribe_long_audio(self, mock_load_model, mock_is_long_audio, mock_transcribe_long_audio):
        """Test transcription of a long audio file."""
        # Setup mocks
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_is_long_audio.return_value = True
        
        # Create expected result
        expected_result = TranscriptionResult(
            text="This is a long audio file with multiple segments",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="This is a", confidence=0.9),
                Segment(id=1, start=2.0, end=4.0, text="long audio file", confidence=0.8),
                Segment(id=2, start=4.0, end=6.0, text="with multiple segments", confidence=0.7)
            ],
            language="en",
            confidence=0.8,
            duration=6.0,
            metadata={"chunked": True}
        )
        mock_transcribe_long_audio.return_value = expected_result
        
        # Create a test WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Initialize transcriber
            transcriber = Transcriber()
            
            # Transcribe
            result = transcriber.transcribe(tmp_path)
            
            # Check that the model was loaded
            assert transcriber.model == mock_model
            
            # Check that the correct methods were called
            mock_is_long_audio.assert_called_once_with(Path(tmp_path))
            mock_transcribe_long_audio.assert_called_once_with(Path(tmp_path))
            
            # Check the result
            assert result == expected_result
            assert result.text == "This is a long audio file with multiple segments"
            assert len(result.segments) == 3
            assert result.language == "en"
            assert result.metadata.get("chunked") is True
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
