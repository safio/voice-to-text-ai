"""
Tests for the local transcriber module.
"""
import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from src.transcription.local_transcriber import LocalTranscriber
from src.transcription.models import TranscriptionOptions, TranscriptionResult, Segment


class TestLocalTranscriber:
    """Test cases for the LocalTranscriber class."""
    
    def test_init_with_default_options(self):
        """Test initialization with default options."""
        transcriber = LocalTranscriber()
        assert transcriber.options is not None
        assert transcriber.options.model_size == "base"
        assert transcriber.options.language is None
        assert transcriber._whisper is None
        assert transcriber._model is None
    
    def test_init_with_custom_options(self):
        """Test initialization with custom options."""
        options = TranscriptionOptions(
            model_size="small",
            language="en",
            word_timestamps=True,
            device="cuda"
        )
        transcriber = LocalTranscriber(options=options)
        assert transcriber.options == options
        assert transcriber.options.model_size == "small"
        assert transcriber.options.language == "en"
        assert transcriber.options.word_timestamps is True
        assert transcriber.options.device == "cuda"
    
    @patch("whisper.load_model")
    def test_lazy_loading_whisper(self, mock_load_model):
        """Test lazy loading of the whisper module."""
        # Setup mock
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Initialize transcriber
        transcriber = LocalTranscriber()
        
        # Whisper should not be loaded yet
        assert transcriber._whisper is None
        
        # Mock the whisper import
        with patch.dict('sys.modules', {'whisper': MagicMock()}):
            # Access the whisper property
            whisper_module = transcriber.whisper
            
            # Check that whisper was loaded
            assert transcriber._whisper is not None
            assert whisper_module is not None
    
    @patch("whisper.load_model")
    def test_lazy_loading_model(self, mock_load_model):
        """Test lazy loading of the whisper model."""
        # Setup mock
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Initialize transcriber
        transcriber = LocalTranscriber(
            options=TranscriptionOptions(model_size="tiny", device="cpu")
        )
        
        # Model should not be loaded yet
        assert transcriber._model is None
        
        # Mock the whisper import
        with patch.dict('sys.modules', {'whisper': MagicMock()}):
            # Set the whisper module directly to avoid import issues in test
            transcriber._whisper = MagicMock()
            
            # Access the model property
            model = transcriber.model
            
            # Check that the model was loaded with correct parameters
            mock_load_model.assert_called_once_with("tiny", device="cpu")
            assert model == mock_model
            
            # Accessing again should not reload the model
            model = transcriber.model
            mock_load_model.assert_called_once()
    
    def test_transcribe_file_not_found(self):
        """Test handling of non-existent input file."""
        # Initialize transcriber
        transcriber = LocalTranscriber()
        
        # Try to transcribe a non-existent file
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe("/path/to/nonexistent/file.wav")
    
    @patch.object(LocalTranscriber, "model", new_callable=MagicMock)
    def test_transcribe_wav_file(self, mock_model):
        """Test transcription of a WAV file."""
        # Setup mock result
        mock_result = {
            "text": "Hello world",
            "language": "en",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.0, "text": "Hello", "confidence": 0.9},
                {"id": 1, "start": 1.0, "end": 2.0, "text": "world", "confidence": 0.8}
            ],
            "duration": 2.0
        }
        mock_model.transcribe.return_value = mock_result
        
        # Create a test WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Initialize transcriber
            transcriber = LocalTranscriber()
            transcriber._model = mock_model  # Set model directly to avoid lazy loading
            
            # Transcribe
            result = transcriber.transcribe(tmp_path)
            
            # Check that the model was called correctly
            mock_model.transcribe.assert_called_once_with(
                tmp_path,
                word_timestamps=False
            )
            
            # Check the result
            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello world"
            assert len(result.segments) == 2
            assert result.language == "en"
            assert result.duration == 2.0
            assert result.metadata["model"] == "whisper-base"
            assert result.metadata["api_based"] is False
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @patch.object(LocalTranscriber, "model", new_callable=MagicMock)
    @patch("src.audio.converter.AudioConverter")
    def test_transcribe_m4a_file(self, mock_converter_class, mock_model):
        """Test transcription of an M4A file with conversion."""
        # Setup mocks
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        mock_converter.convert_m4a_to_wav.return_value = "/path/to/converted.wav"
        
        mock_result = {
            "text": "Converted M4A file",
            "language": "en",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "Converted M4A file", "confidence": 0.85}
            ],
            "duration": 2.0
        }
        mock_model.transcribe.return_value = mock_result
        
        # Create a test M4A file
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Initialize transcriber
            transcriber = LocalTranscriber()
            transcriber._model = mock_model  # Set model directly to avoid lazy loading
            
            # Transcribe
            result = transcriber.transcribe(tmp_path)
            
            # Check that the converter was called
            mock_converter.convert_m4a_to_wav.assert_called_once_with(Path(tmp_path))
            
            # Check that the model was called with the converted file
            mock_model.transcribe.assert_called_once_with(
                "/path/to/converted.wav",
                word_timestamps=False
            )
            
            # Check the result
            assert isinstance(result, TranscriptionResult)
            assert result.text == "Converted M4A file"
            assert len(result.segments) == 1
            assert result.language == "en"
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @patch.object(LocalTranscriber, "model", new_callable=MagicMock)
    def test_transcribe_with_language(self, mock_model):
        """Test transcription with specified language."""
        # Setup mock result
        mock_result = {
            "text": "Bonjour le monde",
            "language": "fr",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "Bonjour le monde", "confidence": 0.9}
            ],
            "duration": 2.0
        }
        mock_model.transcribe.return_value = mock_result
        
        # Create a test WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Initialize transcriber with French language
            transcriber = LocalTranscriber(
                options=TranscriptionOptions(language="fr")
            )
            transcriber._model = mock_model  # Set model directly to avoid lazy loading
            
            # Transcribe
            result = transcriber.transcribe(tmp_path)
            
            # Check that the model was called with the language option
            mock_model.transcribe.assert_called_once_with(
                tmp_path,
                word_timestamps=False,
                language="fr"
            )
            
            # Check the result
            assert result.language == "fr"
            assert result.text == "Bonjour le monde"
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
