"""
Tests for the transcriber factory module.
"""
import pytest
from unittest.mock import patch

from src.transcription.factory import get_transcriber
from src.transcription.transcriber import APITranscriber
from src.transcription.local_transcriber import LocalTranscriber
from src.transcription.models import TranscriptionOptions


class TestTranscriberFactory:
    """Test cases for the transcriber factory function."""
    
    def test_get_llm_transcriber(self):
        """Test getting an LLM (API) transcriber."""
        # Get transcriber with default options
        transcriber = get_transcriber(mode="llm")
        
        # Check that it's the correct type
        assert isinstance(transcriber, APITranscriber)
        assert transcriber.options.model_size == "base"
        
        # Get transcriber with custom options
        options = TranscriptionOptions(model_size="small", language="en")
        transcriber = get_transcriber(mode="llm", options=options)
        
        # Check that options were passed correctly
        assert isinstance(transcriber, APITranscriber)
        assert transcriber.options.model_size == "small"
        assert transcriber.options.language == "en"
    
    def test_get_local_transcriber(self):
        """Test getting a local transcriber."""
        # Get transcriber with default options
        transcriber = get_transcriber(mode="local")
        
        # Check that it's the correct type
        assert isinstance(transcriber, LocalTranscriber)
        assert transcriber.options.model_size == "base"
        
        # Get transcriber with custom options
        options = TranscriptionOptions(model_size="tiny", device="cuda")
        transcriber = get_transcriber(mode="local", options=options)
        
        # Check that options were passed correctly
        assert isinstance(transcriber, LocalTranscriber)
        assert transcriber.options.model_size == "tiny"
        assert transcriber.options.device == "cuda"
    
    def test_get_transcriber_invalid_mode(self):
        """Test handling of invalid transcription mode."""
        # Try to get transcriber with invalid mode
        with pytest.raises(ValueError) as excinfo:
            get_transcriber(mode="invalid")
        
        # Check the error message
        assert "Unsupported transcription mode: invalid" in str(excinfo.value)
