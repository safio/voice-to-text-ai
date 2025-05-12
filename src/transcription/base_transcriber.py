"""
Base transcriber interface for different transcription implementations.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from .models import TranscriptionOptions, TranscriptionResult


class BaseTranscriber(ABC):
    """
    Abstract base class for transcription implementations.
    
    This class defines the interface that all transcriber implementations must follow,
    allowing for different transcription methods (API-based, local, etc.).
    """
    
    def __init__(self, options: TranscriptionOptions = None):
        """
        Initialize the transcriber.
        
        Args:
            options (TranscriptionOptions, optional): Options for controlling the transcription.
                Defaults to None, which uses default settings.
        """
        self.options = options or TranscriptionOptions()
    
    @abstractmethod
    def transcribe(self, audio_path: Union[str, Path]) -> TranscriptionResult:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path (Union[str, Path]): Path to the audio file to transcribe.
                
        Returns:
            TranscriptionResult: The transcription result.
            
        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        pass
