"""
API-based transcriber module for converting audio to text using OpenAI's API.
"""
import os
import tempfile
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import time
from loguru import logger
import openai
from tqdm import tqdm

from .base_transcriber import BaseTranscriber
from .models import TranscriptionOptions, TranscriptionResult, Segment


class APITranscriber(BaseTranscriber):
    """
    Handles transcription of audio files to text using OpenAI's API.
    
    This class provides functionality to transcribe audio files using the
    OpenAI API for speech recognition.
    """
    
    def __init__(self, options: Optional[TranscriptionOptions] = None):
        """
        Initialize the APITranscriber.
        
        Args:
            options (Optional[TranscriptionOptions]): Options for controlling the transcription.
                Defaults to None, which uses default settings.
        """
        super().__init__(options)
        logger.info(f"Initializing APITranscriber with options: {self.options}")
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found in environment variables")
    
    def transcribe(self, audio_path: Union[str, Path]) -> TranscriptionResult:
        """
        Transcribe an audio file to text using OpenAI's API.
        
        Args:
            audio_path (Union[str, Path]): Path to the audio file to transcribe.
                
        Returns:
            TranscriptionResult: The transcription result.
            
        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        audio_path = Path(audio_path)
        
        # Validate audio file
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing audio file: {audio_path}")
        
        # Process audio file
        start_time = time.time()
        
        try:
            # For WAV files, use directly; for M4A, convert first
            if audio_path.suffix.lower() == ".m4a":
                # Import here to avoid circular imports
                from ..audio.converter import AudioConverter
                converter = AudioConverter()
                audio_path = Path(converter.convert_m4a_to_wav(audio_path))
                logger.info(f"Converted M4A to WAV: {audio_path}")
            
            # Open the audio file
            with open(audio_path, "rb") as audio_file:
                # Call OpenAI API for transcription
                logger.info(f"Calling OpenAI API for transcription")
                response = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=self.options.language,
                    response_format="verbose_json"
                )
            
            # Process the response
            result = self._process_api_response(response, audio_path)
            
            elapsed_time = time.time() - start_time
            logger.success(f"Transcription completed in {elapsed_time:.2f} seconds")
            
            return result
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {str(e)}")
            raise
    
    def _process_api_response(self, response, audio_path: Path) -> TranscriptionResult:
        """
        Process the API response from OpenAI.
        
        Args:
            response: The API response from OpenAI.
            audio_path (Path): Path to the audio file.
                
        Returns:
            TranscriptionResult: The transcription result.
        """
        # Extract data from response
        text = response.text
        language = response.language
        
        # Create segments from the response
        segments = []
        if hasattr(response, 'segments') and response.segments:
            for i, segment in enumerate(response.segments):
                segments.append(Segment(
                    id=i,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    confidence=getattr(segment, 'confidence', 1.0)
                ))
        
        # If no segments provided, create a single segment with the full text
        if not segments:
            segments.append(Segment(
                id=0,
                start=0.0,
                end=getattr(response, 'duration', 0.0),
                text=text,
                confidence=1.0
            ))
        
        # Calculate overall confidence and duration
        confidence = sum(s.confidence for s in segments) / len(segments) if segments else 1.0
        duration = segments[-1].end if segments else getattr(response, 'duration', 0.0)
        
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=language,
            confidence=confidence,
            duration=duration,
            metadata={
                "model": "whisper-1",
                "api_based": True,
                "word_timestamps": self.options.word_timestamps,
            }
        )
    
    # Note: We're not implementing chunking for the API version since the API handles long files
    
    # These methods are not needed for the API-based implementation
