"""
Factory module for creating transcriber instances.
"""
from typing import Optional
from loguru import logger

from .base_transcriber import BaseTranscriber
from .transcriber import APITranscriber
from .local_transcriber import LocalTranscriber
from .models import TranscriptionOptions


def get_transcriber(mode: str = "llm", options: Optional[TranscriptionOptions] = None) -> BaseTranscriber:
    """
    Factory function to get the appropriate transcriber based on mode.
    
    Args:
        mode (str): The transcription mode. Options are "llm" (uses OpenAI API) or
                   "local" (uses Whisper locally). Default is "llm".
        options (Optional[TranscriptionOptions]): Options for controlling the transcription.
            Defaults to None, which uses default settings.
            
    Returns:
        BaseTranscriber: An instance of the appropriate transcriber.
        
    Raises:
        ValueError: If an unsupported transcription mode is specified.
    """
    logger.info(f"Creating transcriber with mode: {mode}")
    
    if mode == "llm":
        return APITranscriber(options)
    elif mode == "local":
        return LocalTranscriber(options)
    else:
        raise ValueError(f"Unsupported transcription mode: {mode}")
