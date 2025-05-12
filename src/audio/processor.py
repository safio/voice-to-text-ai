"""
Audio processor module for preprocessing audio files before transcription.
"""
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
from loguru import logger
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import split_on_silence


class AudioProcessor:
    """
    Handles preprocessing of audio files to improve transcription quality.
    
    This class provides functionality to preprocess audio files by applying
    various techniques such as noise reduction, normalization, and silence removal.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AudioProcessor.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration options for audio processing.
                Defaults to None, which uses default settings.
        """
        self.config = config or {
            "normalize_audio": True,
            "remove_silence": True,
            "silence_threshold": -40,  # dB
            "min_silence_len": 500,    # ms
        }
        logger.info("Initializing AudioProcessor with config: {}", self.config)
    
    def preprocess_audio(
        self, 
        input_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Preprocess an audio file to improve transcription quality.
        
        Args:
            input_path (Union[str, Path]): Path to the input audio file.
            output_path (Optional[Union[str, Path]]): Path to save the processed audio file.
                If not provided, will use the same name with '_processed' suffix.
                
        Returns:
            str: Path to the processed audio file.
            
        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        input_path = Path(input_path)
        
        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Determine output path if not provided
        if output_path is None:
            output_path = input_path.with_stem(f"{input_path.stem}_processed")
        else:
            output_path = Path(output_path)
        
        logger.info(f"Preprocessing {input_path} to {output_path}")
        
        try:
            # Load the audio file
            audio = AudioSegment.from_file(input_path)
            
            # Apply preprocessing steps
            processed_audio = self._apply_preprocessing(audio)
            
            # Export processed audio
            processed_audio.export(output_path, format=output_path.suffix.lstrip('.'))
            
            logger.success(f"Successfully preprocessed {input_path} to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error preprocessing {input_path}: {str(e)}")
            raise
    
    def _apply_preprocessing(self, audio: AudioSegment) -> AudioSegment:
        """
        Apply preprocessing steps to the audio.
        
        Args:
            audio (AudioSegment): The audio segment to process.
            
        Returns:
            AudioSegment: The processed audio segment.
        """
        # Apply normalization if enabled
        if self.config["normalize_audio"]:
            logger.debug("Applying normalization")
            audio = normalize(audio)
        
        # Remove silence if enabled
        if self.config["remove_silence"]:
            logger.debug("Removing silence")
            audio = self._remove_silence(audio)
        
        return audio
    
    def _remove_silence(self, audio: AudioSegment) -> AudioSegment:
        """
        Remove silence from the audio.
        
        Args:
            audio (AudioSegment): The audio segment to process.
            
        Returns:
            AudioSegment: The audio segment with silence removed.
        """
        # Split on silence
        chunks = split_on_silence(
            audio,
            min_silence_len=self.config["min_silence_len"],
            silence_thresh=self.config["silence_threshold"]
        )
        
        # Concatenate chunks with a short silence between them
        short_silence = AudioSegment.silent(duration=300)
        processed_audio = chunks[0]
        
        for chunk in chunks[1:]:
            processed_audio = processed_audio + short_silence + chunk
        
        return processed_audio
