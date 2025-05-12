"""
Audio converter module for handling M4A to WAV conversion.
"""
import os
from pathlib import Path
from typing import Optional, Union
from loguru import logger
from pydub import AudioSegment


class AudioConverter:
    """
    Handles conversion of audio files from M4A to WAV format.
    
    This class provides functionality to convert M4A audio files to WAV format,
    which is more widely supported by speech recognition libraries.
    """
    
    def __init__(self):
        """Initialize the AudioConverter."""
        logger.info("Initializing AudioConverter")
    
    def convert_m4a_to_wav(
        self, 
        input_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Convert an M4A file to WAV format.
        
        Args:
            input_path (Union[str, Path]): Path to the input M4A file.
            output_path (Optional[Union[str, Path]]): Path to save the output WAV file.
                If not provided, will use the same name as input with .wav extension.
                
        Returns:
            str: Path to the converted WAV file.
            
        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the input file is not an M4A file.
        """
        input_path = Path(input_path)
        
        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if input_path.suffix.lower() != '.m4a':
            raise ValueError(f"Input file must be an M4A file, got: {input_path.suffix}")
        
        # Determine output path if not provided
        if output_path is None:
            output_path = input_path.with_suffix('.wav')
        else:
            output_path = Path(output_path)
        
        logger.info(f"Converting {input_path} to {output_path}")
        
        try:
            # Load the M4A file
            audio = AudioSegment.from_file(input_path, format="m4a")
            
            # Export as WAV
            audio.export(output_path, format="wav")
            
            logger.success(f"Successfully converted {input_path} to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error converting {input_path} to WAV: {str(e)}")
            raise
