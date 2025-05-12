"""
Agent module for orchestrating the voice-to-text process.
"""
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
from loguru import logger
from pydantic import BaseModel, Field

from ..observability.telemetry import telemetry, observe_function

from ..audio.converter import AudioConverter
from ..audio.processor import AudioProcessor
from ..transcription.transcriber import Transcriber
from ..transcription.models import TranscriptionOptions
from ..llm.processor import LLMProcessor, ProcessingOptions


class AgentOptions(BaseModel):
    """
    Options for controlling the agent behavior.
    """
    output_format: str = Field(
        default="text", 
        description="Output format (text, json)"
    )
    save_intermediate_files: bool = Field(
        default=False, 
        description="Whether to save intermediate files"
    )
    transcription_options: Optional[TranscriptionOptions] = Field(
        default=None, 
        description="Options for the transcription process"
    )
    processing_options: Optional[ProcessingOptions] = Field(
        default=None, 
        description="Options for the LLM processing"
    )


class ProcessResult(BaseModel):
    """
    Result of the voice-to-text processing.
    """
    text: str = Field(..., description="Final processed text")
    summary: Optional[str] = Field(None, description="Summary if requested")
    duration: float = Field(..., description="Processing duration in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class VoiceToTextAgent:
    """
    Autonomous agent for converting voice to text.
    
    This agent orchestrates the entire process of converting voice recordings
    to text, including audio processing, transcription, and LLM enhancement.
    """
    
    def __init__(self, options: Optional[AgentOptions] = None):
        """
        Initialize the VoiceToTextAgent.
        
        Args:
            options (Optional[AgentOptions]): Options for controlling the agent behavior.
                Defaults to None, which uses default settings.
        """
        self.options = options or AgentOptions()
        
        # Initialize components
        self.audio_converter = AudioConverter()
        self.audio_processor = AudioProcessor()
        self.transcriber = Transcriber(options=self.options.transcription_options)
        self.llm_processor = LLMProcessor(options=self.options.processing_options)
        
        logger.info(f"Initializing VoiceToTextAgent with options: {self.options}")
    
    @observe_function(name="process_file")
    def process_file(self, file_path: Union[str, Path], options: Optional[Dict[str, Any]] = None) -> ProcessResult:
        """
        Process a voice recording file to text.
        
        Args:
            file_path (Union[str, Path]): Path to the voice recording file.
            options (Optional[Dict[str, Any]]): Additional options to override the default ones.
                
        Returns:
            ProcessResult: The processing result.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not supported.
        """
        file_path = Path(file_path)
        
        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() != '.m4a':
            raise ValueError(f"Only M4A files are supported, got: {file_path.suffix}")
        
        # Update options if provided
        if options:
            for key, value in options.items():
                if hasattr(self.options, key):
                    setattr(self.options, key, value)
        
        # Start a trace for the entire process
        trace_id = telemetry.start_trace(
            name="voice_to_text_process",
            metadata={
                "file_path": str(file_path),
                "file_size": os.path.getsize(file_path),
                "options": self.options.model_dump(),
            }
        )
        
        logger.info(f"Processing file: {file_path}")
        start_time = time.time()
        
        try:
            # Step 1: Convert M4A to WAV
            logger.info("Step 1: Converting M4A to WAV")
            conversion_span_id = telemetry.start_span(trace_id, "convert_m4a_to_wav")
            wav_path = self._convert_to_wav(file_path)
            telemetry.end_span(conversion_span_id, {"output_path": str(wav_path)})
            
            # Step 2: Preprocess audio
            logger.info("Step 2: Preprocessing audio")
            preprocessing_span_id = telemetry.start_span(trace_id, "preprocess_audio")
            processed_audio_path = self._preprocess_audio(wav_path)
            telemetry.end_span(preprocessing_span_id, {"output_path": str(processed_audio_path)})
            
            # Step 3: Transcribe audio
            logger.info("Step 3: Transcribing audio")
            transcription_span_id = telemetry.start_span(trace_id, "transcribe_audio")
            transcription_result = self._transcribe_audio(processed_audio_path)
            telemetry.end_span(transcription_span_id, {
                "text_length": len(transcription_result.text),
                "language": transcription_result.language,
                "confidence": transcription_result.confidence,
                "duration": transcription_result.duration,
            })
            
            # Step 4: Process with LLM
            logger.info("Step 4: Processing with LLM")
            llm_span_id = telemetry.start_span(trace_id, "process_with_llm")
            enhanced_result = self._process_with_llm(transcription_result.text, trace_id=trace_id)
            
            telemetry.end_span(llm_span_id, {
                "enhanced_text_length": len(enhanced_result.enhanced_text),
                "has_summary": enhanced_result.summary is not None,
                "summary_length": len(enhanced_result.summary) if enhanced_result.summary else 0,
            })
            
            # Clean up intermediate files if not saving them
            if not self.options.save_intermediate_files:
                cleanup_span_id = telemetry.start_span(trace_id, "cleanup_files")
                self._cleanup_files([wav_path, processed_audio_path])
                telemetry.end_span(cleanup_span_id)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Create result
            result = ProcessResult(
                text=enhanced_result.enhanced_text,
                summary=enhanced_result.summary,
                duration=duration,
                metadata={
                    "original_file": str(file_path),
                    "transcription_confidence": transcription_result.confidence,
                    "language": transcription_result.language,
                    "audio_duration": transcription_result.duration,
                    "trace_id": trace_id,  # Include trace ID in metadata for reference
                }
            )
            
            # Track success event and score
            telemetry.track_event(trace_id, "process_completed", {
                "duration": duration,
                "text_length": len(result.text),
                "has_summary": result.summary is not None,
            })
            
            # Score based on confidence
            telemetry.track_score(
                trace_id=trace_id,
                name="transcription_confidence",
                value=transcription_result.confidence,
                comment=f"Transcription confidence for {file_path.name}"
            )
            
            # Ensure all telemetry is sent
            telemetry.flush()
            
            logger.success(f"Processing completed in {duration:.2f} seconds")
            return result
        except Exception as e:
            # Track error event
            telemetry.track_event(trace_id, "process_error", {
                "error": str(e),
                "error_type": type(e).__name__,
            })
            telemetry.flush()
            
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    @observe_function(name="convert_to_wav")
    def _convert_to_wav(self, file_path: Path) -> Path:
        """
        Convert an M4A file to WAV.
        
        Args:
            file_path (Path): Path to the M4A file.
                
        Returns:
            Path: Path to the converted WAV file.
        """
        output_dir = file_path.parent / "temp" if self.options.save_intermediate_files else None
        
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{file_path.stem}.wav" if output_dir else None
        
        wav_path = self.audio_converter.convert_m4a_to_wav(file_path, output_path)
        return Path(wav_path)
    
    @observe_function(name="preprocess_audio")
    def _preprocess_audio(self, file_path: Path) -> Path:
        """
        Preprocess an audio file.
        
        Args:
            file_path (Path): Path to the audio file.
                
        Returns:
            Path: Path to the preprocessed audio file.
        """
        output_dir = file_path.parent / "temp" if self.options.save_intermediate_files else None
        
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{file_path.stem}_processed{file_path.suffix}" if output_dir else None
        
        processed_path = self.audio_processor.preprocess_audio(file_path, output_path)
        return Path(processed_path)
    
    @observe_function(name="transcribe_audio")
    def _transcribe_audio(self, file_path: Path):
        """
        Transcribe an audio file.
        
        Args:
            file_path (Path): Path to the audio file.
                
        Returns:
            TranscriptionResult: The transcription result.
        """
        return self.transcriber.transcribe(file_path)
    
    @observe_function(name="process_with_llm")
    def _process_with_llm(self, text: str, trace_id: Optional[str] = None):
        """
        Process a transcription with an LLM.
        
        Args:
            text (str): The transcription text.
            trace_id (Optional[str]): Trace ID for telemetry tracking.
                
        Returns:
            EnhancedText: The enhanced text.
        """
        return self.llm_processor.process_transcription(text, trace_id=trace_id)
    
    @observe_function(name="cleanup_files")
    def _cleanup_files(self, file_paths: list):
        """
        Clean up intermediate files.
        
        Args:
            file_paths (list): List of file paths to clean up.
        """
        for file_path in file_paths:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Deleted intermediate file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {str(e)}")
