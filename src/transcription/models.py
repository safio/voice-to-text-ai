"""
Pydantic models for transcription results and options.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Segment(BaseModel):
    """
    Represents a time-aligned segment of transcription.
    """
    id: int = Field(..., description="Segment identifier")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    confidence: float = Field(default=1.0, description="Confidence score (0-1)")


class TranscriptionOptions(BaseModel):
    """
    Options for controlling the transcription process.
    """
    language: Optional[str] = Field(
        default=None, 
        description="Language code (e.g., 'en', 'fr'). None for auto-detection."
    )
    model_size: str = Field(
        default="base", 
        description="Model size to use (tiny, base, small, medium, large)"
    )
    word_timestamps: bool = Field(
        default=False, 
        description="Whether to include word-level timestamps"
    )
    chunk_size: int = Field(
        default=30, 
        description="Size of audio chunks in seconds for processing long files"
    )
    device: str = Field(
        default="cpu", 
        description="Device to use for inference (cpu, cuda)"
    )


class TranscriptionResult(BaseModel):
    """
    Result of a transcription operation.
    """
    text: str = Field(..., description="Full transcribed text")
    segments: List[Segment] = Field(default_factory=list, description="Time-aligned segments")
    language: str = Field(..., description="Detected or specified language")
    confidence: float = Field(..., description="Overall confidence score (0-1)")
    duration: float = Field(..., description="Duration of the audio in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
