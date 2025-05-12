"""
Helper functions for the voice-to-text agent.
"""
import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
import json
from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file (Optional[str]): Path to log file. If None, logs to console only.
    """
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    
    # Add file logger if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="1 week",
        )
    
    logger.info(f"Logging initialized with level: {log_level}")


def save_result_to_file(result: Dict[str, Any], output_path: Union[str, Path], format: str = "text"):
    """
    Save processing result to a file.
    
    Args:
        result (Dict[str, Any]): The processing result.
        output_path (Union[str, Path]): Path to save the result.
        format (str): Output format (text, json).
    
    Returns:
        Path: Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "json":
        # Save as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        # Save as text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
            
            if result.get("summary"):
                f.write("\n\n--- SUMMARY ---\n\n")
                f.write(result["summary"])
    
    logger.info(f"Result saved to: {output_path}")
    return output_path


def validate_file_path(file_path: Union[str, Path]) -> Path:
    """
    Validate that a file path exists and has the correct extension.
    
    Args:
        file_path (Union[str, Path]): Path to validate.
    
    Returns:
        Path: Validated Path object.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has an unsupported extension.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() != ".m4a":
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Only .m4a files are supported.")
    
    return file_path


def get_default_output_path(input_path: Path, format: str = "text") -> Path:
    """
    Generate a default output path based on the input path.
    
    Args:
        input_path (Path): Input file path.
        format (str): Output format (text, json).
    
    Returns:
        Path: Default output path.
    """
    extension = ".json" if format.lower() == "json" else ".txt"
    return input_path.with_suffix(extension)
