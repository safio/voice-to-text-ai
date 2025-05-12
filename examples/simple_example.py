"""
Simple example of using the voice-to-text agent.

This script demonstrates how to use the VoiceToTextAgent to convert an M4A file to text.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the src package
sys.path.append(str(Path(__file__).parent.parent))

from src.agent.agent import VoiceToTextAgent, AgentOptions
from src.transcription.models import TranscriptionOptions
from src.llm.processor import ProcessingOptions
from src.utils.helpers import setup_logging

# Load environment variables
load_dotenv()

# Set up logging
setup_logging(log_level="INFO")


def process_audio_file(file_path, output_path=None, summarize=False):
    """
    Process an audio file using the voice-to-text agent.
    
    Args:
        file_path (str): Path to the M4A file to process.
        output_path (str, optional): Path to save the output text. Defaults to None.
        summarize (bool, optional): Whether to generate a summary. Defaults to False.
    
    Returns:
        dict: The processing result.
    """
    # Set up options
    transcription_options = TranscriptionOptions(
        model_size="base",
        language=None,  # Auto-detect language
    )
    
    processing_options = ProcessingOptions(
        enhance_formatting=True,
        fix_punctuation=True,
        summarize=summarize,
        model="gpt-4o",
        temperature=0.3,
    )
    
    agent_options = AgentOptions(
        output_format="text",
        save_intermediate_files=False,
        transcription_options=transcription_options,
        processing_options=processing_options,
    )
    
    # Initialize agent
    agent = VoiceToTextAgent(options=agent_options)
    
    # Process file
    print(f"Processing file: {file_path}")
    result = agent.process_file(file_path)
    
    # Print result
    print("\n--- TRANSCRIPTION RESULT ---\n")
    print(result.text)
    
    if summarize and result.summary:
        print("\n--- SUMMARY ---\n")
        print(result.summary)
    
    print(f"\nProcessing completed in {result.duration:.2f} seconds")
    
    # Save to file if output path is provided
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.text)
            if summarize and result.summary:
                f.write("\n\n--- SUMMARY ---\n\n")
                f.write(result.summary)
        print(f"Output saved to: {output_path}")
    
    return {
        "text": result.text,
        "summary": result.summary,
        "duration": result.duration,
        "metadata": result.metadata,
    }


if __name__ == "__main__":
    # Check if a file path is provided
    if len(sys.argv) < 2:
        print("Usage: python simple_example.py <path_to_m4a_file> [output_path]")
        sys.exit(1)
    
    # Get file path
    file_path = sys.argv[1]
    
    # Get output path if provided
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Process the file
    process_audio_file(file_path, output_path, summarize=True)
