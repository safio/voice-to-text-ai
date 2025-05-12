"""
Main CLI interface for the voice-to-text agent.
"""
import os
import sys
from pathlib import Path
from typing import Optional
import click
from loguru import logger
from dotenv import load_dotenv

from .agent.agent import VoiceToTextAgent, AgentOptions
from .transcription.models import TranscriptionOptions
from .llm.processor import ProcessingOptions
from .utils.helpers import setup_logging, save_result_to_file, validate_file_path, get_default_output_path

# Load environment variables
load_dotenv()


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.option("--save-intermediate", is_flag=True, help="Save intermediate files")
@click.option("--model-size", type=click.Choice(["tiny", "base", "small", "medium", "large"]), default="base", help="Whisper model size")
@click.option("--language", help="Language code (e.g., 'en', 'fr'). None for auto-detection")
@click.option("--enhance-formatting/--no-enhance-formatting", default=True, help="Improve formatting and structure")
@click.option("--fix-punctuation/--no-fix-punctuation", default=True, help="Fix punctuation and capitalization")
@click.option("--summarize", is_flag=True, help="Generate a summary of the content")
@click.option("--llm-model", default="gpt-4o", help="LLM model to use")
@click.option("--temperature", type=float, default=0.3, help="Temperature for LLM generation (0-1)")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]), default="INFO", help="Logging level")
@click.option("--log-file", type=click.Path(), help="Log file path")
def main(
    input_file: str,
    output: Optional[str] = None,
    format: str = "text",
    save_intermediate: bool = False,
    model_size: str = "base",
    language: Optional[str] = None,
    enhance_formatting: bool = True,
    fix_punctuation: bool = True,
    summarize: bool = False,
    llm_model: str = "gpt-4o",
    temperature: float = 0.3,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
):
    """
    Convert an M4A voice recording to text using LLM processing.
    
    INPUT_FILE: Path to the M4A file to process.
    """
    # Set up logging
    setup_logging(log_level, log_file)
    
    try:
        # Validate input file
        input_path = validate_file_path(input_file)
        
        # Determine output path if not provided
        output_path = Path(output) if output else get_default_output_path(input_path, format)
        
        # Set up options
        transcription_options = TranscriptionOptions(
            model_size=model_size,
            language=language,
        )
        
        processing_options = ProcessingOptions(
            enhance_formatting=enhance_formatting,
            fix_punctuation=fix_punctuation,
            summarize=summarize,
            model=llm_model,
            temperature=temperature,
        )
        
        agent_options = AgentOptions(
            output_format=format,
            save_intermediate_files=save_intermediate,
            transcription_options=transcription_options,
            processing_options=processing_options,
        )
        
        # Initialize agent
        agent = VoiceToTextAgent(options=agent_options)
        
        # Process file
        logger.info(f"Processing file: {input_path}")
        result = agent.process_file(input_path)
        
        # Save result
        save_result_to_file(
            {
                "text": result.text,
                "summary": result.summary,
                "duration": result.duration,
                "metadata": result.metadata,
            },
            output_path,
            format,
        )
        
        # Print summary to console
        click.echo(f"Processing completed in {result.duration:.2f} seconds")
        click.echo(f"Output saved to: {output_path}")
        
        if summarize and result.summary:
            click.echo("\nSummary:")
            click.echo(result.summary)
        
        return 0
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        click.echo(f"Error: {str(e)}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
