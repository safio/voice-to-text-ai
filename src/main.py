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
@click.option("--mode", type=click.Choice(["llm", "local"]), default="local", help="Transcription mode (llm uses OpenAI API, local uses Whisper locally)")
@click.option("--model-size", type=click.Choice(["tiny", "base", "small", "medium", "large"]), default="base", help="Whisper model size")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu", help="Device to use for local transcription (cpu or cuda)")
@click.option("--language", help="Language code (e.g., 'en', 'fr'). None for auto-detection")
@click.option("--use-llm/--no-llm", default=False, help="Whether to use LLM for post-processing (default: no)")
@click.option("--enhance-formatting/--no-enhance-formatting", default=True, help="Improve formatting and structure (only with --use-llm)")
@click.option("--fix-punctuation/--no-fix-punctuation", default=True, help="Fix punctuation and capitalization (only with --use-llm)")
@click.option("--summarize", is_flag=True, help="Generate a summary of the content (only with --use-llm)")
@click.option("--llm-model", default="gpt-4o", help="LLM model to use (only with --use-llm)")
@click.option("--temperature", type=float, default=0.3, help="Temperature for LLM generation (0-1) (only with --use-llm)")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]), default="INFO", help="Logging level")
@click.option("--log-file", type=click.Path(), help="Log file path")
def main(
    input_file: str,
    output: Optional[str] = None,
    format: str = "text",
    save_intermediate: bool = False,
    mode: str = "local",
    model_size: str = "base",
    device: str = "cpu",
    language: Optional[str] = None,
    use_llm: bool = False,
    enhance_formatting: bool = True,
    fix_punctuation: bool = True,
    summarize: bool = False,
    llm_model: str = "gpt-4o",
    temperature: float = 0.3,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
):
    """
    Convert an M4A voice recording to text with optional LLM processing.
    
    INPUT_FILE: Path to the M4A file to process.
    
    By default, uses local Whisper transcription without LLM post-processing.
    Use --mode=llm to use OpenAI's API for transcription.
    Use --use-llm to enable LLM post-processing for enhanced formatting.
    """
    # Set up logging
    setup_logging(log_level, log_file)
    
    try:
        # Validate input file
        input_path = validate_file_path(input_file)
        
        # Determine output path if not provided
        output_path = Path(output) if output else get_default_output_path(input_path, format)
        
        # Check for API key if using LLM mode or LLM processing
        if mode == "llm" or use_llm:
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY not found in environment variables")
                if mode == "llm":
                    logger.error("API key is required for LLM transcription mode")
                    click.echo("Error: OPENAI_API_KEY is required for LLM transcription mode", err=True)
                    click.echo("Please set the OPENAI_API_KEY environment variable or use --mode=local", err=True)
                    return 1
                if use_llm:
                    logger.warning("API key is required for LLM post-processing, disabling it")
                    click.echo("Warning: OPENAI_API_KEY not found, disabling LLM post-processing", err=True)
                    use_llm = False
        
        # Set up options
        transcription_options = TranscriptionOptions(
            model_size=model_size,
            language=language,
            device=device,
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
            transcription_mode=mode,
            transcription_options=transcription_options,
            processing_options=processing_options,
            skip_llm_processing=not use_llm,
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
        click.echo(f"Transcription mode: {mode}")
        click.echo(f"LLM post-processing: {'enabled' if use_llm else 'disabled'}")
        click.echo(f"Output saved to: {output_path}")
        
        if use_llm and summarize and result.summary:
            click.echo("\nSummary:")
            click.echo(result.summary)
        
        return 0
    except ImportError as e:
        if "whisper" in str(e).lower() or "numba" in str(e).lower() or "numpy" in str(e).lower():
            logger.error(f"Dependency error: {str(e)}")
            click.echo("Error: Missing or incompatible dependencies for local transcription", err=True)
            click.echo("Please install required packages:", err=True)
            click.echo("  pip install openai-whisper torch numpy<2.0.0 numba", err=True)
            return 1
        else:
            logger.error(f"Import error: {str(e)}")
            click.echo(f"Error: {str(e)}", err=True)
            return 1
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        click.echo(f"Error: {str(e)}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
