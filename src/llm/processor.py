"""
LLM processor module for enhancing transcriptions.
"""
import os
from typing import Optional, Dict, Any, List
from loguru import logger
import openai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from ..observability.telemetry import telemetry, observe_function

# Load environment variables
load_dotenv()


class ProcessingOptions(BaseModel):
    """
    Options for controlling the LLM processing.
    """
    enhance_formatting: bool = Field(
        default=True, 
        description="Improve formatting and structure"
    )
    fix_punctuation: bool = Field(
        default=True, 
        description="Fix punctuation and capitalization"
    )
    summarize: bool = Field(
        default=False, 
        description="Generate a summary of the content"
    )
    model: str = Field(
        default="gpt-4o", 
        description="LLM model to use"
    )
    temperature: float = Field(
        default=0.3, 
        description="Temperature for generation (0-1)"
    )


class EnhancedText(BaseModel):
    """
    Result of LLM processing on transcription.
    """
    original_text: str = Field(..., description="Original transcription text")
    enhanced_text: str = Field(..., description="Enhanced transcription text")
    summary: Optional[str] = Field(None, description="Summary of the content if requested")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LLMProcessor:
    """
    Handles enhancement of transcriptions using an LLM.
    
    This class provides functionality to process raw transcriptions using an LLM
    to improve formatting, fix punctuation, and optionally generate summaries.
    """
    
    def __init__(self, options: Optional[ProcessingOptions] = None):
        """
        Initialize the LLMProcessor.
        
        Args:
            options (Optional[ProcessingOptions]): Options for controlling the processing.
                Defaults to None, which uses default settings.
        """
        self.options = options or ProcessingOptions()
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found in environment variables")
        
        logger.info(f"Initializing LLMProcessor with options: {self.options}")
    
    @observe_function(name="process_transcription")
    def process_transcription(self, text: str, trace_id: Optional[str] = None) -> EnhancedText:
        """
        Process a transcription using an LLM.
        
        Args:
            text (str): The raw transcription text to process.
            trace_id (Optional[str]): Trace ID for telemetry tracking.
                
        Returns:
            EnhancedText: The enhanced transcription.
            
        Raises:
            ValueError: If the text is empty.
            Exception: If there's an error with the LLM API.
        """
        if not text.strip():
            raise ValueError("Transcription text is empty")
        
        logger.info(f"Processing transcription with LLM (length: {len(text)} chars)")
        
        try:
            # Build the prompt based on the options
            prompt = self._build_prompt(text)
            
            # Call the LLM API with telemetry tracking
            response = self._call_llm_api(prompt, trace_id)
            
            # Parse the response
            result = self._parse_response(text, response)
            
            logger.success("LLM processing completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error processing transcription with LLM: {str(e)}")
            raise
    
    def _build_prompt(self, text: str) -> List[Dict[str, str]]:
        """
        Build the prompt for the LLM.
        
        Args:
            text (str): The raw transcription text.
                
        Returns:
            List[Dict[str, str]]: The formatted prompt messages.
        """
        system_prompt = "You are an expert transcription editor. Your task is to improve the quality of a raw speech-to-text transcription."
        
        task_description = "Please process the following raw transcription:"
        
        if self.options.enhance_formatting:
            task_description += "\n- Improve the formatting and structure"
        
        if self.options.fix_punctuation:
            task_description += "\n- Fix punctuation, capitalization, and grammar"
        
        if self.options.summarize:
            task_description += "\n- Provide a concise summary of the content"
        
        task_description += "\n\nKeep the meaning and content exactly the same. Do not add or remove information."
        
        if self.options.summarize:
            task_description += "\n\nFormat your response as follows:\n\nENHANCED TRANSCRIPTION:\n[enhanced text]\n\nSUMMARY:\n[summary]"
        else:
            task_description += "\n\nFormat your response as follows:\n\nENHANCED TRANSCRIPTION:\n[enhanced text]"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{task_description}\n\nRAW TRANSCRIPTION:\n{text}"}
        ]
        
        return messages
    
    def _call_llm_api(self, messages: List[Dict[str, str]], trace_id: Optional[str] = None) -> str:
        """
        Call the LLM API.
        
        Args:
            messages (List[Dict[str, str]]): The formatted prompt messages.
            trace_id (Optional[str]): Trace ID for telemetry tracking.
                
        Returns:
            str: The LLM response.
        """
        # Prepare for telemetry tracking
        prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        start_time = __import__("time").time()
        
        # Call the LLM API
        response = openai.chat.completions.create(
            model=self.options.model,
            messages=messages,
            temperature=self.options.temperature,
        )
        
        completion_text = response.choices[0].message.content
        elapsed_time = __import__("time").time() - start_time
        
        # Track the LLM call with Langfuse if trace_id is provided
        if trace_id:
            # Calculate token usage and costs (approximate)
            input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else None
            output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else None
            
            # Estimate costs (very rough approximation)
            # These rates should be adjusted based on the actual model pricing
            input_cost = None
            output_cost = None
            total_cost = None
            
            if input_tokens and output_tokens:
                if "gpt-4" in self.options.model:
                    input_cost = input_tokens * 0.00003  # $0.03 per 1K tokens
                    output_cost = output_tokens * 0.00006  # $0.06 per 1K tokens
                else:  # Assume gpt-3.5-turbo
                    input_cost = input_tokens * 0.000001  # $0.001 per 1K tokens
                    output_cost = output_tokens * 0.000002  # $0.002 per 1K tokens
                total_cost = input_cost + output_cost
            
            # Track the LLM call
            telemetry.track_llm(
                trace_id=trace_id,
                name="enhance_transcription",
                model=self.options.model,
                prompt=prompt_text,
                completion=completion_text,
                metadata={
                    "temperature": self.options.temperature,
                    "enhance_formatting": self.options.enhance_formatting,
                    "fix_punctuation": self.options.fix_punctuation,
                    "summarize": self.options.summarize,
                    "elapsed_time": elapsed_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost
            )
        
        return completion_text
    
    def _parse_response(self, original_text: str, response: str) -> EnhancedText:
        """
        Parse the LLM response.
        
        Args:
            original_text (str): The original transcription text.
            response (str): The LLM response.
                
        Returns:
            EnhancedText: The parsed enhanced transcription.
        """
        # Extract enhanced text
        enhanced_text = ""
        summary = None
        
        if "ENHANCED TRANSCRIPTION:" in response:
            parts = response.split("ENHANCED TRANSCRIPTION:")
            if len(parts) > 1:
                enhanced_text = parts[1].strip()
                
                # If there's a summary section, extract it
                if "SUMMARY:" in enhanced_text:
                    enhanced_parts = enhanced_text.split("SUMMARY:")
                    enhanced_text = enhanced_parts[0].strip()
                    summary = enhanced_parts[1].strip() if len(enhanced_parts) > 1 else None
        else:
            # Fallback if the format is not as expected
            enhanced_text = response.strip()
        
        return EnhancedText(
            original_text=original_text,
            enhanced_text=enhanced_text,
            summary=summary,
            metadata={
                "model": self.options.model,
                "temperature": self.options.temperature,
                "enhance_formatting": self.options.enhance_formatting,
                "fix_punctuation": self.options.fix_punctuation,
                "summarize": self.options.summarize,
            }
        )
