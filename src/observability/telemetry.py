"""
Telemetry module for observability using Langfuse.
"""
import os
import time
from typing import Optional, Dict, Any, Union
from pathlib import Path
import uuid
from loguru import logger
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.decorators import observe

# Load environment variables
load_dotenv()


class Telemetry:
    """
    Handles telemetry and observability for the voice-to-text agent using Langfuse.
    
    This class provides functionality to track and monitor the performance and usage
    of the voice-to-text agent, including processing time, API calls, and errors.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Telemetry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the Telemetry instance."""
        if self._initialized:
            return
            
        # Get Langfuse API keys from environment variables
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        # Initialize Langfuse client if API keys are available
        self.langfuse = None
        self._disabled = False
        
        if self.public_key and self.secret_key:
            try:
                self.langfuse = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host
                )
                logger.info("Langfuse telemetry initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {str(e)}")
                self._disabled = True
        else:
            logger.warning("Langfuse API keys not found, telemetry disabled")
            self._disabled = True
        
        self._initialized = True
        
    def disable_telemetry(self):
        """Disable telemetry to prevent further errors."""
        if not self._disabled:
            logger.warning("Disabling telemetry due to errors")
            self._disabled = True
            self.langfuse = None
    
    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new trace for tracking a complete process.
        
        Args:
            name (str): Name of the trace.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the trace.
                
        Returns:
            str: Trace ID.
        """
        trace_id = str(uuid.uuid4())
        
        if self._disabled:
            return trace_id
            
        if self.langfuse:
            try:
                # Create trace with error handling for API differences
                trace = self.langfuse.trace(
                    name=name,
                    id=trace_id,
                    metadata=metadata or {}
                )
                logger.debug(f"Started trace: {trace_id}")
            except Exception as e:
                logger.warning(f"Failed to start trace: {str(e)}")
                self.disable_telemetry()
        
        return trace_id
    
    def start_span(
        self, 
        trace_id: str, 
        name: str, 
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new span within a trace for tracking a specific operation.
        
        Args:
            trace_id (str): ID of the parent trace.
            name (str): Name of the span.
            parent_id (Optional[str]): ID of the parent span if this is a child span.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the span.
                
        Returns:
            str: Span ID.
        """
        span_id = str(uuid.uuid4())
        
        if self._disabled:
            return span_id
            
        if self.langfuse:
            try:
                # Try to create a span using the trace object first (new API)
                try:
                    trace = self.langfuse.get_trace(trace_id)
                    if trace:
                        span = trace.span(
                            name=name,
                            id=span_id,
                            parent_id=parent_id,
                            metadata=metadata or {}
                        )
                        logger.debug(f"Started span using trace object: {span_id}")
                        return span_id
                except (AttributeError, Exception):
                    # Fall back to direct span creation (old API)
                    span = self.langfuse.span(
                        name=name,
                        id=span_id,
                        trace_id=trace_id,
                        parent_id=parent_id,
                        metadata=metadata or {}
                    )
                    logger.debug(f"Started span: {span_id}")
            except Exception as e:
                logger.warning(f"Failed to start span: {str(e)}")
                self.disable_telemetry()
        
        return span_id
    
    def end_span(self, span_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        End a span and record its completion.
        
        Args:
            span_id (str): ID of the span to end.
            metadata (Optional[Dict[str, Any]]): Additional metadata to add upon completion.
        """
        if self._disabled:
            return
            
        if self.langfuse:
            try:
                # Check which API version is available
                if hasattr(self.langfuse, 'update_span'):
                    # Old API
                    self.langfuse.update_span(
                        id=span_id,
                        metadata=metadata or {},
                        end=True
                    )
                elif hasattr(self.langfuse, 'span'):
                    # New API
                    span = self.langfuse.span(span_id)
                    if metadata:
                        span.update(metadata=metadata)
                    span.end()
                else:
                    # Fallback - just log that we can't end the span
                    logger.warning(f"Cannot end span {span_id}: Langfuse API methods not found")
                    self.disable_telemetry()
                    return
                    
                logger.debug(f"Ended span: {span_id}")
            except Exception as e:
                logger.warning(f"Failed to end span: {str(e)}")
                self.disable_telemetry()
    
    def track_llm(
        self,
        trace_id: str,
        name: str,
        model: str,
        prompt: str,
        completion: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_cost: Optional[float] = None,
        output_cost: Optional[float] = None,
        total_cost: Optional[float] = None
    ) -> str:
        """
        Track an LLM API call.
        
        Args:
            trace_id (str): ID of the parent trace.
            name (str): Name of the LLM operation.
            model (str): Name of the LLM model used.
            prompt (str): The prompt sent to the LLM.
            completion (str): The response from the LLM.
            parent_id (Optional[str]): ID of the parent span if this is a child operation.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the operation.
            input_cost (Optional[float]): Cost of the input tokens.
            output_cost (Optional[float]): Cost of the output tokens.
            total_cost (Optional[float]): Total cost of the operation.
                
        Returns:
            str: Generation ID.
        """
        generation_id = str(uuid.uuid4())
        
        if self._disabled:
            return generation_id
            
        if self.langfuse:
            try:
                # Try to create a generation using the trace object first (new API)
                try:
                    trace = self.langfuse.get_trace(trace_id)
                    if trace:
                        generation = trace.generation(
                            name=name,
                            id=generation_id,
                            parent_id=parent_id,
                            model=model,
                            prompt=prompt,
                            completion=completion,
                            metadata=metadata or {},
                            input_cost=input_cost,
                            output_cost=output_cost,
                            total_cost=total_cost
                        )
                        logger.debug(f"Tracked LLM call using trace object: {generation_id}")
                        return generation_id
                except (AttributeError, Exception):
                    # Fall back to direct generation creation (old API)
                    generation = self.langfuse.generation(
                        name=name,
                        id=generation_id,
                        trace_id=trace_id,
                        parent_id=parent_id,
                        model=model,
                        prompt=prompt,
                        completion=completion,
                        metadata=metadata or {},
                        input_cost=input_cost,
                        output_cost=output_cost,
                        total_cost=total_cost
                    )
                    logger.debug(f"Tracked LLM call: {generation_id}")
            except Exception as e:
                logger.warning(f"Failed to track LLM call: {str(e)}")
                self.disable_telemetry()
        
        return generation_id
    
    def track_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None
    ) -> None:
        """
        Track a score for evaluating the quality of a process.
        
        Args:
            trace_id (str): ID of the trace to score.
            name (str): Name of the score.
            value (float): Score value (typically between 0 and 1).
            comment (Optional[str]): Comment explaining the score.
        """
        if self._disabled:
            return
            
        if self.langfuse:
            try:
                # Try to create a score using the trace object first (new API)
                try:
                    trace = self.langfuse.get_trace(trace_id)
                    if trace:
                        score = trace.score(
                            name=name,
                            value=value,
                            comment=comment
                        )
                        logger.debug(f"Tracked score using trace object: {name}={value}")
                        return
                except (AttributeError, Exception):
                    # Fall back to direct score creation (old API)
                    score = self.langfuse.score(
                        trace_id=trace_id,
                        name=name,
                        value=value,
                        comment=comment
                    )
                    logger.debug(f"Tracked score: {name}={value}")
            except Exception as e:
                logger.warning(f"Failed to track score: {str(e)}")
                self.disable_telemetry()
    
    def track_event(
        self,
        trace_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track a discrete event within a trace.
        
        Args:
            trace_id (str): ID of the parent trace.
            name (str): Name of the event.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the event.
        """
        if self._disabled:
            return
            
        if self.langfuse:
            try:
                # Try to create an event using the trace object first (new API)
                try:
                    trace = self.langfuse.get_trace(trace_id)
                    if trace:
                        event = trace.event(
                            name=name,
                            metadata=metadata or {}
                        )
                        logger.debug(f"Tracked event using trace object: {name}")
                        return
                except (AttributeError, Exception):
                    # Fall back to direct event creation (old API)
                    event = self.langfuse.event(
                        trace_id=trace_id,
                        name=name,
                        metadata=metadata or {}
                    )
                    logger.debug(f"Tracked event: {name}")
            except Exception as e:
                logger.warning(f"Failed to track event: {str(e)}")
                self.disable_telemetry()
    
    def flush(self) -> None:
        """
        Flush any pending telemetry data to ensure it's sent to Langfuse.
        """
        if self._disabled:
            return
            
        if self.langfuse:
            try:
                self.langfuse.flush()
                logger.debug("Flushed telemetry data")
            except Exception as e:
                logger.warning(f"Failed to flush telemetry data: {str(e)}")
                self.disable_telemetry()


# Create a singleton instance
telemetry = Telemetry()

# ... (rest of the code remains the same)
# Decorator for observing functions
def observe_function(name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for observing functions with Langfuse.
    
    Args:
        name (Optional[str]): Name of the operation. Defaults to the function name.
        metadata (Optional[Dict[str, Any]]): Additional metadata for the operation.
            
    Returns:
        Callable: Decorated function.
    """
    def decorator(func):
        if telemetry.langfuse:
            # Use Langfuse's built-in decorator if available
            # Note: Current Langfuse API doesn't accept metadata in observe()
            decorated_func = observe(name=name or func.__name__)(func)
            
            # If metadata was provided, we'll need to handle it manually
            if metadata:
                def wrapper(*args, **kwargs):
                    # Start a span with metadata if a trace_id is available in kwargs
                    trace_id = kwargs.get('trace_id', None)
                    if trace_id:
                        span_id = telemetry.start_span(trace_id, name or func.__name__, metadata=metadata)
                        try:
                            return decorated_func(*args, **kwargs)
                        finally:
                            telemetry.end_span(span_id)
                    else:
                        return decorated_func(*args, **kwargs)
                return wrapper
            return decorated_func
        else:
            # Fallback implementation if Langfuse is not available
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_time = time.time() - start_time
                    logger.debug(f"Function {name or func.__name__} executed in {elapsed_time:.2f}s")
            return wrapper
    return decorator
