"""
Test script to verify Langfuse connection.
"""
import os
import time
from dotenv import load_dotenv
from src.observability.telemetry import telemetry
from loguru import logger

def test_langfuse_connection():
    """Test if the connection to Langfuse is working properly."""
    # Make sure environment variables are loaded
    load_dotenv()
    
    # Check if Langfuse client is initialized
    if not telemetry.langfuse:
        print("❌ Langfuse client not initialized")
        print("Check your .env file for the following variables:")
        print("  - LANGFUSE_PUBLIC_KEY")
        print("  - LANGFUSE_SECRET_KEY")
        print("  - LANGFUSE_HOST (optional, defaults to https://cloud.langfuse.com)")
        return False
    
    print(f"✓ Langfuse client initialized with host: {telemetry.host}")
    
    # Try to create a test trace
    try:
        # Create a test trace
        trace_id = telemetry.start_trace("connection_test", {"test": True, "timestamp": time.time()})
        print(f"✓ Created test trace with ID: {trace_id}")
        
        # Create a test span
        span_id = telemetry.start_span(trace_id, "test_span", metadata={"test": True})
        print(f"✓ Created test span with ID: {span_id}")
        
        # Add a test event
        telemetry.track_event(trace_id, "test_event", {"test": True})
        print(f"✓ Added test event to trace")
        
        # End the span
        telemetry.end_span(span_id, {"completed": True})
        print(f"✓ Ended test span")
        
        # Add a test score
        telemetry.track_score(trace_id, "connection_test", 1.0, "Connection test successful")
        print(f"✓ Added test score to trace")
        
        # Flush data to Langfuse
        telemetry.flush()
        print(f"✓ Flushed data to Langfuse")
        
        print("\n✅ Connection to Langfuse appears to be working!")
        print(f"You can check your data at: {telemetry.host}")
        print(f"Look for trace ID: {trace_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Langfuse connection: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Langfuse connection...")
    result = test_langfuse_connection()
    
    if result:
        print("\nConnection test completed successfully.")
    else:
        print("\nConnection test failed. Check the error messages above.")
    
    # Give some time for async operations to complete
    time.sleep(2)
