"""
Streamlit web interface for the voice-to-text agent.
"""
import os
import tempfile
from pathlib import Path
import time
import sys
import streamlit as st
from dotenv import load_dotenv
from loguru import logger

# Add the parent directory to the path so we can import the src package
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agent.agent import VoiceToTextAgent, AgentOptions
from src.transcription.models import TranscriptionOptions
from src.llm.processor import ProcessingOptions
from src.utils.helpers import setup_logging

# Import the dependency check function
try:
    from src.transcription.local_transcriber import check_dependencies
    # Check dependencies at startup
    local_deps_ok, local_deps_error = check_dependencies()
except ImportError:
    local_deps_ok = False
    local_deps_error = "Could not import dependency checker"

# Load environment variables
load_dotenv()

# Set up logging
setup_logging()

# Set page configuration
st.set_page_config(
    page_title="Voice-to-Text AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main Streamlit application."""
    st.title("🎙️ Voice-to-Text AI Agent")
    st.markdown(
        """
        Upload an M4A voice recording and convert it to text with flexible processing options.
        Choose between local transcription (no API key needed) or enhanced processing with LLM.
        """
    )
    
    # Sidebar for options
    with st.sidebar:
        st.header("Options")
        
        # Show dependency warning if local dependencies are missing
        if not local_deps_ok:
            st.warning(
                f"⚠️ Local transcription dependencies missing: {local_deps_error}\n\n"
                "Install required packages:\n"
                "```bash\npip install openai-whisper torch numpy<2.0.0 numba\n```"
            )
        
        # Transcription options
        st.subheader("Transcription Options")
        
        transcription_mode = st.radio(
            "Transcription Mode",
            options=["local", "llm"],
            index=0 if local_deps_ok else 1,  # Default to LLM if local deps missing
            help="Local uses Whisper locally, LLM uses OpenAI API",
            horizontal=True,
        )
        
        # Show warning if local mode selected but dependencies missing
        if transcription_mode == "local" and not local_deps_ok:
            st.error(
                "Local transcription mode requires additional dependencies.\n"
                "Please install them using:\n"
                "```bash\npip install openai-whisper torch numpy<2.0.0 numba\n```"
            )
        
        model_size = st.selectbox(
            "Model Size",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower",
        )
        
        if transcription_mode == "local":
            device = st.radio(
                "Device",
                options=["cpu", "cuda"],
                index=0,
                help="Use CUDA if you have a compatible GPU",
                horizontal=True,
            )
        
        language = st.text_input(
            "Language Code",
            value="",
            help="Language code (e.g., 'en', 'fr'). Leave empty for auto-detection",
        )
        
        # LLM processing options
        st.subheader("LLM Processing Options")
        
        use_llm = st.checkbox(
            "Use LLM Post-Processing",
            value=False,
            help="Enable LLM enhancement (requires API key)",
        )
        
        if use_llm:
            enhance_formatting = st.checkbox(
                "Enhance Formatting",
                value=True,
                help="Improve formatting and structure",
            )
            
            fix_punctuation = st.checkbox(
                "Fix Punctuation",
                value=True,
                help="Fix punctuation and capitalization",
            )
            
            summarize = st.checkbox(
                "Generate Summary",
                value=False,
                help="Generate a summary of the content",
            )
            
            llm_model = st.selectbox(
                "LLM Model",
                options=["gpt-3.5-turbo", "gpt-4o", "gpt-4"],
                index=1,
                help="Model to use for LLM processing",
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Temperature for LLM generation (0-1)",
            )
        else:
            # Default values when LLM is not used
            enhance_formatting = True
            fix_punctuation = True
            summarize = False
            llm_model = "gpt-4o"
            temperature = 0.3
        
        # Advanced options
        st.subheader("Advanced Options")
        save_intermediate = st.checkbox(
            "Save Intermediate Files",
            value=False,
            help="Save intermediate files for debugging",
        )
    
    # Main content
    uploaded_file = st.file_uploader("Upload an M4A file", type=["m4a"])
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Process button
        if st.button("Process Audio"):
            # Set up options
            transcription_options = TranscriptionOptions(
                model_size=model_size,
                language=language if language else None,
                device=device if transcription_mode == "local" else "cpu",
            )
            
            processing_options = ProcessingOptions(
                enhance_formatting=enhance_formatting,
                fix_punctuation=fix_punctuation,
                summarize=summarize,
                model=llm_model,
                temperature=temperature,
            )
            
            agent_options = AgentOptions(
                output_format="text",
                save_intermediate_files=save_intermediate,
                transcription_mode=transcription_mode,
                transcription_options=transcription_options,
                processing_options=processing_options,
                skip_llm_processing=not use_llm,
            )
            
            # Initialize agent
            agent = VoiceToTextAgent(options=agent_options)
            
            # Check for API key if needed
            api_key_missing = False
            if transcription_mode == "llm" or use_llm:
                if not os.getenv("OPENAI_API_KEY"):
                    if transcription_mode == "llm":
                        st.error("Error: OPENAI_API_KEY is required for LLM transcription mode. Please set the API key in your .env file or switch to local mode.")
                        api_key_missing = True
                    if use_llm and not api_key_missing:
                        st.warning("Warning: OPENAI_API_KEY not found. Disabling LLM post-processing.")
                        use_llm = False
                        agent_options.skip_llm_processing = True
            
            # Process file with progress
            if not api_key_missing:
                with st.spinner("Processing audio..."):
                    try:
                        start_time = time.time()
                        
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Convert to WAV
                        status_text.text("Step 1/4: Converting M4A to WAV...")
                        progress_bar.progress(10)
                        
                        # Step 2: Preprocess audio
                        status_text.text("Step 2/4: Preprocessing audio...")
                        progress_bar.progress(30)
                        
                        # Step 3: Transcribing
                        status_text.text("Step 3/4: Transcribing audio...")
                        progress_bar.progress(50)
                        
                        # Process file
                        result = agent.process_file(tmp_file_path)
                        
                        # Step 4: LLM Processing (if enabled)
                        if use_llm:
                            status_text.text("Step 4/4: Enhancing with LLM...")
                        else:
                            status_text.text("Step 4/4: Finalizing...")
                        progress_bar.progress(90)
                        
                        # Complete
                        progress_bar.progress(100)
                        status_text.text("Processing complete!")
                    
                        # Display results
                        st.subheader("Transcription Result")
                        st.text_area(
                            "Enhanced Text",
                            result.text,
                            height=300,
                        )
                        
                        # Display summary if available
                        if summarize and result.summary:
                            st.subheader("Summary")
                            st.text_area(
                                "Summary",
                                result.summary,
                                height=150,
                            )
                        
                        # Display metadata
                        st.subheader("Metadata")
                        st.json({
                            "processing_time": f"{result.duration:.2f} seconds",
                            "transcription_mode": result.metadata.get("transcription_mode", "unknown"),
                            "llm_processing": "Enabled" if not result.metadata.get("skipped_llm_processing", True) else "Disabled",
                            "language": result.metadata.get("language", "unknown"),
                            "confidence": result.metadata.get("transcription_confidence", "N/A"),
                            "audio_duration": f"{result.metadata.get('audio_duration', 0):.2f} seconds",
                        })
                    
                        # Create download buttons
                        col1, col2 = st.columns(2)
                        
                        # Save as text file
                        text_file_path = f"{tmp_file_path}_result.txt"
                        with open(text_file_path, "w") as f:
                            f.write(result.text)
                            if result.summary:
                                f.write("\n\nSUMMARY:\n")
                                f.write(result.summary)
                        
                        with col1:
                            with open(text_file_path, "r") as f:
                                st.download_button(
                                    label="Download as Text",
                                    data=f.read(),
                                    file_name=f"{uploaded_file.name.split('.')[0]}_transcription.txt",
                                    mime="text/plain",
                                )
                        
                        # Save as JSON file
                        json_file_path = f"{tmp_file_path}_result.json"
                        with open(json_file_path, "w") as f:
                            import json
                            json.dump({
                                "text": result.text,
                                "summary": result.summary,
                                "metadata": {
                                    "processing_time": result.duration,
                                    "language": result.metadata.get("language", "unknown"),
                                    "confidence": result.metadata.get("transcription_confidence", "N/A"),
                                    "audio_duration": result.metadata.get("audio_duration", 0),
                                }
                            }, f, indent=2)
                        
                        with col2:
                            with open(json_file_path, "r") as f:
                                st.download_button(
                                    label="Download as JSON",
                                    data=f.read(),
                                    file_name=f"{uploaded_file.name.split('.')[0]}_transcription.json",
                                    mime="application/json",
                                )                  
                    except ImportError as e:
                        if "whisper" in str(e).lower() or "numba" in str(e).lower() or "numpy" in str(e).lower():
                            st.error("Error: Missing or incompatible dependencies for local transcription.")
                            st.code("pip install openai-whisper torch numpy<2.0.0 numba", language="bash")
                            logger.error(f"Dependency error: {str(e)}")
                        else:
                            st.error(f"Import error: {str(e)}")
                            logger.error(f"Import error: {str(e)}")
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                        logger.error(f"Error processing audio: {str(e)}")
                    finally:
                        # Clean up
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                        # Only try to remove these if they were created (inside the try block)
                        if 'text_file_path' in locals() and os.path.exists(text_file_path):
                            os.unlink(text_file_path)
                        if 'json_file_path' in locals() and os.path.exists(json_file_path):
                            os.unlink(json_file_path)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center">
            <p>Voice-to-Text AI Agent | Built with Streamlit and OpenAI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
