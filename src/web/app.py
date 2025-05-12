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

# Add the parent directory to the path so we can import the src package
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agent.agent import VoiceToTextAgent, AgentOptions
from src.transcription.models import TranscriptionOptions
from src.llm.processor import ProcessingOptions
from src.utils.helpers import setup_logging

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
        Upload an M4A voice recording and convert it to text with AI enhancement.
        """
    )
    
    # Sidebar for options
    with st.sidebar:
        st.header("Options")
        
        # Transcription options
        st.subheader("Transcription Options")
        model_size = st.selectbox(
            "Model Size",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower",
        )
        
        language = st.text_input(
            "Language Code",
            value="",
            help="Language code (e.g., 'en', 'fr'). Leave empty for auto-detection",
        )
        
        # LLM processing options
        st.subheader("LLM Processing Options")
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
                transcription_options=transcription_options,
                processing_options=processing_options,
            )
            
            # Initialize agent
            agent = VoiceToTextAgent(options=agent_options)
            
            # Process file with progress
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
                    
                    # Step 4: LLM Processing
                    status_text.text("Step 4/4: Enhancing with LLM...")
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
                        "language": result.metadata.get("language", "unknown"),
                        "confidence": result.metadata.get("transcription_confidence", "N/A"),
                        "audio_duration": f"{result.metadata.get('audio_duration', 0):.2f} seconds",
                    })
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as text_file:
                        text_file.write(result.text.encode("utf-8"))
                        text_file_path = text_file.name
                    
                    with open(text_file_path, "rb") as file:
                        col1.download_button(
                            label="Download Text",
                            data=file,
                            file_name=f"{uploaded_file.name.split('.')[0]}.txt",
                            mime="text/plain",
                        )
                    
                    # Create JSON for download
                    import json
                    json_data = {
                        "text": result.text,
                        "summary": result.summary,
                        "metadata": {
                            **result.metadata,
                            "processing_time": result.duration,
                        }
                    }
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as json_file:
                        json_file.write(json.dumps(json_data, indent=2).encode("utf-8"))
                        json_file_path = json_file.name
                    
                    with open(json_file_path, "rb") as file:
                        col2.download_button(
                            label="Download JSON",
                            data=file,
                            file_name=f"{uploaded_file.name.split('.')[0]}.json",
                            mime="application/json",
                        )
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                finally:
                    # Clean up temporary files
                    try:
                        os.unlink(tmp_file_path)
                        os.unlink(text_file_path)
                        os.unlink(json_file_path)
                    except:
                        pass
    
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
