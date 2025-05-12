# Voice-to-Text AI Agent

An autonomous agent for converting M4A voice recordings to text with flexible processing options. Supports both local transcription without LLM dependency and enhanced transcription with LLM processing.

## Features

- Convert M4A audio files to text transcriptions
- Two transcription modes:
  - Local mode: Uses Whisper locally without requiring an API key
  - LLM mode: Uses OpenAI's API for transcription
- Optional LLM post-processing for enhanced formatting and summarization
- Preprocess audio for improved transcription quality
- Autonomous workflow with error handling
- Simple command-line interface
- Optional web interface (with Streamlit)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/voice-to-text-ai.git
cd voice-to-text-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys (optional for local transcription mode):
```bash
# Create a .env file with your API keys
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```

> **Note**: API keys are only required if you use the LLM transcription mode or LLM post-processing. Local transcription mode works without any API keys.

### Local Transcription Dependencies

To use the local transcription mode, you need to install the following dependencies:

```bash
pip install openai-whisper torch numpy<2.0.0 numba
```

Alternatively, you can use the provided installation script:

```bash
python install_local_deps.py
```

This script will check for missing dependencies and install them for you.

**Important Notes**:
- Local transcription requires NumPy < 2.0.0 due to compatibility issues with Whisper
- For GPU acceleration, ensure you have a CUDA-compatible GPU and the appropriate CUDA toolkit installed
- The first time you use local transcription with a specific model size, it will download the model which may take some time depending on your internet connection

## Usage

### Command Line Interface

Convert an M4A file to text using local transcription (default):
```bash
python -m src.main /path/to/audio.m4a
```

Use OpenAI API for transcription:
```bash
python -m src.main /path/to/audio.m4a --mode=llm
```

Enable LLM post-processing for enhanced formatting:
```bash
python -m src.main /path/to/audio.m4a --use-llm --enhance-formatting --fix-punctuation --summarize
```

Specify Whisper model size for local transcription:
```bash
python -m src.main /path/to/audio.m4a --model-size=small
```

Use GPU acceleration for local transcription (if available):
```bash
python -m src.main /path/to/audio.m4a --device=cuda
```

### Python API

```python
from src.agent.agent import VoiceToTextAgent, AgentOptions
from src.transcription.models import TranscriptionOptions
from src.llm.processor import ProcessingOptions

# Initialize with local transcription (no LLM)
agent_options = AgentOptions(
    transcription_mode="local",  # Use "llm" for OpenAI API
    skip_llm_processing=True,  # Skip LLM post-processing
    transcription_options=TranscriptionOptions(
        model_size="base",
        device="cpu"  # Use "cuda" for GPU acceleration
    )
)

# Initialize the agent
agent = VoiceToTextAgent(options=agent_options)

# Process a file
result = agent.process_file("/path/to/audio.m4a")

# Access the result
print(result.text)

# Example with LLM post-processing
agent_with_llm = VoiceToTextAgent(
    options=AgentOptions(
        transcription_mode="local",
        skip_llm_processing=False,
        transcription_options=TranscriptionOptions(model_size="base"),
        processing_options=ProcessingOptions(
            enhance_formatting=True,
            fix_punctuation=True,
            summarize=True
        )
    )
)

result_with_llm = agent_with_llm.process_file("/path/to/audio.m4a")
print(result_with_llm.text)
print(result_with_llm.summary)
```

### Web Interface (Optional)

Start the Streamlit web interface:
```bash
streamlit run src/web/app.py
```

Then open your browser at http://localhost:8501

## Dependencies

- Python 3.9+
- pydub/ffmpeg for audio conversion
- OpenAI Whisper for speech recognition (runs locally or via API)
- PyTorch for running Whisper locally
- OpenAI API for LLM processing (optional)
- Pydantic for data validation
- Streamlit (optional) for web UI

## Project Structure

```
voice_to_text_ai/
├── src/
│   ├── audio/                # Audio processing modules
│   ├── transcription/        # Transcription modules
│   ├── llm/                  # LLM integration
│   ├── agent/                # Agent logic
│   └── utils/                # Utility functions
├── tests/                    # Test directory
└── examples/                 # Example usage and sample files
```

## Development

Run tests:
```bash
pytest
```

Run linting:
```bash
flake8 src tests
```

## License

MIT
