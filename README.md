# Voice-to-Text AI Agent

An autonomous agent for converting M4A voice recordings to text using LLM processing.

## Features

- Convert M4A audio files to text transcriptions
- Preprocess audio for improved transcription quality
- Enhance transcriptions using LLM processing
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

4. Set up API keys:
```bash
# Create a .env file with your API keys
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```

## Usage

### Command Line Interface

Convert an M4A file to text:
```bash
python -m src.main /path/to/audio.m4a
```

With additional options:
```bash
python -m src.main /path/to/audio.m4a --enhance-formatting --fix-punctuation --summarize
```

### Python API

```python
from src.agent.agent import VoiceToTextAgent

# Initialize the agent
agent = VoiceToTextAgent()

# Process a file
result = agent.process_file(
    file_path="/path/to/audio.m4a",
    options={
        "enhance_formatting": True,
        "fix_punctuation": True,
        "summarize": False
    }
)

# Access the result
print(result.text)
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
- OpenAI Whisper for speech recognition
- OpenAI API for LLM processing
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
