# Voice-to-Text AI Agent: No-LLM Planning

## Overview
This document outlines the implementation of a voice-to-text conversion system that gives users the choice between using LLM-based processing or local conversion without LLM dependency. This approach provides flexibility for users with different requirements regarding privacy, performance, and resource availability.

## Architecture Extension
The existing architecture will be extended to support a dual-path approach:

1. **Original LLM Path**: Uses the full pipeline with LLM enhancement
2. **New Local-Only Path**: Uses local speech recognition without LLM dependency

## Implementation Strategy

### 1. Configuration System
- Add a configuration option for users to select their preferred transcription method:
  - `transcription_mode`: Options include "llm" (default) or "local"

### 2. Local Speech Recognition Module
Create a new module for local speech recognition:
```
src/
├── transcription/
│   ├── local_transcriber.py    # Local speech recognition implementation
```

### 3. Transcription Factory
Implement a factory pattern to select the appropriate transcription method:
```python
def get_transcriber(mode: str = "llm") -> BaseTranscriber:
    """Factory function to get the appropriate transcriber based on mode."""
    if mode == "llm":
        return LLMTranscriber()
    elif mode == "local":
        return LocalTranscriber()
    else:
        raise ValueError(f"Unsupported transcription mode: {mode}")
```

## Local Speech Recognition Implementation

### OpenAI Whisper
- State-of-the-art speech recognition model
- Can run completely locally
- Supports multiple languages
- High accuracy for various audio conditions
- Different model sizes available (tiny, base, small, medium, large)
- No API key required when running locally

## User Interface Changes
- Add a mode selection option in the CLI/UI:
  ```
  --mode [llm|local]  Specify the transcription mode (default: llm)
  ```
- For Streamlit UI (if implemented), add a radio button or dropdown for mode selection

## Performance Considerations
- Local transcription will be faster but potentially less accurate
- Local transcription requires no API costs but may need more local resources
- Different local options have different accuracy/resource tradeoffs

## Implementation Phases

### Phase 1: Core Implementation
1. Create the `local_transcriber.py` module with SpeechRecognition implementation
2. Implement the transcription factory pattern
3. Update the agent to use the factory based on configuration
4. Add CLI option for mode selection

### Phase 2: Enhanced Whisper Options
1. Add support for different Whisper model sizes (tiny, base, small, medium, large)
2. Implement language detection and multi-language support
3. Add performance metrics for accuracy assessment

### Phase 3: UI and UX Improvements
1. Add mode selection to Streamlit UI (if implemented)
2. Provide visual feedback on transcription quality
3. Allow switching modes during runtime

## Dependencies
Additional dependencies for local transcription:
- openai-whisper
- ffmpeg (system dependency for audio processing)
- torch (for running the Whisper model)
- numpy

## Testing Strategy
1. Unit tests for each local transcription implementation
2. Integration tests for the factory pattern
3. Comparison tests between LLM and local transcription for accuracy
4. Performance benchmarks for different approaches

## Future Considerations
1. Hybrid approach that uses local transcription with optional LLM enhancement
2. Adaptive system that chooses the best method based on audio quality
3. Fine-tuning options for local models to improve accuracy
