---
Title: Lecture Capture AI Pipeline
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: MIT
python_version: 3.10
---

# Lecture Capture AI Pipeline

An AI-powered tool that automatically processes lecture videos to generate:

-  **Transcription** with precise timestamps
-  **Summaries** for each video segment
-  **Key phrases** extraction
-  **Timeline** organization

## Features

- **Automatic Speech Recognition**: Uses OpenAI's Whisper model for accurate transcription
- **Text Summarization**: Leverages BART for concise segment summaries
- **Key Phrase Extraction**: Identifies important terms using spaCy NLP
- **Timeline Generation**: Creates structured output with timestamps

## How to Use

1. **Upload Video**: Click on the video input area and upload your lecture video
2. **Process**: Click the "🚀 Process Video" button
3. **Review Results**: View the generated timeline with transcriptions, summaries, and key phrases

## Technical Specifications

- **Maximum Video Length**: 15 minutes (optimized for Hugging Face Spaces)
- **Supported Formats**: MP4, AVI, MOV
- **Processing Time**: 2-5 minutes depending on video length
- **Models Used**:
  - Whisper (tiny) for speech recognition
  - BART-large-CNN for summarization
  - spaCy en_core_web_sm for NLP

## Requirements

- Clear audio quality for best transcription results
- Stable internet connection
- Modern web browser

## Limitations

- Processing is limited to 15 minutes of video content
- Requires FFmpeg for video processing
- Performance depends on Hugging Face Spaces resources

## Privacy

- Videos are processed temporarily and not stored
- All temporary files are automatically cleaned up after processing

## License

MIT License - Feel free to use and modify for your projects!

## Support

If you encounter issues, please check:
1. Video format is supported (MP4, AVI, MOV)
2. Audio quality is clear
3. Video length is under 15 minutes

For technical issues, please open an issue on the repository.
