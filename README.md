# HamSabaqAi

SmartVoice is an agent-based audio intelligence system that cleans audio, transcribes speech, summarizes key points, and optionally generates TTS output. It supports diarization, speaker attribution, semantic search, and offline session storage.

## Quick start

### 1) Create and activate a Python environment

Example (conda):

```bash
conda create -n smartvoice python=3.10
conda activate smartvoice
```

### 2) Install dependencies

```bash
pip install -r SmartVoice/requirements.txt
```

### 3) Run the app

```bash
python SmartVoice/app.py
```

Open http://127.0.0.1:7860 in your browser.

## Required and optional setup

### Required
- Python 3.9+ (3.10 recommended)

### Optional but recommended
- FFmpeg for better audio format support (m4a/aac/mp4)
	- Windows: install from https://ffmpeg.org or via winget
- Hugging Face token for diarization
	- Set environment variable `HF_TOKEN`
	- Example (PowerShell):
		```powershell
		$env:HF_TOKEN="your_token_here"
		```
- Ollama for local summarization (if you want LLM summaries)
	- Install Ollama and run `ollama pull llama3.2:1b`
	- The app falls back to extractive summarization if Ollama is not available

## What the app does

- Validates and normalizes audio (mono 16 kHz)
- Optional noise reduction
- Speaker diarization (pyannote)
- Transcription (Faster-Whisper)
- Speaker attribution using enrolled voiceprints
- Summarization (Ollama or extractive fallback)
- Keyword extraction
- Optional TTS generation
- Session history, search, bookmarks, and export

## Data and outputs

- Uploaded audio and outputs are stored under the project `SmartVoice/` folder
- Session history and embeddings are stored in:
	- Windows: `C:\Users\<you>\.smartvoice\`
	- Linux/Mac: `~/.smartvoice/`

## Notes and troubleshooting

- If diarization is skipped, check `HF_TOKEN` and internet access for model download.
- If speaker enrollment is disabled, add `voiceprint.onnx` to `~/.smartvoice/embeddings/` or install `speechbrain`.
- If some audio formats fail, install FFmpeg and retry.
- If summarization is empty, ensure Ollama is running or rely on the extractive fallback.

## Project layout

- SmartVoice/app.py: Flask web app
- SmartVoice/agents.py: LangGraph pipeline and agents
- SmartVoice/storage.py: Offline session storage
- SmartVoice/templates/index.html: UI