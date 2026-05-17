# HumSabaq AI — Intelligent Academic Assistant

> Built for Samsung Innovation Campus (SIC-2026) · AI & ML Program · Pakistan

An integrated AI platform that helps students **query documents**, **transcribe lectures**, and **never miss a class** — all running locally with no API limits.

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Node.js](https://img.shields.io/badge/Node.js-339933?style=flat&logo=nodedotjs&logoColor=white)](https://nodejs.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![MongoDB](https://img.shields.io/badge/MongoDB_Atlas-47A248?style=flat&logo=mongodb&logoColor=white)](https://mongodb.com/atlas)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com)

---

## What is HumSabaq AI?

Students in Pakistan face three painful academic problems: reading through hundreds of pages to find one answer, reviewing hours of lecture recordings, and manually tracking complex timetables. HumSabaq AI solves all three in one platform.

Three AI tools. One dashboard. Zero API limits.

---

## Modules

### 1. Document Manager — RAG-powered Q&A
Upload any PDF, DOCX, XLSX, or TXT file and chat with it instantly.

**How it works:**
- Chunks documents into 500-character overlapping segments
- Scores each chunk against your question using cosine-like token-overlap similarity
- Sends only the top-4 most relevant chunks (max 4,000 chars) to a local LLM
- Streams the answer back token-by-token via Server-Sent Events (typewriter effect)
- Saves full session history to MongoDB Atlas

**Features:**
- Document Q&A via RAG pipeline
- Auto-summary (5 bullet points)
- Quiz generator (5-question MCQ)
- Flashcard generator (6 flip cards)
- Session history with full conversation persistence
- Multi-format: PDF, DOCX, XLSX, TXT (up to 50MB)

**Stack:** Node.js · Express.js · Ollama (LLaMA 3.2:1b) · MongoDB Atlas · SSE Streaming

---

### 2. SmartVoice — Audio Intelligence Pipeline
Record or upload audio and get speaker-attributed transcripts, summaries, and semantic search across all your sessions.

**How it works:**
- Converts audio to mono 16kHz WAV, applies noise reduction
- Faster-Whisper transcribes with VAD filtering (13 languages including Urdu)
- pyannote diarizes speakers; voiceprint embeddings identify enrolled speakers by name
- LangGraph state machine orchestrates the full pipeline
- Ollama summarizes with structured JSON output; extractive fallback if Ollama unavailable
- Sentence-BERT enables semantic search across all historical sessions

**Features:**
- Speaker enrollment (3 voice samples per speaker)
- Diarized transcripts with timestamps and confidence scores
- LLM summarization with extractive fallback
- Semantic search across sessions
- Waveform rendering with bookmark markers
- TTS audio summary output
- Session export (ZIP)
- Supports: WAV, MP3, M4A, FLAC, OGG (up to 250MB)

**Stack:** Python · Flask · Faster-Whisper · pyannote · LangGraph · Ollama · Sentence-BERT · pyttsx3

---

### 3. Schedule Assistant — Smart Timetable Manager
Upload a photo of your university timetable and get automated email reminders 30 minutes before every class — with AI-generated study tips.

**How it works:**
- PyMuPDF extracts text from PDFs; Groq Vision processes images
- Groq AI (LLaMA 3) parses unstructured timetable content into structured JSON
- APScheduler triggers reminder jobs 30 minutes before each class
- Gmail SMTP delivers branded emails with 3 AI-generated subject-specific study tips

**Features:**
- Timetable extraction from JPG, PNG, PDF
- Weekly calendar view (color-coded)
- Multiple saved timetables with ACTIVE/INACTIVE status
- Automated email reminders with AI study tips
- Real-time popup alerts

**Stack:** Python · Django · SQLite · Groq AI (LLaMA 3) · APScheduler · Gmail SMTP · React.js (Vite)

---

## Architecture

```
HumSabaq AI
├── Document Manager     (Node.js / Express.js / MongoDB Atlas)
│   ├── RAG Engine       (vanilla JS — chunkText, cosineLikeSimilarity, retrieveChunks)
│   ├── Ollama SSE       (llama3.2:1b, port 11434)
│   └── Session API      (GET/POST/DELETE /sessions)
│
├── SmartVoice           (Python / Flask)
│   ├── LangGraph Pipeline  (audio → preprocess → diarize → transcribe → summarize → TTS)
│   ├── Faster-Whisper   (STT, VAD filtering)
│   ├── pyannote         (speaker diarization)
│   └── Sentence-BERT    (semantic search)
│
└── Schedule Assistant   (Python / Django / React.js)
    ├── Groq AI          (LLaMA 3 — timetable parsing)
    ├── APScheduler      (30-min reminder triggers)
    └── Gmail SMTP       (email delivery)
```

---

## Key Results

| Module | Result |
|---|---|
| Document Manager | Accurately answers questions on documents 50+ pages via RAG — 5–15s response time on CPU |
| SmartVoice | Speaker attribution with confidence scores (e.g. [eman] 47%, [moman] 57%) using 3-sample voiceprints |
| Schedule Assistant | Extracted 28 classes across 6 days from a single timetable image upload |
| Email Reminders | Delivered 30 minutes before class with AI study tips via Gmail SMTP |

---

## Why local inference?

All AI processing runs locally via Ollama — no external API calls, no usage limits, no data leaving your machine. This makes HumSabaq AI practical in environments with limited internet access and ensures complete privacy of student documents and audio.

---

## Team

Built as the Samsung Innovation Campus SIC-2026 Capstone Project.

| Member | Primary Modules |
|---|---|
| Ammara Tahir | Frontend Dashboard (React), Document Manager backend, Testing & Documentation |
| Ayesha Javaid Bajwa | Document Manager RAG Engine, Express.js architecture, MongoDB schema |
| Syeda Aiman Mumtaz Sherazi | SmartVoice — LangGraph pipeline, Faster-Whisper, pyannote, semantic search |
| Bashair Nasir | Schedule Assistant — Groq AI parsing, APScheduler, Gmail SMTP, Django backend |

**Supervisors:** Tahseen Zia, Ayesha Sahar · Submission: April 5, 2026

---

## Setup

### Prerequisites
- Node.js v18+
- Python 3.10+
- [Ollama](https://ollama.ai) running locally with `llama3.2:1b` pulled
- MongoDB Atlas connection string
- Groq API key (for Schedule Assistant)
- HuggingFace token (for pyannote diarization)

### Document Manager
```bash
cd document-manager
npm install
# Add MONGODB_URI to .env
node server.js
# Runs on http://localhost:3000
```

### SmartVoice
```bash
cd smartvoice
pip install -r requirements.txt
# Add HF_TOKEN to .env
python app.py
```

### Schedule Assistant
```bash
cd schedule-assistant
pip install -r requirements.txt
# Add GROQ_API_KEY and Gmail credentials to .env
python manage.py migrate
python manage.py runserver
# Frontend: cd frontend && npm install && npm run dev
```

---

## Lessons learned

- RAG fundamentally changes what document Q&A can handle — without it, documents over ~30 pages fail silently as relevant content falls outside the context window
- SSE streaming is critical for perceived responsiveness on CPU-bound LLM inference — users see output start immediately even when total generation takes 15 seconds
- A modular architecture with explicit fallback chains (Ollama → extractive, ONNX → SpeechBrain) is essential for production stability
- Speaker enrollment with just 3 voice samples substantially improves diarization quality in real-world multi-speaker scenarios

---

*Samsung Innovation Campus × Bramerz · Pakistan · 2026*
