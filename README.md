# AI-Video-Summarization-Chat

AI-powered video assistant that transcribes YouTube/local videos (English + Hinglish), auto-generates summaries & action items via LLM, and supports RAG-based chat over video content. Built with Python, LangChain, Whisper, Ollama & Streamlit.

## ✨ Features

- 🎥 **Flexible input** — process any YouTube URL or a local audio/video file
- 🗣️ **Dual speech-to-text engine** — local `faster-whisper` for English, [Sarvam AI](https://www.sarvam.ai/) for Hindi-English code-mixed (Hinglish) speech, with automatic language routing per chunk
- 📝 **LLM-powered analysis** — auto-generated title, summary, action items, key decisions, and open questions, via a local LLM through [Ollama](https://ollama.com/)
- 💬 **RAG chat** — ask natural-language questions about the video's content, grounded in the actual transcript using LangChain + ChromaDB + HuggingFace embeddings
- 🖥️ **Streamlit web UI** — tabbed results, live progress tracking, and a chat interface, in addition to a CLI version
- 🧹 **Automatic cleanup** — temporary audio files and vector store collections are deleted after each run

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Audio acquisition | `yt-dlp`, `pydub`, `ffmpeg` |
| Speech-to-text (English) | `faster-whisper` |
| Speech-to-text (Hinglish) | Sarvam AI API |
| LLM orchestration | LangChain (LCEL) |
| LLM inference | Ollama (`llama3.2`) |
| Vector store / RAG | ChromaDB + HuggingFace embeddings |
| UI | Streamlit |

## 📂 Project Structure

```
video_agent/
├── core/
│   ├── transcriber.py     # Whisper + Sarvam transcription, language routing
│   ├── summarizer.py      # Title + summary generation
│   ├── extractor.py       # Action items / decisions / questions extraction
│   ├── rag_engine.py      # RAG chain for chat
│   └── vector_store.py    # Chroma vector store build/load/delete
├── utils/
│   └── audio_processor.py # Download, convert, chunk, cleanup audio
├── main.py                 # CLI entry point
├── app.py                  # Streamlit UI entry point
├── requirements.txt
└── .env                     # API keys & config (not committed)
```

## 🚀 Getting Started

### Prerequisites
- Python ≥ 3.10
- [Ollama](https://ollama.com/) installed and running locally, with a model pulled:
  ```bash
  ollama pull llama3.2
  ```
- [FFmpeg](https://ffmpeg.org/) installed and available on your system PATH
- A [Sarvam AI](https://www.sarvam.ai/) API key (for Hinglish transcription)

### Installation
```bash
git clone https://github.com/ricky231b263/AI-Video-Summarization-Chat.git
cd AI-Video-Summarization-Chat

python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the project root:
```dotenv
WHISPER_MODEL=base
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_STT_MODEL=saaras:v3
SARVAM_STT_LANG=hi-IN
```

### Running

**CLI version:**
```bash
python main.py
```

**Streamlit web UI:**
```bash
python -m streamlit run app.py
```

## 🎯 Usage

1. Provide a YouTube URL or upload a local audio/video file
2. Choose a language mode: `english`, `hinglish`, or `auto` (auto-detect per chunk)
3. Wait for the pipeline to transcribe and analyze the content
4. Review the generated title, summary, action items, key decisions, and open questions
5. Use the chat tab to ask follow-up questions about the video

## ⚡ Performance Notes

- Sarvam API calls are parallelized (I/O-bound), while Whisper transcription is parallelized based on available CPU cores
- The extraction step (action items, decisions, questions) runs as a single combined LLM call instead of three separate passes
- Temporary audio files and Chroma vector store collections are automatically deleted after each run to avoid disk buildup

## 📌 Known Limitations

- Currently tuned for single-user local/demo use; concurrent multi-user deployments would need additional session isolation for file storage
- Transcription and LLM inference run on CPU by default; a GPU significantly improves speed if available

## 📄 License

This project is open source — feel free to fork and adapt it. (Add your preferred license here, e.g. MIT.)
