"""
Streamlit UI for the AI Video Assistant.

Drop this file in the project root (same level as main.py) and run:
    python -m streamlit run app.py
"""

import os
import json
import uuid
from dotenv import load_dotenv
load_dotenv()  # must come before any core/ or utils/ import

import streamlit as st

# If a YTDLP_COOKIES secret is set (Manage app -> Settings -> Secrets),
# write it to cookies.txt so audio_processor.py can pick it up automatically.
if "YTDLP_COOKIES" in st.secrets and not os.path.exists("cookies.txt"):
    with open("cookies.txt", "w") as f:
        f.write(st.secrets["YTDLP_COOKIES"])

from utils.audio_processor import process_input, cleanup_files
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question
from core.vector_store import delete_vector_store

SAMPLE_VIDEO_PATH = "sample_data/sample_video.json"

st.set_page_config(page_title="AI Video Summarizer & Chat", page_icon="🎬", layout="wide")


# =====================================================================
# STYLING
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .hero {
        background: linear-gradient(135deg, #6D28D9 0%, #DB2777 100%);
        padding: 1.8rem 1.6rem;
        border-radius: 18px;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(109, 40, 217, 0.25);
    }
    .hero h1 {
        font-family: 'Poppins', sans-serif;
        color: white;
        font-size: 1.9rem;
        margin: 0 0 0.3rem 0;
    }
    .hero p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin: 0;
    }

    .input-card {
        background: var(--background-color, #ffffff);
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        margin-bottom: 1.2rem;
    }

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        border: none !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6D28D9 0%, #DB2777 100%) !important;
        box-shadow: 0 4px 14px rgba(109, 40, 217, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(109, 40, 217, 0.3);
    }

    .result-title {
        background: linear-gradient(135deg, rgba(109,40,217,0.08) 0%, rgba(219,39,119,0.08) 100%);
        border-left: 4px solid #DB2777;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
    }
    .result-title h2 {
        margin: 0;
        font-family: 'Poppins', sans-serif;
        font-size: 1.35rem;
        word-break: break-word;
    }

    .content-card {
        background: var(--background-color, #ffffff);
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        line-height: 1.65;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 8px 14px;
        font-weight: 600;
    }

    .badge {
        display: inline-block;
        background: rgba(109,40,217,0.12);
        color: #6D28D9;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }

    .sample-badge {
        background: rgba(16,185,129,0.12);
        color: #059669;
    }

    @media (max-width: 640px) {
        .hero h1 { font-size: 1.5rem; }
        .hero p { font-size: 0.9rem; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
</style>
""", unsafe_allow_html=True)


# ---------- Session state ----------
def init_state():
    defaults = {
        "session_id": str(uuid.uuid4())[:8],
        "result": None,
        "chat_history": [],
        "is_sample": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


def reset_session():
    if st.session_state.result and not st.session_state.is_sample:
        try:
            delete_vector_store(st.session_state.result["transcript"])
        except Exception:
            pass
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.is_sample = False


# ---------- Pipeline runner (real video) ----------
def run_pipeline(source: str, language: str, progress_cb):
    is_url_source = source.startswith(("http://", "https://"))

    progress_cb("Downloading / loading audio...", 0.1)
    chunks, wav_path, raw_path = process_input(source)

    progress_cb(f"Transcribing {len(chunks)} chunk(s)...", 0.3)
    transcript = transcribe_all(chunks, language=language)

    progress_cb("Cleaning up temporary audio files...", 0.5)
    cleanup_files(chunks, wav_path, raw_path, is_url_source=is_url_source)

    progress_cb("Generating title...", 0.6)
    title = generate_title(transcript)

    progress_cb("Summarizing...", 0.7)
    summary = summarize(transcript)

    progress_cb("Extracting action items, decisions, questions...", 0.85)
    extracted = extract_all(transcript)

    progress_cb("Building chat index...", 0.95)
    rag_chain = build_rag_chain(transcript)

    progress_cb("Done!", 1.0)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": extracted["action_items"],
        "key_decisions": extracted["key_decisions"],
        "open_questions": extracted["open_questions"],
        "rag_chain": rag_chain,
    }


# ---------- Sample video loader (instant, no YouTube/LLM calls needed for text) ----------
def load_sample_video(progress_cb):
    progress_cb("Loading pre-processed sample...", 0.3)
    with open(SAMPLE_VIDEO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    progress_cb("Building chat index for sample...", 0.7)
    rag_chain = build_rag_chain(data["transcript"])

    progress_cb("Done!", 1.0)

    return {
        "title": data["title"],
        "transcript": data["transcript"],
        "summary": data["summary"],
        "action_items": data["action_items"],
        "key_decisions": data["key_decisions"],
        "open_questions": data["open_questions"],
        "rag_chain": rag_chain,
    }


# =====================================================================
# HERO HEADER
# =====================================================================
st.markdown("""
<div class="hero">
    <h1>🎬 AI Video Summarizer & Chat</h1>
    <p>Transcribe, summarize, and chat with any YouTube video or local file — English & Hinglish supported.</p>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# MAIN INPUT AREA
# =====================================================================
if st.session_state.result is None:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    language = st.selectbox(
        "⚙️ Language",
        options=["hinglish", "english", "auto"],
        index=0,
        help="hinglish → Sarvam AI · english → local Whisper · auto → detect per chunk",
    )

    source = st.text_input(
        "🔗 YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    uploaded_file = st.file_uploader("📁 ...or upload a local audio/video file", type=None)

    col1, col2 = st.columns([2, 1])
    with col1:
        process_clicked = st.button("▶  Process video", type="primary", use_container_width=True)
    with col2:
        sample_clicked = st.button("🎯 Try a sample (instant)", use_container_width=True)

    st.caption(
        "⚠️ Live YouTube downloads may occasionally fail due to YouTube's bot detection on "
        "cloud servers — use **Try a sample** for a guaranteed instant demo."
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if sample_clicked:
        progress_bar = st.progress(0, text="Starting...")

        def progress_cb(msg, pct):
            progress_bar.progress(pct, text=msg)

        try:
            st.session_state.result = load_sample_video(progress_cb)
            st.session_state.chat_history = []
            st.session_state.is_sample = True
            progress_bar.empty()
            st.rerun()
        except Exception as e:
            progress_bar.empty()
            st.error(f"Could not load sample: {e}")

    if process_clicked:
        if uploaded_file is None and not source.strip():
            st.error("Enter a YouTube URL or upload a file first.")
        else:
            input_source = source.strip()
            if uploaded_file is not None:
                temp_upload_path = f"downloads/upload_{st.session_state.session_id}_{uploaded_file.name}"
                os.makedirs("downloads", exist_ok=True)
                with open(temp_upload_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                input_source = temp_upload_path

            progress_bar = st.progress(0, text="Starting...")

            def progress_cb(msg, pct):
                progress_bar.progress(pct, text=msg)

            try:
                st.session_state.result = run_pipeline(input_source, language, progress_cb)
                st.session_state.chat_history = []
                st.session_state.is_sample = False
                progress_bar.empty()
                st.rerun()
            except Exception as e:
                progress_bar.empty()
                st.error(f"Pipeline failed: {e}")

else:
    if st.button("🔄 Start a new video"):
        reset_session()
        st.rerun()


# =====================================================================
# RESULTS
# =====================================================================
result = st.session_state.result

if result:
    badge_html = '<span class="badge sample-badge">🎯 Sample Demo</span>' if st.session_state.is_sample else '<span class="badge">🔴 Analyzed</span>'
    st.markdown(f"""
    <div class="result-title">
        {badge_html}
        <h2>{result['title']}</h2>
    </div>
    """, unsafe_allow_html=True)

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📝 Summary", "✅ Actions", "🔑 Decisions", "❓ Questions", "📄 Transcript", "💬 Chat"]
    )

    with tab_summary:
        st.markdown(f'<div class="content-card">{result["summary"]}</div>', unsafe_allow_html=True)

    with tab_actions:
        st.markdown(f'<div class="content-card">{result["action_items"]}</div>', unsafe_allow_html=True)

    with tab_decisions:
        st.markdown(f'<div class="content-card">{result["key_decisions"]}</div>', unsafe_allow_html=True)

    with tab_questions:
        st.markdown(f'<div class="content-card">{result["open_questions"]}</div>', unsafe_allow_html=True)

    with tab_transcript:
        st.text_area("Full transcript", result["transcript"], height=400, label_visibility="collapsed")

    with tab_chat:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask something about this video...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_question(result["rag_chain"], question)
                st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
