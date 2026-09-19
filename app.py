import os
import uuid
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from utils.audio_processor import process_input, cleanup_files
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question
from core.vector_store import delete_vector_store

st.set_page_config(page_title="AI Video Summarizer & Chat", page_icon="🎬", layout="wide")

# ---------- Session state ----------
def init_state():
    defaults = {
        "session_id": str(uuid.uuid4())[:8],
        "result": None,
        "chat_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

def reset_session():
    if st.session_state.result:
        try:
            delete_vector_store(st.session_state.result["transcript"])
        except Exception:
            pass
    st.session_state.result = None
    st.session_state.chat_history = []

# ---------- Pipeline runner ----------
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

# ---------- Hero Header ----------
st.markdown(
    """
    <div style="background:linear-gradient(90deg,#ff4b1f,#1fddff);
                padding:25px;border-radius:12px;text-align:center;margin-bottom:20px;">
        <h1 style="color:white;">🎬 AI Video Summarizer & Chat</h1>
        <p style="color:white;font-size:18px;">
            Transcribe, summarize & chat with any video — fast & fun!
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Input Area ----------
st.markdown("### 📥 Input")
col1, col2 = st.columns([2,1])

with col1:
    source = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    uploaded_file = st.file_uploader("Upload audio/video file", type=None)

with col2:
    language = st.selectbox("🌐 Language", ["hinglish","english","auto"])
    process_clicked = st.button("▶ Process Video", type="primary", use_container_width=True)
    if st.session_state.result:
        if st.button("🔄 New Video", use_container_width=True):
            reset_session()
            st.rerun()

# ---------- Run pipeline ----------
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
            progress_bar.empty()
            st.rerun()
        except Exception as e:
            progress_bar.empty()
            st.error(f"Pipeline failed: {e}")

# ---------- Results ----------
result = st.session_state.result

if result:
    st.markdown(f"## 🔴 {result['title']}")

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📝 Summary", "✅ Actions", "🔑 Decisions", "❓ Questions", "📄 Transcript", "💬 Chat"]
    )

    def card(content, color):
        st.markdown(
            f"<div style='background:{color};padding:15px;border-radius:10px;margin-bottom:10px;'>{content}</div>",
            unsafe_allow_html=True
        )

    with tab_summary:
        card(result["summary"], "#f0f8ff")

    with tab_actions:
        card(result["action_items"], "#e6ffe6")

    with tab_decisions:
        card(result["key_decisions"], "#fff0f5")

    with tab_questions:
        card(result["open_questions"], "#ffffe0")

    with tab_transcript:
        st.text_area("Full transcript", result["transcript"], height=400)

    with tab_chat:
        for msg in st.session_state.chat_history:
            bg = "#d1f0ff" if msg["role"]=="assistant" else "#fce4ec"
            st.markdown(
                f"<div style='background:{bg};padding:10px;border-radius:8px;margin-bottom:5px;'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        question = st.chat_input("Ask something about this video...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.markdown(
                f"<div style='background:#fce4ec;padding:10px;border-radius:8px;margin-bottom:5px;'>{question}</div>",
                unsafe_allow_html=True
            )
            with st.spinner("Thinking..."):
                answer = ask_question(result["rag_chain"], question)
            st.markdown(
                f"<div style='background:#d1f0ff;padding:10px;border-radius:8px;margin-bottom:5px;'>{answer}</div>",
                unsafe_allow_html=True
            )
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

