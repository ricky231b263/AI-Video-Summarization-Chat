from dotenv import load_dotenv
load_dotenv()   # must come before any core/ or utils/ import

import sys
from utils.audio_processor import process_input, cleanup_files
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question
from core.vector_store import delete_vector_store


def run_pipeline(source: str, language: str = "hinglish") -> dict | None:
    print("Starting AI Video Assistant ...")

    is_url_source = source.startswith(("http://", "https://"))

    try:
        chunks, wav_path, raw_path = process_input(source)
        transcript = transcribe_all(chunks, language=language)
    except Exception as e:
        print(f"❌ Pipeline failed during transcription: {e}")
        return None

    cleanup_files(chunks, wav_path, raw_path, is_url_source=is_url_source)

    print(f"Raw transcription (first 300 chars): {transcript[:300]}")

    title        = generate_title(transcript)
    summary      = summarize(transcript)
    extracted    = extract_all(transcript)
    action_item  = extracted["action_items"]
    decisions    = extracted["key_decisions"]
    questions    = extracted["open_questions"]
    rag_chain    = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish/auto) [hinglish]: ").strip() or "hinglish"

    result = run_pipeline(source, language)
    if result is None:
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"🔴 Title: {result['title']}")
    print(f"\n📝 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\n💬 Chat with your content (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")

    delete_vector_store(result["transcript"])