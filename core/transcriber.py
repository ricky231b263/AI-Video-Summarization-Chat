import os
import time
import multiprocessing
import requests
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel

# ---------- Config ----------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

SARVAM_STT_URL = os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")
SARVAM_MODEL   = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_LANG    = os.getenv("SARVAM_STT_LANG", "hi-IN")

_model = None


# ---------- Whisper (local, faster-whisper backend) ----------
def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} (faster-whisper, int8) ...")
        _model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=2,   # threads per transcription call, so multiple chunks can run concurrently
        )
        print("Whisper model loaded.")
    return _model


def detect_language(chunk_path: str) -> str:
    """Runs faster-whisper's language detector on a chunk. Returns e.g. 'en', 'hi'."""
    model = load_model()
    _, info = model.transcribe(chunk_path, language=None)
    return info.language


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()
    segments, _ = model.transcribe(chunk_path, language="en", beam_size=1, vad_filter=True)
    return " ".join(seg.text for seg in segments).strip()


# ---------- Sarvam (API) ----------
def transcribe_chunk_sarvam(chunk_path: str, retries: int = 3) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    headers = {"api-subscription-key": SARVAM_API_KEY}
    size_mb = os.path.getsize(chunk_path) / 1e6
    print(f"  Uploading {os.path.basename(chunk_path)} ({size_mb:.2f} MB) ...")

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with open(chunk_path, "rb") as f:
                files = {"file": (os.path.basename(chunk_path), f, "audio/wav")}
                data = {
                    "model": SARVAM_MODEL,
                    "language_code": SARVAM_LANG,
                }
                response = requests.post(
                    SARVAM_STT_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=(10, 120),
                )

            if response.status_code != 200:
                print("=== SARVAM ERROR ===")
                print("STATUS :", response.status_code)
                print("BODY   :", response.text)
                print("FILE   :", chunk_path)
                print("MODEL  :", SARVAM_MODEL)
                print("URL    :", SARVAM_STT_URL)
                print("LANG   :", SARVAM_LANG)
                print("====================")

            response.raise_for_status()
            payload = response.json()
            text = payload.get("transcript")
            if text is None:
                print("Unexpected response shape:", payload)
                return ""
            return text

        except requests.exceptions.RequestException as e:
            last_exc = e
            print(f"  Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)

    raise last_exc


# ---------- Router ----------
def transcribe_chunk(chunk_path: str, language: str = "auto") -> str:
    """
    language:
      "auto"     -> detect this chunk's language, route English->Whisper, else->Sarvam
      "english"  -> force Whisper
      "hinglish" -> force Sarvam
    """
    lang = language.lower()

    if lang == "auto":
        detected = detect_language(chunk_path)
        print(f"  Detected language: {detected}")
        if detected == "en":
            return transcribe_chunk_whisper(chunk_path)
        return transcribe_chunk_sarvam(chunk_path)

    if lang == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


# ---------- Batch driver ----------
def transcribe_all(chunks: list, language: str = "auto") -> str:
    print(f"Language mode: {language}")
    results = [""] * len(chunks)

    def worker(i, chunk):
        print(f"Transcribing chunk {i + 1}/{len(chunks)} ...")
        try:
            return i, transcribe_chunk(chunk, language=language)
        except Exception as e:
            print(f"  Chunk {i + 1} failed after retries: {e}")
            return i, ""

    lang = language.lower()
    cpu_count = multiprocessing.cpu_count()

    if lang == "hinglish":
        max_workers = 4                        # network-bound, not limited by CPU cores
    elif lang == "english":
        max_workers = max(1, cpu_count // 2)   # CPU-bound: 2 threads/chunk, so cores/2 chunks at once
    else:  # "auto" — mixed, keep it conservative
        max_workers = max(1, cpu_count // 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, i, chunk) for i, chunk in enumerate(chunks)]
        for future in futures:
            i, text = future.result()
            results[i] = text

    print("Transcription completed.")
    return " ".join(results).strip()