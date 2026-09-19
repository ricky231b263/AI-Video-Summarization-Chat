import yt_dlp
from pydub import AudioSegment
import os
import time

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

COOKIES_FILE = "cookies.txt"

# Different client/strategy combos to try in order if one gets blocked.
# YouTube's blocking behavior varies per client, so trying a few in
# sequence meaningfully improves reliability over using just one.
_CLIENT_STRATEGIES = [
    ["android", "web"],
    ["ios", "web"],
    ["web"],
    ["tv"],
]


def _build_ydl_opts(output_path: str, player_clients: list) -> dict:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
        ],
        "quiet": True,
        "extractor_args": {
            "youtube": {"player_client": player_clients}
        },
    }
    if os.path.exists(COOKIES_FILE):
        opts["cookiefile"] = COOKIES_FILE
    return opts


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    last_exc = None
    for i, clients in enumerate(_CLIENT_STRATEGIES, start=1):
        try:
            print(f"  Attempting download (strategy {i}/{len(_CLIENT_STRATEGIES)}: {clients}) ...")
            ydl_opts = _build_ydl_opts(output_path, clients)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                base, _ = os.path.splitext(ydl.prepare_filename(info))
                return base + ".wav"
        except Exception as e:
            last_exc = e
            print(f"  Strategy {i} failed: {e}")
            time.sleep(2)

    raise RuntimeError(
        f"All download strategies failed for this video. Last error: {last_exc}\n"
        f"This usually means YouTube is currently blocking requests from this "
        f"server's IP, or the video has restrictions (age/region/login-required). "
        f"Try again in a few minutes, or add a cookies.txt file for authenticated access."
    ) from last_exc


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video files to Wav format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "__converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_seconds: int = 30) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_seconds * 1000
    base = os.path.splitext(wav_path)[0]

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk_path = f"{base}_chunk_{i:03d}.wav"
        audio[start:start + chunk_ms].export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def process_input(source: str) -> tuple:
    if source.startswith(("http://", "https://")):
        print("Detected YouTube URL. Downloading audio...")
        raw_path = download_youtube_audio(source)
    else:
        print("Detected local file...")
        raw_path = source

    print("Normalizing to 16kHz mono...")
    wav_path = convert_to_wav(raw_path)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks, wav_path, raw_path


def cleanup_files(chunks: list, wav_path: str, raw_path: str = None, is_url_source: bool = True):
    """Deletes intermediate audio files created for this run.
    If is_url_source is False, raw_path is a user-provided local file and is kept."""
    files_to_delete = set(chunks)
    files_to_delete.add(wav_path)
    if raw_path and raw_path != wav_path and is_url_source:
        files_to_delete.add(raw_path)

    for f in files_to_delete:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception as e:
            print(f"  Could not delete {f}: {e}")

    print(f"Cleaned up {len(files_to_delete)} temporary audio file(s).")
