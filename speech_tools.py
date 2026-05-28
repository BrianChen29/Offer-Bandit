import argparse
import time
from pathlib import Path


DEFAULT_TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DCA"
DEFAULT_STT_MODEL = "base"


def synthesize_speech(text: str, output_path: str, model_name: str = DEFAULT_TTS_MODEL) -> Path:
    """Generate an audio file from text using Coqui TTS."""
    try:
        from TTS.api import TTS
    except ImportError as exc:
        raise RuntimeError(
            "TTS is not installed. Install optional audio dependencies with "
            "`pip install -r requirements-audio.txt`."
        ) from exc

    if not text.strip():
        raise ValueError("Text cannot be empty.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
    tts.tts_to_file(text=text, file_path=str(output))
    return output


def transcribe_audio(audio_file: str, model_name: str = DEFAULT_STT_MODEL) -> str:
    """Transcribe an audio file using Whisper."""
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Whisper is not installed. Install optional audio dependencies with "
            "`pip install -r requirements-audio.txt`."
        ) from exc

    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    return result["text"].strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optional STT/TTS utilities for Offer Bandit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tts_parser = subparsers.add_parser("tts", help="Synthesize speech from text.")
    tts_parser.add_argument("--text", required=True, help="Text to synthesize.")
    tts_parser.add_argument("--output", default="output.wav", help="Output audio path.")
    tts_parser.add_argument("--model", default=DEFAULT_TTS_MODEL, help="Coqui TTS model name.")

    stt_parser = subparsers.add_parser("stt", help="Transcribe an audio file.")
    stt_parser.add_argument("--audio", required=True, help="Audio file path.")
    stt_parser.add_argument("--model", default=DEFAULT_STT_MODEL, help="Whisper model name.")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    started_at = time.time()

    if args.command == "tts":
        output = synthesize_speech(args.text, args.output, args.model)
        print(f"Speech synthesis complete: {output}")
    elif args.command == "stt":
        transcript = transcribe_audio(args.audio, args.model)
        print(transcript)

    print(f"Process time: {time.time() - started_at:.2f} seconds")


if __name__ == "__main__":
    main()
