"""Isolated system TTS worker used for WAV file generation, never microphone recording."""
import json
import sys
from pathlib import Path


def main():
    import pyttsx3
    payload = json.loads(sys.stdin.read())
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    engine.save_to_file(payload["text"], payload["path"])
    engine.runAndWait()
    path = Path(payload["path"])
    if not path.is_file() or path.stat().st_size < 44:
        raise RuntimeError("System TTS did not produce audio")


if __name__ == "__main__":
    main()
