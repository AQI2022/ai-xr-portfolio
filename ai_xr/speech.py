import threading


class Transcriber:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()

    def transcribe(self, audio_file):
        from faster_whisper import WhisperModel
        with self.lock:
            if self.model is None:
                self.model = WhisperModel("tiny", device="cpu", compute_type="int8", download_root="models/whisper")
            segments, info = self.model.transcribe(str(audio_file), vad_filter=True)
            text = " ".join(segment.text for segment in segments)
        return {"text": text.strip(), "language": info.language, "engine": "faster-whisper-tiny"}
