from faster_whisper import WhisperModel
from helpers import Settings
import os
import tempfile


class WhisperLoader:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None

    def _get_model(self):
        if self.model is None:
            self.model = WhisperModel(
                self.settings.MODEL_TYPE,
                device=self.settings.DEVICE,
                compute_type=self.settings.COMPUTE_TYPE
            )
        return self.model

    def transcribe(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        model = self._get_model()

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segments, _ = model.transcribe(tmp_path)
            return " ".join(segment.text for segment in segments)
        finally:
            os.remove(tmp_path)