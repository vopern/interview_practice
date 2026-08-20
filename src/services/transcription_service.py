"""Speech-to-text for voice answers (OpenAI transcription API)."""

import io

from openai import OpenAI


class TranscriptionService:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini-transcribe"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def transcribe(
        self, audio_bytes: bytes, filename: str = "answer.wav", prompt: str = ""
    ) -> str:
        """Transcribe an answer, optionally primed with expected vocabulary.

        `prompt` biases the recogniser towards names it would otherwise mangle —
        the company, the role, and the jargon the interviewer just used. Without
        it, proper nouns come back wrong and the evaluator reads them as
        candidate mistakes.
        """
        extra = {"prompt": prompt} if prompt else {}
        response = self.client.audio.transcriptions.create(
            model=self.model,
            file=(filename, io.BytesIO(audio_bytes)),
            **extra,
        )
        return response.text.strip()
