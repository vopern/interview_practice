"""Text-to-speech for the interviewer's voice (OpenAI TTS API)."""

import hashlib

from openai import OpenAI

# Used when no round profile is available to derive a delivery from — see
# `src.core.prompts_common.tts_instructions`, which is what normally supplies it.
DEFAULT_INSTRUCTIONS = "Speak like a professional, friendly job interviewer."

# The 13 built-in voices of gpt-4o-mini-tts, mapped to the labels the picker
# shows: "alloy" and "verse" tell a candidate nothing about how they sound. The
# descriptions are characterisations from listening, not API-documented facts.
# These differ in timbre only — how the interviewer *delivers* the line is the
# `instructions` axis above, and is not the candidate's choice.
# `alloy` is first because the first entry is the default and the fallback.
VOICES = {
    "alloy": "Alloy — neutral, even-toned",
    "ash": "Ash — firm, businesslike",
    "ballad": "Ballad — soft, expressive",
    "cedar": "Cedar — natural, high fidelity (recommended)",
    "coral": "Coral — bright, warm",
    "echo": "Echo — calm, level",
    "fable": "Fable — animated, storytelling",
    "marin": "Marin — natural, high fidelity (recommended)",
    "nova": "Nova — clear, upbeat",
    "onyx": "Onyx — deep, authoritative",
    "sage": "Sage — gentle, measured",
    "shimmer": "Shimmer — light, airy",
    "verse": "Verse — conversational, natural",
}

DEFAULT_VOICE = next(iter(VOICES))


def resolve_voice(name: str) -> str:
    """A valid voice id for `name`, falling back rather than raising.

    A stale TTS_VOICE in someone's .env, or a name OpenAI has retired, should
    cost the chosen timbre — not the interview.
    """
    return name if name in VOICES else DEFAULT_VOICE


class TTSService:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini-tts",
        voice: str = DEFAULT_VOICE,
        instructions: str = DEFAULT_INSTRUCTIONS,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self._cache: dict[str, bytes] = {}

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        voice = voice or self.voice
        instructions = instructions or self.instructions
        # Voice and instructions are part of the key, not just the text: keyed
        # on the text alone, a repeated line comes back in whichever voice said
        # it first.
        key = hashlib.md5(f"{voice}|{instructions}|{text}".encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]
        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice,
            input=text,
            instructions=instructions,
        )
        audio = bytes(response.read())
        self._cache[key] = audio
        return audio
