"""Unit tests for the interviewer's voice.

Offline like the rest of the suite: `TTSService.__init__` only constructs an
OpenAI client (no request), so a dummy key plus a stub on `.client` is enough.
"""

from src.services import tts_service
from src.services.tts_service import VOICES, TTSService, resolve_voice

# The built-in voices gpt-4o-mini-tts accepts. Hardcoded rather than derived
# from VOICES, so a typo in the catalogue fails the test instead of passing it.
API_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "fable",
    "marin",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
}


class _Response:
    def __init__(self, audio):
        self._audio = audio

    def read(self):
        return self._audio


class _Speech:
    def __init__(self):
        self.calls = []

    def create(self, model, voice, input, instructions):
        self.calls.append(
            {
                "model": model,
                "voice": voice,
                "input": input,
                "instructions": instructions,
            }
        )
        # Distinct payload per (voice, instructions, text), so a cache that
        # ignored any of them would show up as identical bytes.
        return _Response(f"{voice}|{instructions}|{input}".encode("utf-8"))


class _Client:
    def __init__(self):
        self.audio = type("Audio", (), {"speech": _Speech()})()


def _service():
    service = TTSService("test-key")
    service.client = _Client()
    return service


# --- the voice catalogue ---------------------------------------------------


def test_every_offered_voice_is_one_the_api_accepts():
    assert set(VOICES) <= API_VOICES
    assert all(label.strip() for label in VOICES.values())


def test_the_default_voice_is_the_historical_one():
    # Changing this silently re-voices every existing user's interviewer.
    assert tts_service.DEFAULT_VOICE == "alloy"


def test_an_unknown_voice_falls_back_instead_of_raising():
    assert resolve_voice("nonsense") == tts_service.DEFAULT_VOICE
    assert resolve_voice("onyx") == "onyx"


# --- synthesis and caching -------------------------------------------------


def test_voice_and_instructions_are_forwarded():
    service = _service()
    service.synthesize("Tell me about yourself.", "onyx", "Sound bored.")
    (call,) = service.client.audio.speech.calls
    assert call["voice"] == "onyx"
    assert call["instructions"] == "Sound bored."
    assert call["input"] == "Tell me about yourself."


def test_omitted_arguments_fall_back_to_the_instance_defaults():
    service = _service()
    service.synthesize("Hello.")
    (call,) = service.client.audio.speech.calls
    assert call["voice"] == tts_service.DEFAULT_VOICE
    assert call["instructions"] == tts_service.DEFAULT_INSTRUCTIONS


def test_the_same_line_in_the_same_voice_is_cached():
    service = _service()
    first = service.synthesize("Why us?", "sage", "Sound calm.")
    second = service.synthesize("Why us?", "sage", "Sound calm.")
    assert first == second
    assert len(service.client.audio.speech.calls) == 1


def test_switching_voice_or_delivery_is_not_served_from_the_cache():
    """The cache keyed on the text alone would replay the old voice."""
    service = _service()
    base = service.synthesize("Why us?", "sage", "Sound calm.")
    other_voice = service.synthesize("Why us?", "onyx", "Sound calm.")
    other_delivery = service.synthesize("Why us?", "sage", "Sound impatient.")
    assert base != other_voice
    assert base != other_delivery
    assert len(service.client.audio.speech.calls) == 3
