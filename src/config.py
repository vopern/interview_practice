"""Application configuration, driven entirely by environment variables / .env."""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # The LLM runs via the Claude Agent SDK, which authenticates through the
    # local Claude Code login (or ANTHROPIC_API_KEY if exported) — no key is
    # read here.
    #
    # Two models, because the two kinds of call want opposite things. Live
    # interviewer turns sit inside the voice loop, so the user waits on every
    # one of them: fast tier, shallow effort. Planning and evaluation are
    # one-shot and judgment-heavy, and nobody is waiting mid-sentence: strong
    # tier, deep effort.
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    ANTHROPIC_EFFORT = os.getenv("ANTHROPIC_EFFORT", "low")

    ANTHROPIC_REASONING_MODEL = os.getenv("ANTHROPIC_REASONING_MODEL", "claude-opus-5")
    ANTHROPIC_REASONING_EFFORT = os.getenv("ANTHROPIC_REASONING_EFFORT", "high")

    # Voice (optional): without OPENAI_API_KEY the app runs text-only.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")
    TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
    TTS_VOICE = os.getenv("TTS_VOICE", "alloy")

    DATA_DIR = os.getenv("DATA_DIR", "data")

    # Pre-filled setup form values. Unset means an empty form field — the app
    # ships without a company or role baked in.
    DEFAULT_COMPANY = os.getenv("DEFAULT_COMPANY", "")
    DEFAULT_ROLE = os.getenv("DEFAULT_ROLE", "")
    DEFAULT_STAGE = os.getenv("DEFAULT_STAGE", "")

    @classmethod
    def voice_enabled(cls) -> bool:
        return bool(cls.OPENAI_API_KEY)
