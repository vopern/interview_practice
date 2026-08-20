"""Entry point and composition root for the Streamlit app."""

import sys
from pathlib import Path

# Make `src.*` imports resolve when run as `streamlit run frontend/app.py`.
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from frontend.constants import APP_TITLE
from frontend.pages.history_page import render_history_page
from frontend.pages.interview_page import render_interview_page
from frontend.pages.research_page import render_research_page
from src.config import Config
from src.core.company_researcher import CompanyResearcher
from src.core.interview_manager import InterviewManager
from src.services.llm_service import LLMService
from src.services.transcription_service import TranscriptionService
from src.services.tts_service import TTSService
from src.storage.interview_storage import InterviewStorage


def initialize_services() -> None:
    """Construct all services once and cache them in session state."""
    if "manager" in st.session_state:
        return
    storage = InterviewStorage(Config.DATA_DIR)
    llm = LLMService(
        Config.ANTHROPIC_MODEL,
        Config.ANTHROPIC_REASONING_MODEL,
        Config.ANTHROPIC_EFFORT,
        Config.ANTHROPIC_REASONING_EFFORT,
    )
    st.session_state.storage = storage
    st.session_state.manager = InterviewManager(llm, storage)
    st.session_state.researcher = CompanyResearcher(llm, storage)
    if Config.voice_enabled():
        st.session_state.transcription = TranscriptionService(
            Config.OPENAI_API_KEY, Config.STT_MODEL
        )
        st.session_state.tts = TTSService(
            Config.OPENAI_API_KEY, Config.TTS_MODEL, Config.TTS_VOICE
        )
    else:
        st.session_state.transcription = None
        st.session_state.tts = None


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎤", layout="wide")

    initialize_services()

    pages = [
        st.Page(render_interview_page, title="Interview", icon="🎤", default=True),
        st.Page(render_history_page, title="History", icon="📈"),
        st.Page(render_research_page, title="Research", icon="🔎"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
