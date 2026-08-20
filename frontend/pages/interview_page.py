"""Main page: setup form -> live interview -> evaluation."""

import hashlib
import os

import streamlit as st

from frontend.components import render_evaluation, render_transcript
from src.config import Config
from src.core import prompts_common, prompts_interview
from src.models.interview import CANDIDATE, STAGES, TEXT, VOICE, InterviewSettings
from src.services.llm_service import LLMError
from src.services.tts_service import VOICES, resolve_voice

MODE_NEW = "🆕 New interview"
MODE_REPLAY = "🔁 Replay a previous round"


def _init_state() -> None:
    defaults = {
        "phase": "setup",
        "session": None,
        "previous_rounds": 0,
        "interview_complete": False,
        "last_audio_digest": None,
        "pending_tts": None,
        "voice_output": True,
        # Picked on the setup form and fixed for the round.
        "tts_voice": resolve_voice(Config.TTS_VOICE),
        "answer_attempt": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset(phase: str = "setup") -> None:
    st.session_state.phase = phase
    st.session_state.session = None
    st.session_state.interview_complete = False
    st.session_state.last_audio_digest = None
    st.session_state.pending_tts = None
    # Fresh recorder keys for the next session, so its first turn cannot restore a
    # recording left behind by this one.
    st.session_state.answer_attempt += 1


def _start_interview(
    settings: InterviewSettings,
    use_history: bool = True,
    replay_of=None,
) -> None:
    # A replay skips the planning call entirely — nothing to wait on but the
    # interviewer's first line.
    spinner = (
        "Reopening the prepared interview..."
        if replay_of is not None
        else "Preparing your interview (checklist, questions)..."
    )
    try:
        with st.spinner(spinner):
            session, previous_rounds = st.session_state.manager.start(
                settings, use_history, replay_of
            )
    except LLMError as e:
        st.error(f"Could not start the interview: {e}")
        return
    st.session_state.session = session
    st.session_state.previous_rounds = previous_rounds
    st.session_state.phase = "interview"
    st.session_state.interview_complete = False
    st.session_state.last_audio_digest = None
    st.session_state.pending_tts = session.transcript[0].content
    st.rerun()


def _handle_answer(text: str, modality: str = TEXT) -> None:
    try:
        with st.spinner("Interviewer is thinking..."):
            reply, complete = st.session_state.manager.reply(
                st.session_state.session, text, modality
            )
    except LLMError as e:
        st.error(f"Something went wrong — your answer was not lost, try again: {e}")
        return
    if complete:
        st.session_state.interview_complete = True
    st.session_state.pending_tts = reply
    st.rerun()


def _undo_last_answer() -> None:
    """Take back the last answer so the candidate can give it again."""
    if not st.session_state.manager.undo_last_answer(st.session_state.session):
        return
    # Set from the reply just deleted, and while it is true the answer input is
    # never rendered — the candidate would have nothing to redo with.
    st.session_state.interview_complete = False
    st.session_state.last_audio_digest = None
    st.session_state.answer_attempt += 1
    st.session_state.pending_tts = None  # don't re-speak the deleted reply
    st.rerun()


def _finish_interview() -> None:
    try:
        with st.spinner("The interviewer is filling out the evaluation..."):
            st.session_state.manager.finish(st.session_state.session)
    except LLMError as e:
        st.error(f"Evaluation failed, try again: {e}")
        return
    st.session_state.phase = "evaluation"
    st.session_state.pending_tts = None
    st.rerun()


def _play_pending_tts() -> None:
    """Speak the latest interviewer message once, if voice output is on."""
    # Always claim this element slot, even on runs with nothing to play: an
    # element that disappears shifts everything after it (the audio recorder in
    # particular), which remounts it mid-run and breaks st.audio_input ("An
    # error has occurred, please try again").
    slot = st.container()
    text = st.session_state.pending_tts
    st.session_state.pending_tts = None
    if not text or not st.session_state.voice_output or st.session_state.tts is None:
        return
    session = st.session_state.session
    try:
        audio = st.session_state.tts.synthesize(
            text,
            resolve_voice(st.session_state.tts_voice),
            # The timbre is the candidate's pick; the delivery comes from the
            # persona, pressure and mood drawn for this round.
            prompts_common.tts_instructions(session.round_profile if session else {}),
        )
    except Exception:
        return  # voice is a nice-to-have; never block the interview on it
    # Random byte changes the payload hash so Streamlit re-autoplays repeated text.
    with slot:
        st.audio(audio + os.urandom(1), format="audio/mp3", autoplay=True)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def _render_setup() -> None:
    st.title("🎤 Interview Trainer")
    st.markdown(
        "Set up a practice interview. An AI interviewer will question you, probe your "
        "answers, and give you a score, a filled-out interviewer checklist, and feedback "
        "at the end. Repeat with the same settings and it will avoid old questions and "
        "check your progress."
    )
    if not Config.voice_enabled():
        st.caption(
            "Voice input/output is disabled (no OPENAI_API_KEY set) — text only."
        )

    # The mode switch has to sit outside the form below: widgets inside a form do
    # not rerun until it is submitted, so a mode chosen in there could not swap
    # the fields underneath it.
    mode = MODE_NEW
    if st.session_state.storage.list_slugs():
        mode = st.radio(
            "How do you want to start?",
            [MODE_NEW, MODE_REPLAY],
            horizontal=True,
            label_visibility="collapsed",
        )
    if mode == MODE_REPLAY:
        _render_replay_setup()
    else:
        _render_new_setup()


def _voice_picker():
    """The timbre picker, or None when voice is off. Shared by both setup modes."""
    if not Config.voice_enabled():
        return None
    return st.selectbox(
        "🔊 Interviewer voice",
        list(VOICES),
        index=list(VOICES).index(st.session_state.tts_voice),
        format_func=VOICES.get,
        help=(
            "The timbre only. How the interviewer delivers it — pace, warmth, "
            "how much pressure you hear — follows the persona and mood drawn "
            "for the round, and changes between rounds."
        ),
    )


def _render_replay_setup() -> None:
    """Re-enter a stored round: same plan, same interviewer, a fresh conversation.

    Not an `st.form`: the round list depends on the interview picked above it,
    and a form would not refresh it until submit.
    """
    storage = st.session_state.storage
    st.markdown(
        "Run a stored round again. Its checklist, planned questions, interviewer, "
        "persona and mood are reused exactly as they were — nothing is "
        "planned afresh — so you can answer the same interview better. Where it goes "
        "from there is up to your answers."
    )
    labels = {
        f"{settings.company} — {settings.role} ({settings.stage})": slug
        for slug, settings in storage.list_slugs()
    }
    sessions = storage.load_all(labels[st.selectbox("Interview", list(labels))])
    if not sessions:
        st.info("No stored rounds for this interview.")
        return
    round_labels = {
        f"{s.created_at.replace('T', ' ')}"
        + (f" — {s.evaluation.score}/100" if s.evaluation else " — no evaluation"): s
        for s in reversed(sessions)
    }
    source = round_labels[st.selectbox("Round to replay", list(round_labels))]

    with st.expander("What gets reused"):
        profile = source.round_profile
        mood = profile.get("mood") or []
        st.markdown(
            f"**Interviewer:** {profile.get('archetype', 'unknown')} · pressure "
            f"{profile.get('pressure', '?')}/3"
            + (f" · feeling {' and '.join(mood)}" if mood else "")
        )
        st.markdown(f"**Length:** about {source.settings.duration_minutes} minutes")
        if source.plan.checklist:
            st.markdown(
                "**Checklist:** "
                + ", ".join(c.criterion for c in source.plan.checklist)
            )
        if source.plan.interviewer_profession:
            with st.expander("Who ran this round"):
                st.markdown(source.plan.interviewer_profession)
        if source.plan.round_purpose:
            with st.expander("What this round was for"):
                st.markdown(source.plan.round_purpose)
        if source.plan.interviewer_role:
            st.markdown("**What this interviewer had to establish**")
            st.markdown(source.plan.interviewer_role)
        if source.plan.questions:
            st.markdown("**Planned questions**")
            for i, question in enumerate(source.plan.questions, 1):
                st.markdown(f"{i}. {question}")
            st.caption(
                "Roughly the order the round runs in — the interviewer keeps to it "
                "loosely, presses where an answer opens something up, and skips ahead "
                "when time runs short."
            )
        st.caption(
            "The opening line is spoken fresh, so it will be worded differently — "
            "everything above comes from the stored round."
        )

    voice = _voice_picker()
    if st.button("▶️ Replay this round", type="primary"):
        if voice:
            st.session_state.tts_voice = voice
        _start_interview(source.settings, replay_of=source)


def _render_new_setup() -> None:
    with st.form("setup"):
        col1, col2, col3 = st.columns(3)
        with col1:
            # The companies with research files are the ones worth offering: a
            # typo here costs the round its context silently, since the lookup
            # in `start` just comes back empty.
            companies = st.session_state.storage.list_context_companies()
            # DEFAULT_COMPANY may name a company with no context folder yet —
            # offer it rather than dropping it, the same way the stage box below
            # treats an unknown DEFAULT_STAGE.
            options = companies + (
                [Config.DEFAULT_COMPANY]
                if Config.DEFAULT_COMPANY and Config.DEFAULT_COMPANY not in companies
                else []
            )
            company = st.selectbox(
                "Company",
                options,
                index=(
                    options.index(Config.DEFAULT_COMPANY)
                    if Config.DEFAULT_COMPANY in options
                    else None
                ),
                accept_new_options=True,
                placeholder="Pick or type a company",
                help=(
                    "The companies you have files for in `data/context/`. Any other "
                    "name works too — that round just runs without research context."
                ),
            )
        with col2:
            role = st.text_input(
                "Role",
                value=Config.DEFAULT_ROLE,
                placeholder="e.g. Senior Data Scientist",
            )
        with col3:
            # DEFAULT_STAGE unset (or naming a stage we don't offer) leaves the
            # box empty rather than silently picking one for the candidate.
            stage_index = (
                STAGES.index(Config.DEFAULT_STAGE)
                if Config.DEFAULT_STAGE in STAGES
                else None
            )
            stage = st.selectbox(
                "Interview stage",
                STAGES,
                index=stage_index,
                placeholder="Choose a stage",
            )
        duration = st.select_slider(
            "Interview length",
            options=[10, 15, 20, 30, 45, 60],
            value=30,
            format_func=lambda m: f"{m} min",
            help=(
                "Sets the scope of the interview: fewer, more focused questions for a "
                "short session; broader and deeper coverage for a long one."
            ),
        )
        background = st.text_area(
            "Background information (optional)",
            placeholder=(
                "Paste anything the interviewer should tailor the interview to: "
                "your CV highlights, the job ad, topics you want to practice..."
            ),
            height=160,
            help=(
                "Files dropped in `data/context/<company>/` (company research, recruiter "
                "notes...) are picked up automatically and appended to this field."
            ),
        )
        without_history = st.checkbox(
            "Start without history",
            help=(
                "Plan this round as if you had never practiced this interview before: "
                "no carried-over checklist, no avoided questions or scenarios, no "
                "pressure escalation, and no progress notes in the feedback. Research "
                "files in `data/context/<company>/` are still used — they are your "
                "preparation, not a record of past rounds."
            ),
        )
        # The only place the voice is chosen: it is fixed for the round, so the
        # interviewer cannot change voice halfway through an interview.
        voice = _voice_picker()
        submitted = st.form_submit_button("Start interview", type="primary")

    if submitted:
        # The company box returns None until something is picked or typed.
        company = (company or "").strip()
        if not company or not role.strip() or not stage:
            st.warning("Please fill in company, role and interview stage.")
            return
        if voice:
            # Before starting: `_start_interview` queues the opening line for
            # playback, which reads this on the rerun that follows.
            st.session_state.tts_voice = voice
        _start_interview(
            InterviewSettings(
                company=company,
                role=role.strip(),
                stage=stage,
                background=background.strip(),
                duration_minutes=duration,
            ),
            use_history=not without_history,
        )


def _round_caption(session) -> str:
    """Which stored rounds this one is running against, in one line."""
    if session.replay_of:
        # Session ids are "%Y%m%d_%H%M%S", so the date is the first eight chars.
        source_id = session.replay_of
        date = (
            f"{source_id[:4]}-{source_id[4:6]}-{source_id[6:8]}"
            if len(source_id) >= 8
            else source_id
        )
        return (
            f"Replay of the round from {date} — same checklist, planned questions "
            "and interviewer."
        )
    if not session.use_history:
        return "Running without history — previous rounds are ignored this round."
    if st.session_state.previous_rounds:
        return (
            f"Round {st.session_state.previous_rounds + 1} — previous questions and "
            "feedback are being taken into account."
        )
    return ""


def _render_sidebar() -> None:
    session = st.session_state.session
    with st.sidebar:
        st.subheader("Session")
        st.markdown(
            f"**{session.settings.company}** — {session.settings.role}\n\n"
            f"Stage: {session.settings.stage} · ~{session.settings.duration_minutes} min"
        )
        if caption := _round_caption(session):
            st.caption(caption)
        if st.session_state.tts is not None:
            st.session_state.voice_output = st.toggle(
                "🔊 Interviewer voice", value=st.session_state.voice_output
            )
            # Read-only: the voice is fixed for the round, chosen at setup.
            st.caption(VOICES[st.session_state.tts_voice])
        st.divider()
        answered = any(t.role == CANDIDATE for t in session.transcript)
        if st.button(
            "🏁 End interview & get evaluation",
            disabled=not answered,
            help="Give at least one answer first." if not answered else None,
            use_container_width=True,
        ):
            _finish_interview()
        if st.button(
            "↩️ Redo my last answer",
            disabled=not answered,
            help=(
                "Deletes your last answer and the interviewer's reply."
                if answered
                else "You haven't answered anything yet."
            ),
            use_container_width=True,
        ):
            _undo_last_answer()
        if st.button("🗑️ Abandon session (not saved)", use_container_width=True):
            _reset()
            st.rerun()


def _render_interview() -> None:
    session = st.session_state.session
    _render_sidebar()
    st.title(f"Interview — {session.settings.company}")

    render_transcript(session.transcript)
    _play_pending_tts()

    if st.session_state.interview_complete:
        st.success("The interviewer has everything they need.")
        if st.button("🏁 Get my evaluation", type="primary"):
            _finish_interview()
        return

    # Voice answer
    if st.session_state.transcription is not None:
        # Key by turn: a fresh widget per question, so the recorder never
        # restores the previous answer's stale recording after a rerun. The
        # attempt counter is part of the key because undoing an answer shortens
        # the transcript back to a length already used this session, and the
        # widget would then restore the recording just deleted and resubmit it.
        audio = st.audio_input(
            "🎙️ Record your answer (or type below)",
            key=f"answer_audio_{len(session.transcript)}_{st.session_state.answer_attempt}",
        )
        if audio is not None:
            digest = hashlib.md5(audio.getvalue()).hexdigest()
            if digest != st.session_state.last_audio_digest:
                st.session_state.last_audio_digest = digest
                try:
                    with st.spinner("Transcribing..."):
                        text = st.session_state.transcription.transcribe(
                            audio.getvalue(),
                            # Prime the recogniser with the company, role and
                            # what was just asked, so proper nouns survive.
                            prompt=prompts_interview.transcription_hint(
                                session.settings, session.transcript
                            ),
                        )
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    text = ""
                if text:
                    _handle_answer(text, VOICE)

    # Text answer
    if user_input := st.chat_input("Type your answer..."):
        _handle_answer(user_input)


def _render_evaluation() -> None:
    session = st.session_state.session
    st.title("Your evaluation")
    st.caption(
        f"{session.settings.company} — {session.settings.role} ({session.settings.stage}). "
        "This session has been saved and will inform your next practice round."
    )
    render_evaluation(session.evaluation)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "🔁 Practice again (same settings)",
            type="primary",
            use_container_width=True,
        ):
            settings = session.settings
            _reset()
            _start_interview(settings)
    with col2:
        if st.button("🆕 New interview", use_container_width=True):
            _reset()
            st.rerun()

    with st.expander("Full transcript"):
        render_transcript(session.transcript)


def render_interview_page() -> None:
    _init_state()
    phase = st.session_state.phase
    if phase == "interview":
        _render_interview()
    elif phase == "evaluation":
        _render_evaluation()
    else:
        _render_setup()
