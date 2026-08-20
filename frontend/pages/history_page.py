"""Browse past interview sessions, scores, and evaluations."""

import pandas as pd
import streamlit as st

from frontend.components import render_evaluation, render_transcript


def render_history_page() -> None:
    st.title("📈 Interview history")

    storage = st.session_state.storage
    slugs = storage.list_slugs()
    if not slugs:
        st.info("No stored sessions yet. Finish an interview and it will show up here.")
        return

    labels = {
        f"{settings.company} — {settings.role} ({settings.stage})": slug
        for slug, settings in slugs
    }
    choice = st.selectbox("Interview", list(labels))
    sessions = storage.load_all(labels[choice])
    if not sessions:
        st.info("No sessions found for this interview.")
        return

    st.dataframe(
        pd.DataFrame(
            {
                "Date": [s.created_at.replace("T", " ") for s in sessions],
                "Score": [
                    s.evaluation.score if s.evaluation else None for s in sessions
                ],
                "Questions answered": [
                    sum(1 for t in s.transcript if t.role == "candidate")
                    for s in sessions
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    session_labels = {
        f"{s.created_at.replace('T', ' ')}"
        + (f" — {s.evaluation.score}/100" if s.evaluation else " — no evaluation"): s
        for s in reversed(sessions)
    }
    selected = session_labels[st.selectbox("Session", list(session_labels))]

    if selected.evaluation:
        render_evaluation(selected.evaluation)
    else:
        st.warning("This session has no evaluation.")

    with st.expander("Full transcript"):
        render_transcript(selected.transcript)
