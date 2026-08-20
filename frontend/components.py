"""Rendering helpers shared between the interview and history pages."""

import streamlit as st

from frontend.constants import CANDIDATE_AVATAR, INTERVIEWER_AVATAR
from src.models.interview import Evaluation, Turn

RATING_LABELS = {
    "met": "✅ Met",
    "partial": "🟡 Partial",
    "not_met": "❌ Not met",
    "not_assessed": "⚪ Not assessed",
}

ADDRESSED_LABELS = {
    "full": "✅ Answered",
    "partial": "🟡 Partly answered",
    "avoided": "❌ Not answered",
}

CANDIDATE_QUESTION_LABELS = {
    "strong": "✅ Strong",
    "reasonable": "🟡 Reasonable",
    "weak": "❌ Weak",
    "red_flag": "🚩 Red flag",
}

CANDIDATE_QUESTIONS_RATING_LABELS = {
    "strong": "✅ Strong",
    "adequate": "🟡 Adequate",
    "weak": "❌ Weak",
}

# How loudly an interviewer would raise the one takeaway in the debrief.
TAKEAWAY_SEVERITY_LABELS = {
    1: "🟡 Light concern",
    2: "🟠 Would come up in the debrief",
    3: "🚩 Red flag — the reason not to hire",
}


def render_transcript(transcript: list[Turn]) -> None:
    for turn in transcript:
        if turn.role == "interviewer":
            with st.chat_message("assistant", avatar=INTERVIEWER_AVATAR):
                st.markdown(turn.content)
        else:
            with st.chat_message("user", avatar=CANDIDATE_AVATAR):
                st.markdown(turn.content)


def render_candidate_questions(evaluation: Evaluation) -> None:
    """The read on the questions the candidate asked at the end.

    Absent for sessions stored before this was evaluated, so the whole section
    is skipped rather than rendered empty.
    """
    review = evaluation.candidate_questions
    if not review:
        return

    st.subheader("Your questions")
    if review.rating == "not_assessed":
        st.caption(
            "Not assessed — the interviewer never really opened the floor for your "
            "questions."
        )
        if review.impression:
            st.caption(review.impression)
        return

    label = CANDIDATE_QUESTIONS_RATING_LABELS.get(review.rating, review.rating)
    st.caption(f"{label} — what these questions said about you")
    if review.impression:
        st.markdown(review.impression)

    for question in review.questions:
        signal = CANDIDATE_QUESTION_LABELS.get(question.signal, question.signal)
        st.markdown(f"{signal} — **{question.question}**")
        if question.comment:
            st.caption(question.comment)

    if review.better_questions:
        st.markdown("**Ask this next time**")
        for better in review.better_questions:
            st.markdown(f"- {better}")


def render_key_takeaway(evaluation: Evaluation) -> None:
    """The one thing to fix, given the weight of a callout rather than a bullet.

    Absent for sessions stored before this was evaluated, so the whole block is
    skipped rather than rendered empty.
    """
    takeaway = evaluation.key_takeaway
    if not takeaway:
        return

    box = {1: st.info, 2: st.warning, 3: st.error}.get(takeaway.severity, st.warning)
    label = TAKEAWAY_SEVERITY_LABELS.get(takeaway.severity, "")
    lines = [f"**Take this into the next interview: {takeaway.point}**", label]
    if takeaway.verdict:
        lines.append(takeaway.verdict)
    if takeaway.evidence:
        lines.append(f"*Where it showed:* {takeaway.evidence}")
    box("\n\n".join(line for line in lines if line))


def render_evaluation(evaluation: Evaluation) -> None:
    col_score, col_summary = st.columns([0.2, 0.8])
    with col_score:
        st.metric("Score", f"{evaluation.score}/100")
    with col_summary:
        st.markdown(evaluation.summary)

    render_key_takeaway(evaluation)

    if evaluation.progress_notes:
        st.info(f"**Progress since previous rounds:** {evaluation.progress_notes}")

    st.subheader("Interviewer checklist")
    for result in evaluation.results:
        label = RATING_LABELS.get(result.rating, result.rating)
        with st.expander(f"{label} — {result.criterion}"):
            if result.evidence:
                st.markdown(f"**Evidence:** {result.evidence}")
            if result.comment:
                st.markdown(f"**Comment:** {result.comment}")

    if evaluation.answer_review:
        st.subheader("Question by question")
        missed = sum(1 for a in evaluation.answer_review if a.addressed != "full")
        st.caption(
            f"{len(evaluation.answer_review)} questions asked"
            + (f" — {missed} not fully answered" if missed else " — all fully answered")
        )
        for answer in evaluation.answer_review:
            label = ADDRESSED_LABELS.get(answer.addressed, answer.addressed)
            st.markdown(f"{label} — **{answer.question}**")
            if answer.comment:
                st.caption(answer.comment)

    render_candidate_questions(evaluation)

    col_strengths, col_improvements = st.columns(2)
    with col_strengths:
        st.subheader("Strengths")
        for s in evaluation.strengths:
            st.markdown(f"- {s}")
    with col_improvements:
        st.subheader("Work on this")
        for i in evaluation.improvements:
            st.markdown(f"- {i}")
