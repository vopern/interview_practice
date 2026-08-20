"""Research a company into data/context/<company>/, and read what is there.

The folder this page writes to is the same one the interview planner reads, so a
report saved here shows up in the next interview for that company automatically.
"""

from datetime import datetime
from pathlib import Path

import streamlit as st

from src.services.llm_service import LLMError

BRIEF_PLACEHOLDER = (
    "ABComp — Senior Data Scientist in New York.\n"
    "Focus on the recommendations org and any recent strategy news.\n"
    "Job ad: https://..."
)

BRIEF_HELP = (
    "One box for everything: the **company name** (required), the **role** you are "
    "aiming at, anything you want the research to **focus** on, and the **job posting** "
    "text or URL. The research is about the company only — it never looks at your CV "
    "and never scores your fit."
)


def _run_research(brief: str) -> None:
    try:
        # The label is the progress display: the researcher reports each search,
        # fetch and aside as it happens, and the callback runs on this script
        # thread, so the box repaints while the call is still blocking. The
        # headline only — a run is up to 60 turns, and a full log in the box
        # would push the rest of the page around while it is being read.
        with st.status("Researching — this takes a few minutes.") as box:
            result = st.session_state.researcher.research(
                brief, on_progress=lambda line: box.update(label=line)
            )
            box.update(label=f"Researched {result.company}", state="complete")
    except LLMError as e:
        st.error(f"Research failed: {e}")
        return
    st.session_state.research_notice = str(result.path)
    st.rerun()


def _render_run() -> None:
    with st.form("research"):
        brief = st.text_area(
            "What should be researched?",
            height=140,
            placeholder=BRIEF_PLACEHOLDER,
            help=BRIEF_HELP,
        )
        submitted = st.form_submit_button("Research company", type="primary")
    if submitted:
        if not brief.strip():
            st.warning("Name a company to research.")
        else:
            _run_research(brief)


def _file_caption(path: Path) -> str:
    stat = path.stat()
    when = datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y %H:%M")
    return f"{when} · {stat.st_size / 1024:.0f} kB"


def _render_reports() -> None:
    storage = st.session_state.storage
    companies = storage.list_context_companies()
    if not companies:
        st.info(
            "No context files yet. Research a company above, or drop your own notes "
            "into `data/context/<company>/`."
        )
        return

    company = st.selectbox("Company", companies)
    files = storage.list_context_files(company)
    # Newest first: storage returns them in the planner's (alphabetical) order,
    # but the file someone wants to read here is the one just researched.
    by_recency = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    labels = {f"{p.name}  ({_file_caption(p)})": p for p in by_recency}
    chosen = labels[st.selectbox("File", list(labels))]

    # Everything in this folder goes to the interview planner, so say so — a
    # stale report is not obviously stale until you know it is still being read.
    st.caption(
        f"All {len(files)} file(s) in this folder are given to the interview planner "
        f"when you start an interview with {company}. Delete the ones you no longer "
        "want it to see."
    )

    text = storage.read_context_file(chosen)
    if not text:
        st.warning("This file is empty or could not be read as text.")
    elif chosen.suffix.lower() in {".md", ".markdown"}:
        st.markdown(text)
    else:
        st.code(text, language=None)


def render_research_page() -> None:
    st.title("🔎 Company research")
    st.caption(
        "Researches a company and saves the briefing as interview context, so the "
        "interviewer knows what it is talking about."
    )

    notice = st.session_state.pop("research_notice", None)
    if notice:
        st.success(f"Saved to `{notice}`")

    _render_run()
    st.divider()
    st.subheader("Saved context")
    _render_reports()
