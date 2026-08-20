"""Business logic: researches a company and files the report as interview context.

Frontend-agnostic — no streamlit imports, same as `interview_manager`. A sibling
of `InterviewManager` rather than a fifth method on it: that class exposes
exactly four methods so a future HTTP API can wrap them unchanged, and research
is a different lifecycle — minutes long, tool-driven, and producing a file
rather than a session.

What it produces lands in `data/context/<company>/`, which means the next
interview for that company picks it up with no further wiring: the planner
already concatenates that folder into its prompt.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src.core import prompts_preparation
from src.services.llm_service import LLMError, LLMService, ProgressEvent
from src.storage.interview_storage import InterviewStorage

# A research run is minutes of searching and reading, so it says what it is
# doing as it goes. The wording lives here rather than in `llm_service` (which
# stays generic) or in the page (which should not have to know what a tool call
# looks like), and so does the clip that keeps a headline to one line.
HEADLINE_MAX = 90


@dataclass
class ResearchResult:
    """A finished research run. Not persisted as JSON — `path` is the artifact."""

    company: str
    role: str
    focus: str
    path: Path
    markdown: str


def _clip(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= HEADLINE_MAX:
        return text
    return text[: HEADLINE_MAX - 1].rstrip() + "…"


def _headline(event: ProgressEvent) -> Optional[str]:
    """Phrase one progress event as a single line, or `None` to stay silent.

    Never returns an empty string: a blank headline would clear whatever the UI
    is showing and read as the run having stalled.
    """
    if event.kind == "tool":
        if event.name == "WebSearch" and event.detail:
            return _clip(f'Searching the web: "{event.detail}"')
        if event.name == "WebFetch" and event.detail:
            # Scheme-less, so more of the actual page survives the clip.
            return _clip(f"Reading {event.detail.split('://', 1)[-1].rstrip('/')}")
        if not event.name:
            return None
        return _clip(f"{event.name}: {event.detail}" if event.detail else event.name)
    if event.kind == "text":
        # The model's aside between two tool calls. Only its first line is a
        # headline; the rest is usually the beginning of the report itself.
        first = event.detail.strip().splitlines()[0].strip(" #*_-")
        return _clip(first) if first else None
    return None


def _emit(on_progress: Optional[Callable[[str], None]], line: Optional[str]) -> None:
    if on_progress is not None and line:
        on_progress(line)


class CompanyResearcher:
    def __init__(self, llm: LLMService, storage: InterviewStorage):
        self.llm = llm
        self.storage = storage

    def research(
        self,
        brief: str,
        now: Optional[datetime] = None,
        *,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> ResearchResult:
        """Research the company described in a free-text brief and save the report.

        Two model calls, in this order and for this reason: the UI collects a
        single text box, but the report has to be filed under a company folder,
        so the company name has to be resolved out of the prose before anything
        is written. The first call is cheap, tool-less and one-shot; it also
        fails fast, so an unusable brief costs seconds rather than a full
        research run.

        The second call is the research itself, and returns Markdown rather than
        JSON — a long tool loop with a schema stapled on top is the fragile
        combination.

        `on_progress` is called with a one-line description of each step as it
        starts: the two stages here, and every search, fetch and aside inside
        the research call.
        """
        if not brief.strip():
            raise LLMError("Describe the company to research first.")

        now = now or datetime.now()
        _emit(on_progress, "Reading the request…")
        parsed = self.llm.complete_json(
            "You extract structured fields from a short request. Be literal.",
            [
                {
                    "role": "user",
                    "content": prompts_preparation.get_research_brief_prompt(brief),
                }
            ],
            prompts_preparation.COMPANY_RESEARCH_BRIEF_SCHEMA,
        )
        company = (parsed.get("company") or "").strip()
        role = (parsed.get("role") or "").strip()
        focus = (parsed.get("focus") or "").strip()
        if not company:
            raise LLMError(
                "No company name found in that request. Start the text with the "
                "company you want researched."
            )

        target = f"{company} — {role}" if role else company
        _emit(on_progress, f"Researching {target}…")
        markdown = self.llm.complete_research(
            prompts_preparation.COMPANY_RESEARCH_SYSTEM,
            prompts_preparation.get_company_research_prompt(
                company, role, focus, now=now
            ),
            on_event=lambda event: _emit(on_progress, _headline(event)),
        )

        # Timestamped, never overwritten: a report is research the candidate may
        # have annotated, and the folder is also where their own notes live.
        _emit(on_progress, "Saving the report…")
        filename = f"research_{now.strftime('%Y%m%d_%H%M%S')}.md"
        path = self.storage.save_company_context(company, filename, markdown)
        return ResearchResult(
            company=company, role=role, focus=focus, path=path, markdown=markdown
        )
