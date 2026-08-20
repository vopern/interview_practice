"""CompanyResearcher tests against a stubbed LLM (no network, no web search)."""

from datetime import datetime

import pytest

from src.core import prompts_preparation
from src.core.company_researcher import CompanyResearcher
from src.services.llm_service import LLMError, ProgressEvent
from src.storage.interview_storage import InterviewStorage

BRIEF_DATA = {
    "company": "Acme",
    "role": "Machine Learning Engineer",
    "focus": "Their rocket telemetry team and the last funding round.",
}

REPORT = "# Acme — Company Briefing\n\n## At a glance\n- They build rockets."


EVENTS = [
    ProgressEvent("text", "", "Let me start with their own site.\n\nThen the news."),
    ProgressEvent("tool", "WebSearch", "Acme rocket telemetry team"),
    ProgressEvent("tool", "WebFetch", "https://acme.example.com/about/"),
    ProgressEvent("tool", "Thinking about it", ""),
]


class StubLLM:
    def __init__(self, brief_data=None, report=REPORT, events=()):
        self.brief_data = BRIEF_DATA if brief_data is None else brief_data
        self.report = report
        self.events = events
        self.json_prompts = []
        self.research_prompts = []

    def complete_json(self, system, messages, schema):
        assert schema is prompts_preparation.COMPANY_RESEARCH_BRIEF_SCHEMA
        self.json_prompts.append(messages[0]["content"])
        return self.brief_data

    def complete_research(self, system, prompt, *, on_event=None):
        self.research_prompts.append(prompt)
        for event in self.events:
            if on_event is not None:
                on_event(event)
        return self.report


def make_researcher(tmp_path, llm=None):
    return CompanyResearcher(llm or StubLLM(), InterviewStorage(str(tmp_path)))


NOW = datetime(2026, 8, 5, 14, 30, 5)


def test_research_saves_a_timestamped_report_the_planner_can_read(tmp_path):
    llm = StubLLM()
    storage = InterviewStorage(str(tmp_path))
    researcher = CompanyResearcher(llm, storage)

    result = researcher.research("Acme, ML engineer, rocket telemetry", now=NOW)

    assert result.company == "Acme"
    assert result.role == "Machine Learning Engineer"
    assert result.path.name == "research_20260805_143005.md"
    assert result.path.read_text(encoding="utf-8") == REPORT
    # The whole point: it lands in the folder the interview planner reads.
    assert "They build rockets." in storage.load_company_context("Acme")


def test_the_brief_is_parsed_before_the_research_call_is_spent(tmp_path):
    """The extraction call resolves the folder, and steers the research prompt."""
    llm = StubLLM()
    make_researcher(tmp_path, llm).research("Acme, ML engineer", now=NOW)

    assert "Acme, ML engineer" in llm.json_prompts[0]
    prompt = llm.research_prompts[0]
    assert "Research Acme and write" in prompt
    assert "Machine Learning Engineer" in prompt
    assert "rocket telemetry team" in prompt
    assert "05 August 2026" in prompt


def test_a_brief_with_no_company_never_reaches_the_research_call(tmp_path):
    llm = StubLLM(brief_data={"company": "", "role": "", "focus": ""})
    with pytest.raises(LLMError, match="No company name"):
        make_researcher(tmp_path, llm).research("something about rockets", now=NOW)
    assert llm.research_prompts == []


def test_an_empty_brief_costs_no_model_call_at_all(tmp_path):
    llm = StubLLM()
    with pytest.raises(LLMError):
        make_researcher(tmp_path, llm).research("   ", now=NOW)
    assert llm.json_prompts == []
    assert llm.research_prompts == []


def test_progress_reports_each_stage_and_each_step_of_the_research_call(tmp_path):
    llm = StubLLM(events=EVENTS)
    lines = []

    make_researcher(tmp_path, llm).research("Acme", now=NOW, on_progress=lines.append)

    assert lines == [
        "Reading the request…",
        "Researching Acme — Machine Learning Engineer…",
        "Let me start with their own site.",
        'Searching the web: "Acme rocket telemetry team"',
        "Reading acme.example.com/about",
        "Thinking about it",
        "Saving the report…",
    ]


def test_a_run_with_no_progress_callback_behaves_exactly_as_before(tmp_path):
    llm = StubLLM(events=EVENTS)
    result = make_researcher(tmp_path, llm).research("Acme", now=NOW)
    assert result.markdown == REPORT


def test_a_rejected_brief_never_claims_research_started(tmp_path):
    """The failure is raised between the two stage lines, not after them."""
    llm = StubLLM(brief_data={"company": "", "role": "", "focus": ""}, events=EVENTS)
    lines = []

    researcher = make_researcher(tmp_path, llm)
    with pytest.raises(LLMError):
        researcher.research("rockets", now=NOW, on_progress=lines.append)

    assert lines == ["Reading the request…"]


def test_repeat_research_never_overwrites_an_earlier_report(tmp_path):
    storage = InterviewStorage(str(tmp_path))
    researcher = CompanyResearcher(StubLLM(), storage)

    first = researcher.research("Acme", now=NOW)
    second = researcher.research("Acme", now=datetime(2026, 9, 1, 9, 0, 0))

    assert first.path != second.path
    assert [p.name for p in storage.list_context_files("Acme")] == [
        first.path.name,
        second.path.name,
    ]
