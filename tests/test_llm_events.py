"""What the research call reports as it works.

`events_from_message` is the one piece of `llm_service` that can be tested
offline: it is pure, and the SDK message types it reads are plain dataclasses,
so nothing here opens a connection or starts the CLI.
"""

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from src.services.llm_service import ProgressEvent, events_from_message


def assistant(*blocks) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model="claude-opus-5")


def test_a_tool_call_reports_the_search_query_and_the_fetched_url():
    message = assistant(
        ToolUseBlock(id="1", name="WebSearch", input={"query": "Acme funding"}),
        ToolUseBlock(
            id="2",
            name="WebFetch",
            input={"url": "https://acme.example.com", "prompt": "revenue?"},
        ),
    )

    assert events_from_message(message) == [
        ProgressEvent("tool", "WebSearch", "Acme funding"),
        # `url` wins over `prompt`: it is what the model is actually reading.
        ProgressEvent("tool", "WebFetch", "https://acme.example.com"),
    ]


def test_a_tool_with_nothing_worth_quoting_still_reports_the_call():
    message = assistant(ToolUseBlock(id="1", name="WebSearch", input={}))
    assert events_from_message(message) == [ProgressEvent("tool", "WebSearch", "")]


def test_the_models_own_narration_is_reported_but_empty_text_is_not():
    message = assistant(
        TextBlock(text="  Now the funding round.  "), TextBlock(text=" ")
    )
    assert events_from_message(message) == [
        ProgressEvent("text", "", "Now the funding round.")
    ]


def test_thinking_and_tool_results_are_never_reported():
    """Both are the noise this feature exists to leave out: the model's internal
    draft, and pages of fetched web page."""
    message = assistant(
        ThinkingBlock(thinking="I should check the headcount claim", signature="sig"),
        ToolResultBlock(tool_use_id="1", content="a whole web page"),
        TextBlock(text="Checking the headcount."),
    )

    assert events_from_message(message) == [
        ProgressEvent("text", "", "Checking the headcount.")
    ]


def test_messages_that_are_not_assistant_turns_report_nothing():
    result = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        total_cost_usd=0.0,
    )
    assert events_from_message(result) == []
