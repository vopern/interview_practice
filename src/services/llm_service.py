"""Thin adapter around the Claude Agent SDK.

Exposes three generic calls (plain text, schema-enforced JSON, and plain text
with web access); all prompt content lives in src/core/prompts_*.py and all
orchestration in src/core/interview_manager.py and src/core/company_researcher.py.

The SDK drives the bundled Claude Code CLI, so authentication uses whatever
Claude Code is logged in with (a claude.ai subscription via ``claude login``,
or ``ANTHROPIC_API_KEY`` if set) — the app itself needs no API key.
"""

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ServerToolUseBlock,
    TextBlock,
    ToolUseBlock,
    query,
)

# A retry allowance for the SDK's structured-output validation loop; a plain
# tool-less completion finishes in a single turn.
MAX_TURNS = 3

# Research is a genuine tool loop — a search, a read, a follow-up search — so it
# needs a turn budget of a different order than the two above.
RESEARCH_TOOLS = ["WebSearch", "WebFetch"]
RESEARCH_MAX_TURNS = 60


class LLMError(Exception):
    """The model call failed or was refused."""


@dataclass
class ProgressEvent:
    """One thing the model did on its way to an answer.

    Not phrased for a UI: this layer stays generic, so it reports *what
    happened* (`kind`, `name`, `detail`) and the caller decides how — or whether
    — to say it. `kind` is ``"tool"`` (a tool call, `name` is the tool) or
    ``"text"`` (the model's own narration between tool calls, `name` empty).
    """

    kind: str
    name: str
    detail: str


# The input field worth reporting, in the order a tool is likely to carry one:
# WebSearch passes `query`, WebFetch a `url` plus the `prompt` it reads it with.
_DETAIL_KEYS = ("query", "url", "prompt")


def _tool_detail(tool_input: dict) -> str:
    for key in _DETAIL_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def events_from_message(message) -> list[ProgressEvent]:
    """Pick the reportable moments out of one SDK stream message.

    Only assistant turns say anything about progress. Thinking blocks and tool
    *results* are dropped as noise. Pure and module-level, so it can be tested
    without the SDK ever opening a connection.
    """
    if not isinstance(message, AssistantMessage):
        return []
    events = []
    for block in message.content:
        if isinstance(block, (ToolUseBlock, ServerToolUseBlock)):
            events.append(
                ProgressEvent("tool", block.name, _tool_detail(block.input or {}))
            )
        elif isinstance(block, TextBlock) and block.text.strip():
            events.append(ProgressEvent("text", "", block.text.strip()))
    return events


def _unescape(text: str) -> str:
    # Models occasionally double-escape whitespace inside JSON strings, leaving
    # a literal backslash-n in the parsed value that the UI would show verbatim.
    return text.replace("\\n", "\n").replace("\\t", "\t")


def _unescape_deep(value):
    if isinstance(value, str):
        return _unescape(value)
    if isinstance(value, list):
        return [_unescape_deep(v) for v in value]
    if isinstance(value, dict):
        return {k: _unescape_deep(v) for k, v in value.items()}
    return value


class LLMService:
    """Routes each call to one of two model tiers.

    ``complete`` drives the live interviewer, where the candidate waits on every
    turn: fast tier, shallow effort. ``complete_json`` drives planning and
    evaluation — one-shot, judgment-heavy, nobody waiting mid-sentence: strong
    tier, deep effort. Effort is the larger quality/latency lever of the two;
    leave thinking at the SDK default (adaptive) rather than disabling it.

    ``complete_research`` is the strong tier too, and the only call with tools.
    """

    def __init__(
        self,
        model: str,
        reasoning_model: str | None = None,
        effort: str = "low",
        reasoning_effort: str = "high",
    ):
        self.model = model
        self.reasoning_model = reasoning_model or model
        self.effort = effort
        self.reasoning_effort = reasoning_effort

    def _run(
        self,
        system: str,
        prompt: str,
        output_format: dict | None = None,
        *,
        reasoning: bool = False,
        research: bool = False,
        on_event: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> ResultMessage:
        # Everything but research is pure text generation with no filesystem,
        # web or bash access. Research gets exactly two tools, and gets them
        # pre-approved: there is no can_use_tool callback and a Streamlit server
        # cannot answer a permission prompt, so an unapproved tool call hangs.
        options = ClaudeAgentOptions(
            system_prompt=system,
            model=self.reasoning_model if reasoning or research else self.model,
            effort=self.reasoning_effort if reasoning or research else self.effort,
            tools=RESEARCH_TOOLS if research else [],
            allowed_tools=RESEARCH_TOOLS if research else [],
            max_turns=RESEARCH_MAX_TURNS if research else MAX_TURNS,
            output_format=output_format,
        )

        async def run() -> ResultMessage | None:
            result: ResultMessage | None = None
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result = message
                elif on_event is not None:
                    for event in events_from_message(message):
                        # Progress is advisory: a callback that raises must not
                        # take a minutes-long research run down with it.
                        try:
                            on_event(event)
                        except Exception:
                            pass
            return result

        result = asyncio.run(run())
        if result is None:
            raise LLMError("The model call ended without a result.")
        if result.subtype != "success" or result.is_error:
            raise LLMError(f"The model call failed ({result.subtype}): {result.errors}")
        return result

    @staticmethod
    def _messages_to_prompt(messages: list[dict]) -> str:
        """Flatten role-tagged messages into a single prompt.

        The Agent SDK takes one user prompt per query rather than an
        assistant/user message history, so multi-turn conversations are
        rendered as a labeled transcript with an instruction to continue it.
        """
        if len(messages) == 1:
            return messages[0]["content"]
        transcript = "\n\n".join(f"[{m['role']}]: {m['content']}" for m in messages)
        return (
            f"<conversation>\n{transcript}\n</conversation>\n\n"
            "Continue this conversation: write the assistant's next reply only, "
            "without the [assistant] label."
        )

    def complete(self, system: str, messages: list[dict]) -> str:
        result = self._run(system, self._messages_to_prompt(messages))
        if not result.result:
            raise LLMError("The model returned an empty response.")
        return _unescape(result.result)

    def complete_json(self, system: str, messages: list[dict], schema: dict) -> dict:
        result = self._run(
            system,
            self._messages_to_prompt(messages),
            output_format={"type": "json_schema", "schema": schema},
            reasoning=True,
        )
        if result.structured_output is None:
            raise LLMError("The model returned no structured output.")
        return _unescape_deep(result.structured_output)

    def complete_research(
        self,
        system: str,
        prompt: str,
        *,
        on_event: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> str:
        """A single completion that may search and fetch the web as it works.

        Never combined with ``output_format``: this is a long-running tool loop
        ending in free-form prose, and a schema on top of that is the fragile
        combination. It takes minutes rather than seconds, which is why it is
        the one call that reports progress — pass ``on_event`` to be told about
        each search, fetch and aside as it happens.
        """
        result = self._run(system, prompt, research=True, on_event=on_event)
        if not result.result:
            raise LLMError("The model returned an empty response.")
        # No _unescape here: the result is a Markdown document rather than a
        # value pulled out of a JSON string, so a literal backslash-n in a code
        # block or a path is the author's, not an escaping artefact.
        return result.result
