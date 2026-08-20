"""InterviewManager flow tests against a stubbed LLM (no network)."""

import re

from src.core.interview_manager import InterviewManager
from src.core import prompts_common, prompts_interview, prompts_preparation
from src.models.interview import CANDIDATE, TEXT, VOICE, InterviewSettings
from src.storage.interview_storage import InterviewStorage

PLAN_DATA = {
    "checklist": [
        {
            "id": "motivation",
            "criterion": "Motivation",
            "description": "Knows why Acme",
        },
        {"id": "skills", "criterion": "Skills", "description": "Relevant experience"},
    ],
    "curveball": "a launch window closes in two days and half the team wants to slip it",
    "questions": ["Why Acme?", "Walk me through a project"],
    "mood_adjectives": [
        "brisk",
        "distracted",
        "curious",
        "impatient",
        "warm",
        "sceptical",
        "tired",
        "focused",
        "guarded",
        "upbeat",
    ],
    "company_brief": "Acme builds rockets.",
    "culture_anchors": ["Launch fast, learn faster", "Own the whole rocket"],
    # Simulated model output, so it may name specifics freely — the ban on
    # enumerating them applies to the prompt, not to what a planner returns.
    "interviewer_role": (
        "You are the recruiter who screens for the rockets team. Before anyone "
        "reaches the hiring manager you must be able to say when they could start, "
        "where they would work from, and what they need to earn."
    ),
}

EVALUATION_DATA = {
    "score": 81,
    "results": [
        {
            "id": "motivation",
            "criterion": "Motivation",
            "rating": "met",
            "evidence": "rockets",
            "comment": "good",
        },
        {
            "id": "skills",
            "criterion": "Skills",
            "rating": "partial",
            "evidence": "some experience",
            "comment": "thin",
        },
    ],
    "strengths": ["clear"],
    "improvements": ["more numbers"],
    "summary": "Strong round.",
    "key_takeaway": {
        "point": "Be specific: name the number you moved",
        "severity": 2,
        "evidence": '"a lot better" — said three times, never a figure',
        "verdict": "An interviewer would call this unverifiable.",
    },
    "progress_notes": "",
    "topics_covered": ["the rocket telemetry project"],
    "answer_review": [
        {"question": "Why Acme?", "addressed": "full", "comment": "Answered directly."},
        {
            "question": "Walk me through a project, and what you'd do differently",
            "addressed": "partial",
            "comment": "Never said what he would do differently.",
        },
    ],
}


def _is_plan_schema(schema):
    """Which of the two schemas the manager is calling with.

    Structural rather than `is prompts_preparation.PLAN_SCHEMA`: the plan schema is built per
    round now, because the stages that have no use for a curveball do not carry
    the field at all.
    """
    return "checklist" in schema["properties"]


class StubLLM:
    def __init__(self, reply_text="Interesting. Can you give an example?"):
        self.reply_text = reply_text
        self.json_prompts = []
        self.text_calls = []

    def complete(self, system, messages, max_tokens=8000):
        self.text_calls.append((system, messages))
        return self.reply_text

    def complete_json(self, system, messages, schema, max_tokens=16000):
        self.json_prompts.append(messages[0]["content"])
        return PLAN_DATA if _is_plan_schema(schema) else EVALUATION_DATA


def make_manager(tmp_path, llm=None):
    return InterviewManager(llm or StubLLM(), InterviewStorage(str(tmp_path)))


SETTINGS = InterviewSettings("Acme", "Engineer", "HR Screen", "background info")
SHORT_SETTINGS = InterviewSettings(
    "Acme", "Engineer", "HR Screen", "background info", duration_minutes=10
)
# A stage that does want a situational scenario and a culture probe, so the
# stage-gated halves of the plan are covered from both sides.
BEHAVIOURAL_SETTINGS = InterviewSettings(
    "Acme", "Engineer", "Hiring Manager", "background info"
)


def test_start_creates_session_with_a_live_opening(tmp_path):
    """The opening is generated, not planned, so it can carry the round's mood."""
    llm = StubLLM(reply_text="Morning — thanks for making the time. Why Acme?")
    manager = make_manager(tmp_path, llm)
    session, previous_rounds = manager.start(SETTINGS)
    assert previous_rounds == 0
    assert session.plan.questions == PLAN_DATA["questions"]
    assert len(session.transcript) == 1
    assert session.transcript[0].role == "interviewer"
    assert (
        session.transcript[0].content
        == "Morning — thanks for making the time. Why Acme?"
    )

    # Opening turn: no answer to react to, and the closing gate shut.
    system, messages = llm.text_calls[0]
    assert "the candidate has just joined" in system
    assert "may NOT end the interview" in system
    assert messages == [
        {"role": "user", "content": "(The candidate has joined the interview.)"}
    ]


def test_the_opening_is_spoken_in_the_rounds_mood(tmp_path):
    """Two of the planner's ten adjectives are drawn and reach the live interviewer."""
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)

    mood = session.round_profile["mood"]
    assert len(mood) == 2
    assert set(mood) < set(PLAN_DATA["mood_adjectives"])

    system, _ = llm.text_calls[0]
    assert f"Your mood today, specifically: {' and '.join(mood)}." in system


def test_a_thin_mood_pool_does_not_break_the_round(tmp_path):
    """A degraded plan response should cost variety, not the interview."""
    assert prompts_common.pick_mood([]) == []
    assert prompts_common.pick_mood(["brisk"]) == []


def test_reply_appends_both_turns(tmp_path):
    manager = make_manager(tmp_path)
    session, _ = manager.start(SETTINGS)
    reply, complete = manager.reply(session, "Because rockets.")
    assert not complete
    assert reply == "Interesting. Can you give an example?"
    assert [t.role for t in session.transcript] == [
        "interviewer",
        "candidate",
        "interviewer",
    ]


def test_complete_token_is_stripped_and_flagged(tmp_path):
    llm = StubLLM(
        reply_text=f"Thanks, that's all I need. {prompts_interview.COMPLETE_TOKEN}"
    )
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SHORT_SETTINGS)
    # The closing gate only opens once the candidate's questions phase is due.
    # One-word answers, so the round runs to the ceiling of the answer band.
    _, ceiling = prompts_interview.answer_guards(SHORT_SETTINGS.duration_minutes)
    for _ in range(ceiling - 1):
        _, complete = manager.reply(session, "Answer.")
        assert not complete
    reply, complete = manager.reply(session, "Answer.")
    assert complete
    assert prompts_interview.COMPLETE_TOKEN not in reply
    assert prompts_interview.COMPLETE_TOKEN not in session.transcript[-1].content


def test_early_close_is_refused(tmp_path):
    """The model closing in the same breath as the first answer must not end the round.

    That failure mode cost a real candidate a "you only asked one question"
    criticism they had no way to avoid.
    """
    llm = StubLLM(
        reply_text=f"Any questions? No? Great, we're done. {prompts_interview.COMPLETE_TOKEN}"
    )
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    reply, complete = manager.reply(session, "Answer.")
    assert not complete
    assert prompts_interview.COMPLETE_TOKEN not in reply


def test_interviewer_prompt_reserves_a_questions_phase(tmp_path):
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Answer.")
    system, _ = llm.text_calls[-1]
    assert "may NOT end the interview" in system
    assert "this is candidate answer 1." in system
    assert "minutes have gone" in system


def test_voice_answers_are_marked_and_flagged_to_the_evaluator(tmp_path):
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Spoken answer.", VOICE)
    manager.reply(session, "Typed answer.")
    assert [t.modality for t in session.transcript if t.role == CANDIDATE] == [
        VOICE,
        TEXT,
    ]

    manager.finish(session)
    evaluation_prompt = llm.json_prompts[-1]
    assert "CANDIDATE (spoken, auto-transcribed): Spoken answer." in evaluation_prompt
    assert "CANDIDATE: Typed answer." in evaluation_prompt


def test_culture_anchors_reach_the_interviewer_and_the_evaluator(tmp_path):
    """A published values framework has to be screened for, not just sit in the context."""
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(BEHAVIOURAL_SETTINGS)
    manager.reply(session, "Answer.")

    system, _ = llm.text_calls[-1]
    assert "Launch fast, learn faster" in system
    assert "never name them to the candidate" in system

    manager.finish(session)
    assert "Own the whole rocket" in llm.json_prompts[-1]


def test_a_stage_that_screens_for_no_principles_is_not_told_to(tmp_path):
    """The anchors are still extracted — they are company facts, and the evaluator
    reads them — but a round with no culture question in its pool has nothing to
    screen against, and being told to screen strictly only improvises one."""
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)  # an HR screen
    manager.reply(session, "Answer.")

    system, _ = llm.text_calls[-1]
    assert "Launch fast, learn faster" not in system
    assert session.plan.culture_anchors == PLAN_DATA["culture_anchors"]


def test_the_interviewers_job_reaches_them_but_never_the_evaluator(tmp_path):
    """The interviewer needs it to triage; the evaluator must not have it.

    A list of what "should have been covered" is exactly what the fairness rule
    forbids — a question the interviewer owed and never asked is not_assessed,
    never a mark against the candidate.
    """
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    assert session.plan.interviewer_role == PLAN_DATA["interviewer_role"]
    manager.reply(session, "Answer.")

    system, _ = llm.text_calls[-1]
    assert "what they need to earn" in system
    assert "is yours to get" in system

    manager.finish(session)
    assert "what they need to earn" not in llm.json_prompts[-1]

    stored = manager.storage.load_sessions(SETTINGS)[-1]
    assert stored.plan.interviewer_role == PLAN_DATA["interviewer_role"]


def test_a_replay_keeps_the_interviewer_it_was_planned_with(tmp_path):
    manager = make_manager(tmp_path)
    source = _finished_round(manager)
    session, _ = manager.start(source.settings, replay_of=source)
    assert session.plan.interviewer_role == source.plan.interviewer_role


def test_plan_prompt_requires_culture_coverage(tmp_path):
    llm = StubLLM()
    make_manager(tmp_path, llm).start(BEHAVIOURAL_SETTINGS)
    plan_prompt = llm.json_prompts[-1]
    assert '"culture_anchors"' in plan_prompt
    assert "at least one criterion must assess lived fit" in plan_prompt
    assert "without naming the principle or the framework" in plan_prompt


def test_the_culture_criterion_and_its_question_are_on_one_switch(tmp_path):
    """Never one without the other: a criterion nothing asks about can only ever
    come back not_assessed, which costs the round a scorecard slot for nothing."""
    llm = StubLLM()
    make_manager(tmp_path, llm).start(SETTINGS)  # an HR screen
    plan_prompt = llm.json_prompts[-1]
    # Still extracted — the brief and the evaluator both want them.
    assert '"culture_anchors"' in plan_prompt
    assert "at least one criterion must assess lived fit" not in plan_prompt
    assert "without naming the principle or the framework" not in plan_prompt


def test_prompts_forbid_revealing_the_rubric(tmp_path):
    """Planned questions and live turns must not read the scorecard out loud."""
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Answer.")

    plan_prompt = llm.json_prompts[0]
    opening_system, _ = llm.text_calls[0]
    system, _ = llm.text_calls[-1]
    for prompt in (plan_prompt, system):
        assert "NEVER REVEAL WHAT YOU ARE MEASURING" in prompt
        assert "Do not pre-announce what a strong answer contains" in prompt
    assert "No question may restate the criterion it tests" in plan_prompt
    # The opening is a live turn now, so its rubric guard lives there.
    assert "never a list of what you are assessing" in opening_system


def test_answers_are_evaluated_against_the_questions_they_answer(tmp_path):
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Answer.")
    evaluation = manager.finish(session)

    prompt = llm.json_prompts[-1]
    assert "JUDGE ANSWERS AGAINST THE QUESTIONS THEY ANSWER" in prompt
    assert "A half-answered question" in prompt
    assert "Did it take probing to get there?" in prompt
    assert '"answer_review"' in prompt

    assert [a.addressed for a in evaluation.answer_review] == ["full", "partial"]
    assert evaluation.answer_review[1].comment.startswith("Never said")


def test_analysis_is_generated_before_the_score(tmp_path):
    """Structured outputs emit fields in schema order; the score must follow its evidence."""
    keys = list(prompts_interview.evaluation_schema(True)["properties"])
    assert keys.index("answer_review") < keys.index("results") < keys.index("score")


def test_answer_review_survives_the_storage_round_trip(tmp_path):
    manager = make_manager(tmp_path)
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Answer.")
    manager.finish(session)

    stored = manager.storage.load_sessions(SETTINGS)[-1]
    assert [a.to_dict() for a in stored.evaluation.answer_review] == EVALUATION_DATA[
        "answer_review"
    ]
    assert stored.evaluation.key_takeaway.to_dict() == EVALUATION_DATA["key_takeaway"]


def test_evaluation_prompt_carries_the_fairness_and_transcription_clauses(tmp_path):
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Answer.")
    manager.finish(session)
    evaluation_prompt = llm.json_prompts[-1]
    assert "RATE WHAT THE CANDIDATE HAD THE CHANCE TO SHOW" in evaluation_prompt
    assert "is not a valid criticism" in evaluation_prompt
    assert "transcription artifact" in evaluation_prompt


def test_progress_notes_dropped_from_schema_without_history(tmp_path):
    assert (
        "progress_notes" not in prompts_interview.evaluation_schema(False)["properties"]
    )
    assert "progress_notes" in prompts_interview.evaluation_schema(True)["properties"]

    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Answer.")
    manager.finish(session)
    assert session.evaluation.progress_notes == ""


def test_score_is_anchored_to_the_checklist_ratings(tmp_path):
    class GenerousLLM(StubLLM):
        def complete_json(self, system, messages, schema, max_tokens=16000):
            data = super().complete_json(system, messages, schema, max_tokens)
            if _is_plan_schema(schema):
                return data
            return {**data, "score": 99}  # ratings imply 75

    manager = make_manager(tmp_path, GenerousLLM())
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Answer.")
    evaluation = manager.finish(session)
    assert evaluation.score == 85  # 75 implied + the 10-point delivery allowance


def test_failed_reply_leaves_transcript_unchanged(tmp_path):
    class FailingLLM(StubLLM):
        """Serves the opening, then fails — the failure under test is in `reply`."""

        def complete(self, system, messages, max_tokens=8000):
            if not self.text_calls:
                return super().complete(system, messages, max_tokens)
            raise RuntimeError("boom")

    manager = make_manager(tmp_path, FailingLLM())
    session, _ = manager.start(SETTINGS)
    try:
        manager.reply(session, "Answer.")
    except RuntimeError:
        pass
    assert len(session.transcript) == 1  # retryable: no dangling candidate turn


def test_undo_last_answer_removes_both_turns(tmp_path):
    manager = make_manager(tmp_path)
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Because rockets.")
    assert manager.undo_last_answer(session)
    assert [t.role for t in session.transcript] == ["interviewer"]


def test_undo_last_answer_refuses_on_the_opening_turn(tmp_path):
    """The interviewer's opening must survive any number of undos."""
    manager = make_manager(tmp_path)
    session, _ = manager.start(SETTINGS)
    assert not manager.undo_last_answer(session)
    assert len(session.transcript) == 1


def test_undo_restores_the_previous_question(tmp_path):
    """After an undo the candidate faces the question they were asked before."""
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    llm.reply_text = "First follow-up."
    manager.reply(session, "Answer one.")
    llm.reply_text = "Second follow-up."
    manager.reply(session, "Answer two.")

    assert manager.undo_last_answer(session)
    assert session.transcript[-1].role == "interviewer"
    assert session.transcript[-1].content == "First follow-up."
    assert "Answer two." not in [t.content for t in session.transcript]


def test_undo_reshuts_the_closing_gate(tmp_path):
    """`may_close` counts answers in the transcript, so undoing rewinds it.

    Undoing once and answering again lands on the same count and may close again —
    that is the point. Undoing past the budget must put the gate back.
    """
    llm = StubLLM(
        reply_text=f"Thanks, that's all I need. {prompts_interview.COMPLETE_TOKEN}"
    )
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SHORT_SETTINGS)
    _, ceiling = prompts_interview.answer_guards(SHORT_SETTINGS.duration_minutes)
    for _ in range(ceiling):
        _, complete = manager.reply(session, "Answer.")
    assert complete

    assert manager.undo_last_answer(session)
    assert manager.undo_last_answer(session)
    # Same stub, same closing token — only the shortened transcript differs.
    _, complete = manager.reply(session, "Answer again.")
    assert not complete
    assert prompts_interview.COMPLETE_TOKEN not in session.transcript[-1].content


def test_undo_preserves_the_round_profile(tmp_path):
    """Persona, mood and plan are drawn once per round and must survive an undo."""
    manager = make_manager(tmp_path)
    session, _ = manager.start(SETTINGS)
    profile, plan = session.round_profile, session.plan
    manager.reply(session, "Answer.")
    manager.undo_last_answer(session)
    assert session.round_profile is profile
    assert session.plan is plan
    assert session.evaluation is None


def test_finish_persists_and_next_round_sees_history(tmp_path):
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Because rockets.")
    evaluation = manager.finish(session)
    assert evaluation.score == 81
    assert session.evaluation is not None

    # Second round: history digest should feed the plan prompt
    session2, previous_rounds = manager.start(SETTINGS)
    assert previous_rounds == 1
    plan_prompt = llm.json_prompts[-1]
    assert "Why Acme?" in plan_prompt  # previously asked question listed
    assert "more numbers" in plan_prompt  # previous improvement listed
    assert "81" in plan_prompt  # previous score listed


def test_the_fixed_criterion_is_scored_every_round_and_never_replanned(tmp_path):
    """Added in Python, so its wording — and its ratings — stay comparable.

    The planner never sees it: shown it as a criterion to carry over, it writes
    its own version and the round is scored on the same thing twice.
    """
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    criterion = prompts_common.STRUCTURE_CRITERION

    session, _ = manager.start(SETTINGS)
    assert [c.id for c in session.plan.checklist][-1] == criterion.id
    assert criterion.criterion in llm.text_calls[-1][0]  # the live interviewer

    manager.reply(session, "Because rockets.")
    manager.finish(session)
    assert criterion.criterion in llm.json_prompts[-1]  # the evaluator

    manager.start(SETTINGS)
    plan_prompt = llm.json_prompts[-1]
    carried = plan_prompt.split("CHECKLIST USED IN THE PREVIOUS ROUND:")[1]
    carried = carried.split("Keep the")[0]
    assert "[motivation]" in carried
    assert f"[{criterion.id}]" not in carried


def test_planned_scenario_is_recorded_and_avoided_next_round(tmp_path):
    """The planner writes the round's curveball; later rounds must not get it again.

    On a stage that asks for one — an HR screen does not, and is covered below.
    """
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    settings = InterviewSettings("Acme", "Engineer", "Technical", "background info")

    session, _ = manager.start(settings)
    # Nothing to avoid on the first round.
    assert "SITUATIONAL SCENARIOS USED IN RECENT ROUNDS" not in llm.json_prompts[0]
    assert session.round_profile["curveball"] == PLAN_DATA["curveball"]
    manager.reply(session, "Because rockets.")
    manager.finish(session)

    manager.start(settings)
    plan_prompt = llm.json_prompts[-1]
    assert "SITUATIONAL SCENARIOS USED IN RECENT ROUNDS" in plan_prompt
    assert PLAN_DATA["curveball"] in plan_prompt


def test_the_round_records_who_ran_it_and_what_it_was_for(tmp_path):
    """Snapshotted on the plan, not re-derived on display: the constant it came
    from is prompt text and will be edited, and the History tab has to keep
    showing what this round actually ran with. A replay carries it over."""
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    hr = prompts_common.INTERVIEWER_PROFESSIONS["HR Screen"]
    session, _ = manager.start(SETTINGS)
    who = prompts_common.profession_block(hr)
    what = prompts_common.round_purpose_block(hr)
    assert session.plan.interviewer_profession == who
    assert session.plan.round_purpose == what

    manager.reply(session, "Answer.")
    manager.finish(session)
    stored = manager.storage.load_sessions(SETTINGS)[-1]
    assert stored.plan.interviewer_profession == who
    assert stored.plan.round_purpose == what

    replay, _ = manager.start(SETTINGS, replay_of=stored)
    assert replay.plan.interviewer_profession == who
    assert replay.plan.round_purpose == what


def test_a_stage_with_no_use_for_a_scenario_is_never_asked_for_one(tmp_path):
    """Not a prompt-only rule: a required field comes back filled whether or not
    the round has any use for it, so the field leaves the schema as well."""
    schemas = []

    class SchemaSpy(StubLLM):
        def complete_json(self, system, messages, schema, max_tokens=16000):
            schemas.append(schema)
            return super().complete_json(system, messages, schema, max_tokens)

    llm = SchemaSpy()
    manager = make_manager(tmp_path, llm)
    manager.start(SETTINGS)  # an HR screen

    assert "curveball" not in schemas[0]["properties"]
    assert "curveball" not in schemas[0]["required"]
    plan_prompt = llm.json_prompts[0]
    assert '"curveball"' not in plan_prompt
    # And the scenario bullet's absence must not leave a hole in the numbering:
    # a list that jumps from 5 to 7 reads as an item the prompt forgot to
    # include. Checked as a sequence rather than against a fixed number, since
    # the optional bullets that renumber it are gated per stage.
    numbered = re.findall(r"^(\d+)\. \"", plan_prompt, flags=re.MULTILINE)
    assert numbered == [str(i) for i in range(1, len(numbered) + 1)]
    assert plan_prompt.count(f'{len(numbered)}. "questions"') == 1


def _finished_round(manager, session_id=None, settings=SETTINGS):
    """Run and store one round. Ids are second-resolution, so tests set their own."""
    session, _ = manager.start(settings)
    if session_id:
        session.id = session_id
    manager.reply(session, "Answer.")
    manager.finish(session)
    return session


def test_replay_reuses_the_plan_without_a_planner_call(tmp_path):
    """The point of a replay: the same prepared interview, not a new one."""
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    source = _finished_round(manager)
    planner_calls, text_calls = len(llm.json_prompts), len(llm.text_calls)

    session, _ = manager.start(source.settings, replay_of=source)

    assert len(llm.json_prompts) == planner_calls  # nothing was planned afresh
    assert len(llm.text_calls) == text_calls + 1  # only the opening is generated
    assert session.plan.to_dict() == source.plan.to_dict()
    assert session.round_profile == source.round_profile
    assert session.replay_of == source.id
    # Copies: the caller is still holding the stored session.
    assert session.plan is not source.plan
    assert session.round_profile is not source.round_profile
    assert len(session.transcript) == 1


def test_replay_sees_the_history_its_source_round_saw(tmp_path):
    """Not the source itself: its questions must not land on the avoid list."""
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    _finished_round(manager, "20260101_090000")
    second = _finished_round(manager, "20260201_090000")
    _finished_round(manager, "20260301_090000")

    session, previous_rounds = manager.start(second.settings, replay_of=second)
    assert previous_rounds == 1

    opening_system, _ = llm.text_calls[-1]
    assert "practiced this interview 1 time(s) before" in opening_system
    # Every later turn resolves the same scope, not the full history.
    manager.reply(session, "Answer.")
    system, _ = llm.text_calls[-1]
    assert "practiced this interview 1 time(s) before" in system


def test_replay_is_stored_alongside_its_source(tmp_path):
    manager = make_manager(tmp_path)
    source = _finished_round(manager, "20260101_090000")

    session, _ = manager.start(source.settings, replay_of=source)
    session.id = "20260201_090000"
    manager.reply(session, "A better answer this time.")
    manager.finish(session)

    stored = manager.storage.load_sessions(SETTINGS)
    assert [s.id for s in stored] == ["20260101_090000", "20260201_090000"]
    assert stored[0].transcript[1].content == "Answer."  # source left alone
    assert stored[1].replay_of == "20260101_090000"  # marker survives the round trip


def test_start_without_history_ignores_stored_rounds(tmp_path):
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    _finished_round(manager, "20260101_090000")

    session, previous_rounds = manager.start(SETTINGS, use_history=False)
    session.id = "20260201_090000"
    assert previous_rounds == 0
    assert "HISTORY OF PREVIOUS PRACTICE ROUNDS" not in llm.json_prompts[-1]
    assert "CHECKLIST USED IN THE PREVIOUS ROUND" not in llm.json_prompts[-1]
    assert session.round_profile["pressure"] == 1  # no escalation to inherit

    manager.reply(session, "Answer.")
    system, _ = llm.text_calls[-1]
    assert "practiced this interview" not in system

    manager.finish(session)
    assert "FEEDBACK FROM PREVIOUS PRACTICE ROUNDS" not in llm.json_prompts[-1]
    assert session.evaluation.progress_notes == ""


def test_start_without_history_still_loads_company_context(tmp_path):
    """Research files are the candidate's preparation, not a record of past rounds."""
    folder = tmp_path / "context" / "acme"
    folder.mkdir(parents=True)
    (folder / "research.md").write_text("They build rockets.", encoding="utf-8")

    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS, use_history=False)

    assert "They build rockets." in session.settings.company_context
    assert "They build rockets." in llm.json_prompts[-1]


def test_start_appends_static_company_context(tmp_path):
    folder = tmp_path / "context" / "acme"
    folder.mkdir(parents=True)
    (folder / "research.md").write_text("They build rockets.", encoding="utf-8")

    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)

    # Kept apart from the candidate's own background: it is their prep material,
    # not something the interviewer may treat as the correct answers.
    assert session.settings.background == "background info"
    assert "They build rockets." in session.settings.company_context
    assert "They build rockets." in llm.json_prompts[-1]  # reaches the plan prompt
    assert SETTINGS.company_context == ""  # caller's settings untouched


def test_company_context_is_distilled_before_reaching_the_live_interviewer(tmp_path):
    folder = tmp_path / "context" / "acme"
    folder.mkdir(parents=True)
    (folder / "research.md").write_text(
        "Questions to ask them: what is the roadmap?", encoding="utf-8"
    )

    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Answer.")

    system, _ = llm.text_calls[-1]
    assert "Acme builds rockets." in system  # the plan's brief
    assert "Questions to ask them" not in system  # ...not the raw prep notes


def test_persona_is_drawn_once_and_held_for_the_whole_interview(tmp_path):
    """A random draw must be stored, not re-rolled: one interviewer per interview."""
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    assert session.round_profile["archetype"]

    for _ in range(3):
        manager.reply(session, "Answer.")
    personas = {
        system.split('INTERVIEWER PERSONA FOR THIS ROUND — "')[1].split('"')[0]
        for system, _ in llm.text_calls
    }
    assert personas == {session.round_profile["archetype"]}


def test_round_profile_survives_the_storage_round_trip(tmp_path):
    """Later rounds can only avoid recent personas if the old ones were persisted."""
    manager = make_manager(tmp_path)
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Answer.")
    manager.finish(session)

    stored = manager.storage.load_sessions(SETTINGS)[-1]
    assert stored.round_profile == session.round_profile


def test_plan_prompt_varies_between_rounds(tmp_path):
    llm = StubLLM()
    manager = make_manager(tmp_path, llm)
    session, _ = manager.start(SETTINGS)
    manager.reply(session, "Answer.")
    manager.finish(session)
    first = llm.json_prompts[0]

    manager.start(SETTINGS)
    second = llm.json_prompts[-1]

    def persona(prompt: str) -> str:
        return prompt.split('INTERVIEWER PERSONA FOR THIS ROUND — "')[1].split('"')[0]

    assert persona(first) != persona(second)
    assert "the rocket telemetry project" in second  # story already used, ask for fresh
    assert "Skills (rated partial)" in second  # last round's weak spot gets probed


def test_history_digest_empty_without_previous():
    assert prompts_preparation.history_digest([]) == ""
