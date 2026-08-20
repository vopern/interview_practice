"""Unit tests for the prompt-construction helpers.

These cover the parts that are deliberately decided in Python rather than left
to the model: round variation, the closing gate, transcript labelling and the
history digest.
"""

from datetime import datetime

from src.core import prompts_common, prompts_interview, prompts_preparation
from src.models.interview import (
    STAGES,
    TAKEAWAY_SEVERITIES,
    VOICE,
    ChecklistItem,
    ChecklistResult,
    Evaluation,
    InterviewPlan,
    InterviewSession,
    InterviewSettings,
    KeyTakeaway,
    Turn,
)

SETTINGS = InterviewSettings("Acme", "Engineer", "HR Screen")
# A stage that does want a scenario and a culture probe. The two exist so the
# stage-gated pieces of the plan are tested from both sides: an HR screen is not
# asked for either, and that is the point of the gate rather than a gap in it.
DEEP_SETTINGS = InterviewSettings("Acme", "Engineer", "Hiring Manager")


def _session(
    created_at="2026-01-01T10:00:00", questions=None, evaluation=None, profile=None
):
    return InterviewSession(
        id="1",
        created_at=created_at,
        settings=SETTINGS,
        plan=InterviewPlan(
            checklist=[ChecklistItem("motivation", "Motivation", "why us")],
            questions=questions or ["Why Acme?"],
        ),
        evaluation=evaluation,
        round_profile=profile or {},
    )


def _flat(text: str) -> str:
    """Prompt text with its line wrapping collapsed.

    The prompts wrap at ~90 characters, so a phrase worth asserting on often
    straddles a newline. Only for assertions about wording; where the layout
    itself is the point (an indent that has to survive), match the raw string.
    """
    return " ".join(text.split())


def _evaluation(**kwargs):
    defaults = dict(
        score=70,
        results=[ChecklistResult("motivation", "Motivation", "partial", "e", "c")],
        strengths=["clear"],
        improvements=["more numbers"],
        summary="ok",
        topics_covered=["the re-ranking service"],
    )
    return Evaluation(**{**defaults, **kwargs})


# --- round variation -------------------------------------------------------


def test_profile_is_drawn_from_the_pool():
    profile = prompts_common.pick_round_profile([])
    assert profile["archetype"] in {
        a["name"] for a in prompts_common.INTERVIEWER_ARCHETYPES
    }
    assert profile["style"]
    # The scenario is the planner's job now, not the draw's.
    assert "curveball" not in profile


def test_first_round_persona_is_not_predictable():
    """Random, not a fixed rotation: repeated draws must not all agree."""
    drawn = {prompts_common.pick_round_profile([])["archetype"] for _ in range(50)}
    assert len(drawn) > 1


def test_draw_avoids_personas_used_in_recent_rounds():
    used = [a["name"] for a in prompts_common.INTERVIEWER_ARCHETYPES[:2]]
    previous = [_session(profile={"archetype": name}) for name in used]
    for _ in range(30):
        assert prompts_common.pick_round_profile(previous)["archetype"] not in used


def test_draw_falls_back_to_the_full_pool_when_everything_is_recent():
    previous = [
        _session(profile={"archetype": a["name"]})
        for a in prompts_common.INTERVIEWER_ARCHETYPES
    ]
    profile = prompts_common.pick_round_profile(
        previous
    )  # must not raise on an empty pool
    assert profile["archetype"] in {
        a["name"] for a in prompts_common.INTERVIEWER_ARCHETYPES
    }


def test_draw_is_reproducible_with_an_injected_rng():
    import random

    first = prompts_common.pick_round_profile([], rng=random.Random(7))
    assert prompts_common.pick_round_profile([], rng=random.Random(7)) == first


def test_pressure_ramps_with_rounds_and_with_a_strong_last_score():
    assert prompts_common.pick_round_profile([])["pressure"] == 1
    assert prompts_common.pick_round_profile([_session(), _session()])["pressure"] == 3
    assert prompts_common.pick_round_profile([], last_score=85)["pressure"] == 2


def test_every_archetype_says_how_it_sounds():
    for archetype in prompts_common.INTERVIEWER_ARCHETYPES:
        assert archetype["delivery"].strip()
        # Looked up by name at speech time, so the two must not drift apart.
        assert archetype["name"] in prompts_common._ARCHETYPE_DELIVERY


def test_tts_instructions_carry_the_persona_pressure_and_mood():
    archetype = prompts_common.INTERVIEWER_ARCHETYPES[3]  # the quiet one
    spoken = prompts_common.tts_instructions(
        {
            "archetype": archetype["name"],
            "pressure": 3,
            "mood": ["distracted", "sceptical"],
        }
    )
    assert archetype["delivery"] in spoken
    assert prompts_common._PRESSURE_DELIVERY[3] in spoken
    assert "distracted and sceptical" in spoken


def test_tts_instructions_survive_a_profile_that_predates_them():
    """Stored sessions and thin planner output must cost tone, not the interview."""
    assert prompts_common.tts_instructions({}) == prompts_common.TTS_BASE_INSTRUCTIONS
    # An archetype the pool no longer has, and no pressure or mood recorded.
    assert prompts_common.tts_instructions({"archetype": "the retired one"}) == (
        prompts_common.TTS_BASE_INSTRUCTIONS
    )
    no_mood = prompts_common.tts_instructions(
        {"archetype": "the quiet one", "mood": []}
    )
    assert "feeling" not in no_mood


def test_recent_scenarios_lists_the_last_rounds_newest_first():
    previous = [
        _session(profile={"curveball": "an old one"}),
        _session(profile={}),  # a round that recorded no scenario
        _session(profile={"curveball": "the newest one"}),
    ]
    assert prompts_common.recent_scenarios(previous) == ["the newest one", "an old one"]
    assert prompts_common.recent_scenarios([]) == []


def test_pick_mood_draws_two_distinct_adjectives():
    import random

    pool = ["brisk", "tired", "curious", "impatient", "warm"]
    mood = prompts_common.pick_mood(pool, rng=random.Random(3))
    assert len(mood) == 2
    assert len(set(mood)) == 2
    assert set(mood) < set(pool)
    assert prompts_common.pick_mood(pool, rng=random.Random(3)) == mood


def test_pick_mood_needs_two_genuinely_different_adjectives():
    """A pool the planner filled with one repeated word is not a choice."""
    assert prompts_common.pick_mood([]) == []
    assert prompts_common.pick_mood(["brisk"]) == []
    assert prompts_common.pick_mood(["brisk", "Brisk ", ""]) == []


def test_the_session_clock_and_the_mood_pool_reach_the_planner():
    """Wall-clock time is the one genuinely varying input the plan prompt has."""
    from datetime import datetime

    prompt, _ = prompts_preparation.get_plan_prompt(
        SETTINGS,
        "",
        prompts_common.pick_round_profile([]),
        now=datetime(2026, 8, 7, 17, 45),
    )
    assert "Friday at 17:45" in prompt
    assert '"mood_adjectives"' in prompt
    assert "must not all be positive" in prompt


def test_plan_prompt_asks_the_model_for_a_scenario_and_avoids_recent_ones():
    profile = prompts_common.pick_round_profile([])  # carries no "curveball" key
    prompt, _ = prompts_preparation.get_plan_prompt(
        DEEP_SETTINGS, "", profile, "", ["the pipeline broke"]
    )
    assert '"curveball"' in prompt
    assert "the pipeline broke" in prompt
    assert "does not have to be technical" in prompt


def test_a_screening_round_is_asked_for_no_scenario_and_shown_no_old_ones():
    """The recent-scenarios list goes with the bullet. Left in on its own it is
    four more situations to build questions around — the opposite of its job."""
    profile = prompts_common.pick_round_profile([])
    prompt, _ = prompts_preparation.get_plan_prompt(
        SETTINGS, "", profile, "", ["the pipeline broke"]
    )
    assert '"curveball"' not in prompt
    assert "the pipeline broke" not in prompt
    assert "SITUATIONAL SCENARIOS USED IN RECENT ROUNDS" not in prompt


def test_the_planner_must_write_the_role_before_the_round_built_on_it():
    """Key order is generation order: the checklist and the questions are both
    told to derive from the interviewer's own job, so a job written after them
    would be post-hoc narration rather than the thing they were built from."""
    keys = list(prompts_preparation.plan_schema(True)["properties"])
    assert keys.index("culture_anchors") < keys.index("interviewer_role")
    assert keys.index("interviewer_role") < keys.index("mood_adjectives")
    assert keys.index("interviewer_role") < keys.index("checklist")
    assert keys.index("checklist") < keys.index("curveball") < keys.index("questions")
    # additionalProperties is False, so required has to name every property.
    assert set(prompts_preparation.PLAN_SCHEMA["required"]) == set(
        prompts_preparation.PLAN_SCHEMA["properties"]
    )


def test_a_stage_with_no_scenario_drops_the_field_rather_than_emptying_it():
    """Asking for a string the round has no use for gets a string back — the same
    reason `evaluation_schema` removes progress_notes instead of asking for "".
    And `required` has to keep naming every property, so dropping it from there
    alone would leave a property the model must return."""
    schema = prompts_preparation.plan_schema(False)
    assert "curveball" not in schema["properties"]
    assert "curveball" not in schema["required"]
    assert set(schema["required"]) == set(schema["properties"])
    # Nothing else moves, and the order that survives is the load-bearing part.
    full = [
        k
        for k in prompts_preparation.plan_schema(True)["properties"]
        if k != "curveball"
    ]
    assert list(schema["properties"]) == full


# --- who runs the round, and what it is for --------------------------------


def test_every_offered_stage_has_a_profession():
    """The picker's stages and the professions are one list in two places."""
    for stage in STAGES:
        assert (
            prompts_common.resolve_profession(stage)
            is prompts_common.INTERVIEWER_PROFESSIONS[stage]
        )


def test_who_the_interviewer_is_leads_both_prompts():
    """The app does not have one interviewer moving through a process — it has a
    different trained professional in every round, and that is the most important
    context either prompt gets.

    It also replaces the generic identity the interviewer prompt used to open on
    ("You play an experienced interviewer at Acme") rather than sitting under it:
    a concrete exemplar outranks anything specific stated further down, which is
    the same failure the opening's worked example caused.
    """
    recruiter = prompts_common.INTERVIEWER_PROFESSIONS["HR Screen"]["profession"]
    manager = prompts_common.INTERVIEWER_PROFESSIONS["Hiring Manager"]["profession"]

    plan_prompt, _ = prompts_preparation.get_plan_prompt(
        SETTINGS, "", prompts_common.pick_round_profile([])
    )
    assert plan_prompt.startswith("WHO THE INTERVIEWER IS")
    assert recruiter in plan_prompt
    # Everything the planner writes is preparation for that person.
    assert plan_prompt.index(recruiter) < plan_prompt.index("WHAT THIS ROUND IS FOR")

    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"], "Hi.")
    system = prompts_interview.get_interviewer_system_prompt(
        SETTINGS, plan, "", prompts_common.pick_round_profile([]), answered=1
    )
    assert system.startswith("WHO YOU ARE")
    assert recruiter in system
    assert "You play an experienced interviewer" not in system

    # A different stage is a different professional, not the same one relabelled.
    deep = prompts_interview.get_interviewer_system_prompt(
        DEEP_SETTINGS, plan, "", prompts_common.pick_round_profile([]), answered=1
    )
    assert manager in deep
    assert recruiter not in deep


def test_the_profession_is_the_job_and_never_the_vacancy():
    """`settings.role` may be any job at all, so a hardcoded engineering manager
    here would run a marketing round with the wrong person in the chair."""
    for stage, profession in prompts_common.INTERVIEWER_PROFESSIONS.items():
        text = profession["profession"].lower()
        for hardcoded in ("software", "engineering manager", "codebase", "developer"):
            assert hardcoded not in text, stage


def test_an_unknown_stage_costs_the_steering_and_not_the_interview():
    """A stored session, a hand-edited settings file, a future addition to STAGES."""
    profession = prompts_common.resolve_profession("Coffee Chat")
    assert profession is prompts_common._GENERIC_PROFESSION
    # Falls back to what the app did before the constant existed.
    assert profession["curveball"] and profession["culture_probe"]
    assert prompts_common.round_purpose_block(profession)
    assert prompts_common.profession_block(profession)


def test_the_interviewers_job_is_reasoned_out_and_never_handed_over_as_a_list():
    """The whole point of the field. A worked example would be copied every
    round for every stage, and would be wrong for most of them."""
    prompt, _ = prompts_preparation.get_plan_prompt(
        InterviewSettings("Acme", "Engineer", "Case Study"),
        "",
        prompts_common.pick_round_profile([]),
    )
    # Scoped to the bullet itself: item 1 legitimately says "relocation notes"
    # when telling the planner what to keep OUT of the company brief.
    bullet = prompt.split('3. "interviewer_role"')[1].split('4. "mood_adjectives"')[0]
    bullet = bullet.lower()
    for hardcoded in (
        "notice period",
        "relocation",
        "visa",
        "work authorisation",
        "work authorization",
        "salary",
        "compensation expectation",
        "start date",
        "reason for leaving",
    ):
        assert hardcoded not in bullet


def test_standing_questions_may_recur_where_exploratory_ones_may_not():
    """A question whose value is being unexpected is spent once; a question the
    job puts on this interviewer is asked of everyone, every time."""
    digest = prompts_preparation.history_digest([_session()])
    prompt, _ = prompts_preparation.get_plan_prompt(
        SETTINGS, digest, prompts_common.pick_round_profile([])
    )
    assert "The exploratory questions must be new" in prompt
    assert "asks it of everyone, every time" in prompt


def test_the_wrap_up_turn_spends_itself_on_a_read_before_a_fact():
    """The round spent its wrap-up on another logistics question while a
    checklist criterion still had nothing on it. A fact can be settled in writing
    after the call; a judgement about the candidate cannot."""
    wrap_up = _flat(prompts_interview._PHASE_INSTRUCTIONS["wrap_up"])
    assert "An unrated checklist item comes first" in wrap_up
    assert "a read on the candidate cannot" in wrap_up
    assert wrap_up.index("unrated checklist item") < wrap_up.index("your own job")


def test_the_question_pool_is_a_running_order_for_the_slot():
    """The pool is the round's running order, so its length is the slot's — about
    three minutes a question — plus the couple the interviewer drops when an
    answer is worth pressing on. Never fewer than the criteria, though: a
    criterion no question reaches cannot be rated at all."""
    for duration in (10, 15, 30, 45, 60):
        items_lo, items_hi, questions_lo, questions_hi = (
            prompts_preparation._plan_sizes(duration)
        )
        assert questions_lo == max(items_hi, prompts_common.answer_budget(duration))
        assert questions_hi == questions_lo + 2
        assert questions_lo >= items_hi
    assert prompts_preparation._plan_sizes(15)[2:] == (5, 7)
    assert prompts_preparation._plan_sizes(60)[2:] == (20, 22)


def test_the_planner_orders_the_round_and_the_scorecard_apart():
    """Conversation order for the questions, importance for the checklist: the
    interviewer follows the one and cuts by the other."""
    profile = prompts_common.pick_round_profile([])
    prompt, _ = prompts_preparation.get_plan_prompt(SETTINGS, "", profile)
    flat = _flat(prompt)
    assert "ORDER THEM AS THE CONVERSATION WOULD RUN" in flat
    assert "ORDER THEM BY IMPORTANCE, the most load-bearing criterion" in flat
    assert flat.index("ORDER THEM BY IMPORTANCE") < flat.index(
        "ORDER THEM AS THE CONVERSATION WOULD RUN"
    )
    assert "COVER THE CHECKLIST, in the order of importance you just gave it" in flat
    assert "A criterion no question reaches cannot be rated at all" in flat


def test_curveball_domains_are_all_offered_in_a_fresh_order_each_round():
    """Every domain stays on offer; only the order the planner reads them in moves."""
    import random

    orders = {
        prompts_common.curveball_domains(random.Random(seed)) for seed in range(20)
    }
    assert len(orders) > 1
    for order in orders:
        words = order.replace(" or ", ", ").split(", ")
        assert sorted(words) == sorted(prompts_common._CURVEBALL_DOMAINS)


def test_the_permuted_domains_reach_the_plan_prompt():
    import random

    profile = prompts_common.pick_round_profile([])
    prompt, _ = prompts_preparation.get_plan_prompt(
        DEEP_SETTINGS, "", profile, rng=random.Random(1)
    )
    assert (
        f"which may be\n   {prompts_common.curveball_domains(random.Random(1))}."
        in prompt
    )


# --- the closing gate ------------------------------------------------------


def _at_pace(answered: int, duration: int, words_per_answer: int = 0) -> float:
    """Minutes spoken after `answered` answers, nominal pace unless told otherwise.

    Nominal is the whole slot spent over the whole answer budget. A word count
    instead prices each exchange directly — the interviewer's own share of it
    included, since both sides of the table use up the room.
    """
    if not words_per_answer:
        return duration * answered / prompts_common.answer_budget(duration)
    return answered * (words_per_answer + 55) / prompts_common.WORDS_PER_MINUTE


def _closes_at(duration: int, words_per_answer: int = 0) -> int:
    for answered in range(1, 200):
        spoken = _at_pace(answered, duration, words_per_answer)
        if prompts_interview.interview_phase(spoken, answered, duration)[1]:
            return answered
    raise AssertionError("the round never closed")


def test_interview_phase_reserves_a_questions_phase_before_closing():
    budget = prompts_common.answer_budget(30)  # 10 answers at the nominal pace
    at = _at_pace
    assert prompts_interview.interview_phase(at(1, 30), 1, 30) == ("core", False)
    assert prompts_interview.interview_phase(at(budget - 2, 30), budget - 2, 30) == (
        "wrap_up",
        False,
    )
    assert prompts_interview.interview_phase(at(budget, 30), budget, 30) == (
        "questions",
        True,
    )


def test_every_round_gets_a_wrap_up_turn_before_it_may_close():
    """The turn that hands the floor over. One answer long enough to cross both
    thresholds at once must not skip it."""
    for duration in (10, 15, 30, 45, 60):
        for words in (25, 350, 900, 5000):
            phases = []
            for answered in range(1, 200):
                spoken = _at_pace(answered, duration, words)
                phase, may_close = prompts_interview.interview_phase(
                    spoken, answered, duration
                )
                phases.append(phase)
                if may_close:
                    break
            assert "wrap_up" in phases, (duration, words)


def test_talking_longer_costs_questions_and_terseness_earns_them():
    """The point of pacing by words: the slot is spent by what was said in it.
    The guards stop either extreme running away — a monologue cannot close the
    round before there is enough to rate, and one-liners cannot run it forever."""
    for duration in (10, 15, 30, 45, 60):
        floor, ceiling = prompts_interview.answer_guards(duration)
        rambler, typical, terse = (_closes_at(duration, w) for w in (900, 350, 25))
        assert rambler <= typical <= terse
        assert floor <= rambler and terse <= ceiling
    # At 45 minutes the spread is the whole point: 9 answers against 22.
    assert _closes_at(45, 900) == 9
    assert _closes_at(45, 25) == 22


# --- transcripts and transcription ----------------------------------------


def test_transcript_block_flags_only_spoken_answers():
    block = prompts_interview.transcript_block(
        [
            Turn("interviewer", "Why us?"),
            Turn("candidate", "Spoken.", VOICE),
            Turn("candidate", "Typed."),
        ]
    )
    assert "CANDIDATE (spoken, auto-transcribed): Spoken." in block
    assert "CANDIDATE: Typed." in block


def test_transcription_hint_primes_names_and_recent_context():
    hint = prompts_interview.transcription_hint(
        SETTINGS, [Turn("interviewer", "Tell me about Kubernetes at Acme.")]
    )
    assert "Acme" in hint and "Engineer" in hint
    assert "Kubernetes" in hint


# --- history digest --------------------------------------------------------


def test_history_digest_empty_without_previous():
    assert prompts_preparation.history_digest([]) == ""


def test_history_digest_carries_stories_and_weak_spots():
    digest = prompts_preparation.history_digest([_session(evaluation=_evaluation())])
    assert "the re-ranking service" in digest
    assert "Motivation (rated partial)" in digest
    assert "more numbers" in digest


def test_history_digest_dedupes_and_caps_old_questions():
    sessions = [_session(questions=[f"Question {i}", "Why Acme?"]) for i in range(40)]
    digest = prompts_preparation.history_digest(sessions)
    listed = [line for line in digest.splitlines() if line.startswith("- ")]
    assert len(listed) <= prompts_preparation._MAX_PAST_QUESTIONS
    assert digest.count("- Why Acme?") == 1


def test_the_chance_rule_covers_the_criterion_no_question_can_raise():
    """The two halves of one principle, and what a round without the fixed
    criterion renders instead.

    Nothing in a transcript asks for structure, so the half of the rule written
    for topics the interviewer never raised would bury it every round. Sessions
    planned before it existed must take neither the criterion nor its rules.
    """
    ordinary = ChecklistItem("ownership", "Ownership", "what was theirs")
    plan = InterviewPlan(
        checklist=[ordinary, prompts_common.STRUCTURE_CRITERION],
        questions=["Why Acme?"],
    )
    prompt = prompts_interview.get_evaluation_prompt(SETTINGS, plan, [], [])
    assert "ALWAYS RATED" in prompt
    assert 'never "not_assessed" for want of one' in _flat(prompt)
    # Filler is the delivery adjustment, the shape of the answer is the
    # criterion, and one habit must not be charged to both.
    assert "not charged to both" in _flat(prompt)

    older = InterviewPlan(checklist=[ordinary], questions=["Why Acme?"])
    fixed_id = f"[{prompts_common.STRUCTURE_CRITERION.id}]"
    for prompt in (
        prompts_interview.get_evaluation_prompt(SETTINGS, older, [], []),
        prompts_interview.get_interviewer_system_prompt(SETTINGS, older, "", {}, 3),
    ):
        assert "ALWAYS RATED" not in prompt
        assert fixed_id not in prompt


def test_core_checklist_block_pins_ids_across_rounds():
    block = prompts_preparation.core_checklist_block([_session()])
    assert "[motivation] Motivation" in block
    assert "SAME ids" in block
    assert prompts_preparation.core_checklist_block([]) == ""


# --- schema and formatting -------------------------------------------------


def test_wrapping_quotes_are_stripped_from_prose_fields():
    evaluation = Evaluation.from_dict(
        {
            "score": 70,
            "results": [],
            "strengths": [],
            "improvements": [],
            "summary": '"Solid."',
        }
    )
    assert evaluation.summary == "Solid."


def test_rubric_score_ignores_thin_coverage():
    assessed = Evaluation(
        score=0,
        results=[
            ChecklistResult("a", "A", "met"),
            ChecklistResult("b", "B", "partial"),
        ],
        strengths=[],
        improvements=[],
        summary="",
    )
    assert assessed.rubric_score() == 75

    thin = Evaluation(
        score=0,
        results=[ChecklistResult("a", "A", "met")]
        + [ChecklistResult(str(i), "X", "not_assessed") for i in range(3)],
        strengths=[],
        improvements=[],
        summary="",
    )
    assert thin.rubric_score() is None  # too little assessed to anchor to


def test_evaluation_judges_the_questions_the_candidate_asked():
    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"])
    prompt = prompts_interview.get_evaluation_prompt(
        SETTINGS,
        plan,
        [Turn("candidate", "Do you offer home office?")],
        [],
    )
    assert "JUDGE THE QUESTIONS THE CANDIDATE ASKED" in prompt
    # Content, not count, and never a penalty for a floor they never got.
    assert "never the number of them" in prompt
    assert 'rate this "not_assessed"' in prompt
    # The closing questions feed the same +/-10 adjustment, not a new axis.
    assert "as recorded in candidate_questions" in prompt


def test_an_opening_the_candidate_was_never_invited_into_is_still_theirs():
    """The counterweight to the fairness rule, and the gap it left. Run over a
    real recruiter screen, the evaluator saw a stated salary band go by without
    a word from the candidate and excused it — "he did not ask for one" — while
    the human debrief of the same call opened with it. Nobody invites a
    candidate to take these moments; that is what makes them the expensive ones.
    """
    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"])
    prompt = prompts_interview.get_evaluation_prompt(SETTINGS, plan, [], [])
    flat = _flat(prompt)
    assert "a chance without being questions" in flat
    assert "Something the interviewer stated rather than asked" in flat
    assert "Judge what they did with it" in flat
    assert "including saying nothing" in flat
    # Narrow on purpose: it does not reopen everything the interviewer skipped.
    assert "This is a narrow list, not a licence" in flat
    assert "interviewer never raised is still not_assessed" in flat
    # The floor is the candidate's once it is open, and the interviewer's before.
    assert "the candidate closed it themselves" in flat
    assert "not a quota" in flat
    # And the rule it qualifies is untouched.
    assert "RATE WHAT THE CANDIDATE HAD THE CHANCE TO SHOW" in prompt
    assert "never the number of them" in prompt


def test_a_criterion_nobody_asked_about_stays_not_assessed_with_evidence_lying_around():
    """The failure: a screen never asked why this company, the candidate said
    something about it while defending a different challenge, and the evaluator
    rated the criterion "partial" off that — then made the manner of its arrival
    the criticism ("it arrived as a defence under challenge rather than as
    motivation he brought"). Both halves mark the candidate down for the
    interviewer's running order, and the second drags rubric_score() with it."""
    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"])
    flat = _flat(prompts_interview.get_evaluation_prompt(SETTINGS, plan, [], []))
    assert "however much incidental evidence turned up" in flat
    assert "not a weak version of the answer they were never asked for" in flat
    assert "never make the incidental arrival itself the finding" in flat
    assert "a fact about the interviewer's running order" in flat
    # And it must not eat the carve-out, which is the counterweight to it.
    assert "they are the whole of the list" in flat
    assert "do not read the incidental-evidence rule as narrowing these four" in flat


def test_the_candidates_own_terms_are_not_a_confidence_they_may_keep():
    """CONFIDENTIALITY_RULE protects the candidate's employers. Reaching the
    evaluator without this, it also excused vagueness about the candidate's own
    availability and expectations — which is the one thing a screen exists to
    settle, and the planner already carries the same carve-out."""
    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"])
    prompt = prompts_interview.get_evaluation_prompt(SETTINGS, plan, [], [])
    flat = _flat(prompt)
    assert "This protects the candidate's employers and nobody else." in flat
    assert "theirs to state, and vagueness about those is a finding" in flat


def test_a_why_us_answer_that_would_fit_any_employer_is_weak():
    """Eight recorded real screens, eight rounds in which nothing company-
    specific was said — the most repeated finding in the candidate's own
    debriefs, and one the evaluator had no rule for."""
    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"])
    prompt = prompts_interview.get_evaluation_prompt(SETTINGS, plan, [], [])
    flat = _flat(prompt)
    assert "could have been said about any employer in the same market" in flat
    assert "Where it stayed generic, say so and quote it" in flat
    # Still bounded by the fairness rule: only where the question was asked.
    assert "never marked down for not knowing a fact nobody asked for" in flat


def test_delivery_is_the_candidates_and_the_noise_is_the_recognisers():
    """The caveat used to file filler words, false starts and run-on sentences
    with mis-transcribed names, and the evaluator is told never to list an
    artifact as an improvement — so the one finding every human debrief of these
    calls carries ("basically" twenty times) was ruled out by construction. A
    recogniser mangles names; it does not add hedges."""
    assert "filler" not in prompts_interview.TRANSCRIPTION_CAVEAT
    assert "false starts" not in prompts_interview.TRANSCRIPTION_CAVEAT
    assert (
        "homophones, mangled technical terms" in prompts_interview.TRANSCRIPTION_CAVEAT
    )

    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"])
    prompt = prompts_interview.get_evaluation_prompt(SETTINGS, plan, [], [])
    flat = _flat(prompt)
    assert "What a recogniser does not invent is how the candidate speaks." in flat
    # A pattern, not a tally, and it moves the delivery adjustment rather than
    # the checklist — the score still comes from the criteria alone.
    assert "only as a pattern, never as one instance and never as a tally" in flat
    assert "never turns a met criterion into a partial one" in flat

    # Evaluator-only: the live interviewer is told not to react to garbled input,
    # and must not start noticing how the candidate speaks either.
    system = prompts_interview.get_interviewer_system_prompt(
        SETTINGS, plan, "", prompts_common.pick_round_profile([]), answered=1
    )
    assert "how the candidate speaks" not in system


def test_candidate_questions_are_scored_before_the_score():
    """Structured output is generated in key order, so the read must come first.

    The prompt requires the score to be consistent with this pass; a model can
    only honour that if it has already written it.
    """
    for include_progress_notes in (True, False):
        keys = list(
            prompts_interview.evaluation_schema(include_progress_notes)["properties"]
        )
        assert keys.index("candidate_questions") < keys.index("score")
        assert keys.index("answer_review") < keys.index("candidate_questions")
        assert ("progress_notes" in keys) is include_progress_notes


def test_evaluation_asks_for_one_takeaway_that_may_repeat_the_last_round():
    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"])
    previous = _evaluation(
        key_takeaway=KeyTakeaway("Be specific: name the number", 2, "e", "v")
    )
    prompt = prompts_interview.get_evaluation_prompt(
        SETTINGS, plan, [Turn("candidate", "We improved things a lot.")], [previous]
    )
    assert "THE ONE THING TO TAKE AWAY" in prompt
    assert "the reason not to hire" in prompt
    # Saying it again is the finding, not a failure to find something new.
    assert "Repeat it when it is still the answer" in prompt
    assert "the one thing they were told to fix: Be specific: name the number" in prompt
    # Always filled in, and never a second scoring axis.
    assert "There is always one." in prompt
    assert 'changes nothing about "score"' in prompt
    # A habit shown three times beats one weak moment.
    assert "It need not come from one answer" in prompt


def test_the_takeaway_is_spoken_to_the_candidate_and_read_off_the_transcript():
    """The whole callout is one voice, and it is built from the turns, not the findings.

    Both halves of one real failure: a verdict written for a colleague ("he accepted
    it and moved on") rendered directly under a point written to the candidate, and
    every one of its three examples clipped to the fragment the improvement above it
    had already compressed it to, dropping the rest of the candidate's turn.
    """
    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"])
    prompt = prompts_interview.get_evaluation_prompt(
        SETTINGS, plan, [Turn("candidate", "That is fair enough.")], []
    )
    assert "The whole takeaway is one person speaking to one person." in prompt
    assert (
        "Take it from the transcript, not from the findings you have just written"
        in prompt
    )
    assert "read each one to its end" in prompt
    # It may land on the same finding as the sharpest improvement, but not by copying it.
    assert "because both are true of the same interview" in prompt


def test_every_field_of_the_evaluation_is_written_to_the_candidate():
    """One voice, and the per-field markers are what broke it.

    A real evaluation read as two authors: the four fields the prompt marked as
    addressed to the candidate came back as "you", and everything else — the
    strengths, the evidence, the question-by-question comments — came back as
    "he", because marking four fields says the other eight are exempt.
    """
    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"])
    prompt = prompts_interview.get_evaluation_prompt(
        SETTINGS, plan, [Turn("candidate", "That is fair enough.")], [_evaluation()]
    )
    flat = _flat(prompt)
    assert "ONE VOICE: WRITE IT TO THE CANDIDATE" in prompt
    assert 'every string in it is spoken to them as "you"' in flat
    # The other person in the transcript is not re-voiced with them.
    assert "The interviewer stays in the third person throughout" in flat
    # Nor are the four things that are quoted or copied rather than written.
    assert "keeps the candidate's own words" in flat
    assert '"criterion" is copied from the checklist above exactly as written' in flat
    assert "in the wording it was put in" in flat
    assert '"topics_covered" is a list of plain noun phrases with no pronoun' in flat

    # And no field bullet re-scopes the rule to itself, which is the bug coming
    # back one field at a time. progress_notes is in here too, hence the history.
    produce = _flat(prompt.split("\nProduce:\n")[1])
    assert "addressed to the candidate" not in produce
    assert 'written to the candidate ("you")' not in produce


def test_the_takeaway_is_chosen_after_the_findings_it_chooses_from():
    """Key order again: it is a pick among the improvements, so they come first."""
    for include_progress_notes in (True, False):
        keys = list(
            prompts_interview.evaluation_schema(include_progress_notes)["properties"]
        )
        assert keys.index("improvements") < keys.index("key_takeaway")
        assert keys.index("results") < keys.index("key_takeaway")
        fields = prompts_interview._EVALUATION_PROPERTIES["key_takeaway"]["properties"]
        assert list(fields)[:2] == ["point", "severity"]
        assert fields["severity"]["enum"] == TAKEAWAY_SEVERITIES


def test_the_confidentiality_rule_reaches_all_three_prompts():
    """Each prompt demands the leak on its own, so each one has to be told.

    The planner writes criteria only a leak satisfies ("a named counterparty"),
    the interviewer turns that into "name the person", and the evaluator marks
    the absence down — which history_digest then feeds into the next round.
    """
    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"], "Hi.")
    prompt, _ = prompts_preparation.get_plan_prompt(
        SETTINGS, "", prompts_common.pick_round_profile([])
    )
    system = prompts_interview.get_interviewer_system_prompt(
        SETTINGS, plan, "", prompts_common.pick_round_profile([]), answered=1
    )
    evaluation = prompts_interview.get_evaluation_prompt(SETTINGS, plan, [], [])

    for text in (prompt, system, evaluation):
        assert "names of colleagues, managers and" in text
        assert "monetised value" in text
    # The interviewer carries it as a bullet, so it has to survive the indent.
    assert "\n  clients, and absolute figures" in system
    assert "Never press for a name or an exact figure." in system
    assert "Rate the specificity they could legitimately give" in evaluation


def test_interviewer_prompt_bans_praise_and_name_corrections():
    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"], "Hi.")
    system = prompts_interview.get_interviewer_system_prompt(
        SETTINGS, plan, "", prompts_common.pick_round_profile([]), answered=1
    )
    assert "Do NOT praise" in system
    assert "never repeat a garbled name back" in system


def test_the_interviewers_job_reaches_them_live_and_survives_its_absence():
    """Improvised follow-ups and end-of-round triage have to serve it too, and a
    session stored before the field existed must drop the block, not the round."""
    plan = InterviewPlan(
        [ChecklistItem("m", "Motivation", "d")],
        ["Why Acme?"],
        "Hi.",
        interviewer_role="You are the recruiter. You must know when they can start.",
    )
    system = prompts_interview.get_interviewer_system_prompt(
        SETTINGS, plan, "", prompts_common.pick_round_profile([]), answered=1
    )
    assert "You must know when they can start." in system
    assert "is yours to get" in system
    # It decides how the round is asked, not just what is left over at the end of
    # it: scoped to end-of-round triage, it read as inapplicable on every turn
    # that still had time on the clock — the opening above all.
    assert "governs the whole round, not the end of it" in system
    flat = _flat(system)
    # A running order it keeps to loosely: it presses where an answer earns it,
    # and cuts by the checklist rather than by position when the clock is short.
    assert "in roughly the order this round should run" in flat
    assert "it is a running order, not a script" in flat
    assert "past the questions whose checklist item you have already rated" in flat

    old = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"], "Hi.")
    assert (
        "Your own job in this process"
        not in prompts_interview.get_interviewer_system_prompt(
            SETTINGS, old, "", prompts_common.pick_round_profile([]), answered=1
        )
    )


def test_the_opening_is_derived_from_the_round_and_not_from_an_example():
    """The bug this whole change came out of. The opening carried its own worked
    example of what to say, so every stage said it: an HR screen opened on "what
    you actually work on and how you operate day to day" — a competency round's
    line — while the job description that said otherwise sat further up the
    prompt, scoped to end-of-round triage.

    A concrete exemplar inside the instruction that writes the turn outranks
    anything general stated elsewhere, so the exemplar had to go rather than be
    balanced against.
    """
    plan = InterviewPlan(
        [ChecklistItem("m", "Motivation", "d")],
        ["Why Acme?"],
        "Hi.",
        interviewer_role="You are the recruiter. You must know when they can start.",
    )
    opening = prompts_interview.get_interviewer_system_prompt(
        SETTINGS, plan, "", prompts_common.pick_round_profile([]), answered=0
    )
    assert "background and how you work" not in opening
    assert "what THIS round is for" in opening
    # What replaces it: the stage's own purpose, and the plain-words permission
    # that DISCRETION_RULE otherwise argues the model out of.
    assert prompts_common.INTERVIEWER_PROFESSIONS["HR Screen"]["purpose"] in opening
    assert "gives nothing away" in opening
    # Still not a rubric read-out, and still exactly one question.
    assert "never a list of what you are assessing" in opening
    assert "do not stack a second onto it" in opening

    deep = prompts_interview.get_interviewer_system_prompt(
        DEEP_SETTINGS, plan, "", prompts_common.pick_round_profile([]), answered=0
    )
    assert prompts_common.INTERVIEWER_PROFESSIONS["HR Screen"]["purpose"] not in deep


def test_the_opening_question_is_the_candidates_own_ground():
    """A pressure-1 devil's advocate opened a real screen on the employment gap
    in the CV — the sharpest thing in the file, before the candidate had said a
    word. The persona owns how hard the round presses; it does not own turn one.

    Stated as a property of the question, never as an example of one: an exemplar
    here is what the previous fix had to remove.
    """
    plan = InterviewPlan(
        [ChecklistItem("m", "Motivation", "d")],
        ["Why Acme?"],
        "Hi.",
        interviewer_role="You are the recruiter. You must know when they can start.",
    )
    opening = _flat(
        prompts_interview.get_interviewer_system_prompt(
            SETTINGS, plan, "", prompts_common.pick_round_profile([]), answered=0
        )
    )
    assert "out of their own experience and preparation" in opening
    assert "without having to defend anything first" in opening
    assert "none of that starts on the first question" in opening
    assert (
        "the sharpest thing in front of you is not where a real interview begins"
        in (opening)
    )
    # The order is the conversation's, so question 1 is the usual opener — but
    # the persona still decides, and this is the one turn that says so.
    assert "the first of them is normally where you begin" in opening
    assert "would really open somewhere else" in opening
    # And what the opening sentence promises, the round then owes.
    assert "it is a promise the round has to keep" in opening

    # Still a property and not a worked example — no stage-blind opening line.
    for exemplar in (
        "background and how you work",
        "tell me about yourself",
        "walk me through your cv",
        "why this role",
    ):
        assert exemplar not in opening.lower()


def test_the_round_is_run_the_way_its_stage_is_run_on_every_turn():
    """Not just the opening: the pace and depth belong to the stage, and a
    persona drawn without reference to it would otherwise set them alone."""
    plan = InterviewPlan([ChecklistItem("m", "Motivation", "d")], ["Why Acme?"])
    for answered in (0, 1, 4):
        system = prompts_interview.get_interviewer_system_prompt(
            SETTINGS, plan, "", prompts_common.pick_round_profile([]), answered=answered
        )
        assert prompts_common.INTERVIEWER_PROFESSIONS["HR Screen"]["practice"] in system


def test_the_candidates_questions_phase_closes_the_interviewers_own():
    """The pool is larger than the slot, so the phase has to say plainly that
    unasked questions stay unasked — otherwise the gate reads as inapplicable."""
    instructions = prompts_interview._PHASE_INSTRUCTIONS
    assert "Do NOT open a new topic of your own" in instructions["questions"]
    assert "however many are still unasked" in instructions["questions"]
    # "the last one you get" overstated it: at the nominal pace two answers land
    # in wrap_up, and a round that believes it has less room than it has drops a
    # question it had time for. How many there really are depends on how much has
    # been said by then, which is why the wording covers both.
    assert "one or two left at most" in instructions["wrap_up"]
    budget = prompts_common.answer_budget(30)
    wrap_up = [
        a
        for a in range(1, budget + 1)
        if prompts_interview.interview_phase(_at_pace(a, 30), a, 30)[0] == "wrap_up"
    ]
    assert len(wrap_up) == 2


def test_research_prompt_stays_a_company_briefing_not_a_candidate_assessment():
    """The delta from a general job-application research prompt: no candidate."""
    prompt = prompts_preparation.get_company_research_prompt(
        "Acme", "ML Engineer", "The telemetry team.", now=datetime(2026, 8, 5)
    )
    assert "Research Acme and write" in prompt
    assert "ML Engineer" in prompt
    assert "The telemetry team." in prompt
    assert "05 August 2026" in prompt

    assert "not an assessment of a candidate" in prompt
    assert "must not ask for one or imagine one" in prompt
    assert "No fit scores" in prompt
    assert "Do not rank several open roles against anybody" in prompt
    # It tells the model never to address a reader, so outside the sentence that
    # bans those words the prompt must not use them either.
    without_the_ban = prompt.replace(
        '"the candidate", "the reader",\n  "your background" or "your experience"', ""
    )
    for banned in ("the reader", "your background", "the candidate", "you are"):
        assert banned not in without_the_ban


def test_research_prompt_omits_the_angle_when_no_role_is_given():
    prompt = prompts_preparation.get_company_research_prompt(
        "Acme", now=datetime(2026, 8, 5)
    )
    assert "ANGLE:" not in prompt
    assert "ASKED FOR SPECIFICALLY" not in prompt
    assert "# Acme — Company Briefing" in prompt


def test_research_brief_prompt_carries_the_request_verbatim():
    prompt = prompts_preparation.get_research_brief_prompt(
        "Acme, ML engineer, telemetry team"
    )
    assert "Acme, ML engineer, telemetry team" in prompt
    assert set(prompts_preparation.COMPANY_RESEARCH_BRIEF_SCHEMA["required"]) == {
        "company",
        "role",
        "focus",
    }
