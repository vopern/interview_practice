"""Everything from the first turn onwards: running the round, and rating it.

`get_interviewer_system_prompt` conducts the interview `prompts_preparation`
planned, and `get_evaluation_prompt` rates the transcript that came out of it.
They share a module because they share the material only they need — the caveat
over machine-transcribed answers, and the fairness invariant, which governs both
what the interviewer has to give the candidate room to show and what the
evaluator may then hold against them. The round profile and the rules the planner
obeys as well are `prompts_common`.

The evaluation prompt's docstring documents the exact JSON contract expected
back; the matching schema constant is enforced server-side via structured
outputs, so no client-side JSON repair is needed.

Two of the four cross-cutting concerns handled in Python rather than left to the
model's judgement live here:

* **Transcription noise** — voice answers reach the evaluator with ASR errors in
  them, and `TRANSCRIPTION_CAVEAT` stops those being scored as candidate
  mistakes. It covers what the recogniser *invents* — names, numbers,
  homophones, punctuation — and deliberately not how the candidate speaks:
  hedges and filler are in the transcript because they were said, and the
  evaluator is told separately how to treat them (see `get_evaluation_prompt`).
* **Opportunity fairness** — the candidate can only be marked down for what the
  interviewer actually gave them room to show. The interviewer prompt reserves a
  real questions phase, and the evaluator is told to check the transcript for
  the opening before criticising an omission.

The other two are properties of the round rather than of the transcript, so they
sit with the material that draws one, in `prompts_common`.
"""

from src.core.prompts_common import (
    CONFIDENTIALITY_RULE,
    DISCRETION_RULE,
    STRUCTURE_CRITERION,
    answer_budget,
    profession_block,
    profile_block,
    resolve_profession,
    round_purpose_block,
)
from src.models.interview import (
    INTERVIEWER,
    VOICE,
    Evaluation,
    InterviewPlan,
    InterviewSettings,
    Turn,
)

# Sentinel the interviewer appends when every checklist item is covered.
# The manager strips it from the visible reply and flips the UI state.
COMPLETE_TOKEN = "[INTERVIEW_COMPLETE]"

# Shared by the interviewer (who must not react to garbled input) and the
# evaluator (who must not score it).
TRANSCRIPTION_CAVEAT = """\
The candidate may be answering by voice: their turns are machine-transcribed and
contain speech-recognition errors. Treat every proper noun (company, product,
team, person, paper, library and tool names), every acronym and every number as
potentially mis-transcribed. A wrong company or product name in the transcript is
a transcription artifact, not evidence about the candidate. The same goes for
homophones, mangled technical terms and missing punctuation — those are how speech
transcribes, not how the candidate thinks. Read through the noise to the intended
meaning and judge content, structure and reasoning rather than the transcript's
surface."""


def split_checklist(plan: InterviewPlan) -> tuple[list, object]:
    """The criteria a question has to raise, and the one that needs none.

    Both prompts render the two apart, so the rule that decides whether a
    criterion can be rated at all arrives with the criterion rather than
    paragraphs away. Sessions planned before the fixed criterion existed return
    None for it, and both prompts then drop the block and its rule.
    """
    asked_for = [c for c in plan.checklist if c.id != STRUCTURE_CRITERION.id]
    fixed = next((c for c in plan.checklist if c.id == STRUCTURE_CRITERION.id), None)
    return asked_for, fixed


# ---------------------------------------------------------------------------
# SECTION 1: CONDUCTING THE INTERVIEW
# ---------------------------------------------------------------------------


# The share of the slot after which the interviewer's own questions are over,
# and the band the answer count is allowed to hold the word clock inside.
_WRAP_UP_AT = 0.8
_MIN_ANSWERS_SHARE = 0.6
_MAX_ANSWERS_SHARE = 1.5


def answer_guards(duration_minutes: int) -> tuple[int, int]:
    """Fewest and most candidate answers a round may run to, whatever the pace.

    The band the word clock decides inside. Public because the closing gate is
    only testable against it: how many answers a round takes now depends on how
    much was said in them.
    """
    budget = answer_budget(duration_minutes)
    floor = max(3, round(_MIN_ANSWERS_SHARE * budget))
    return floor, max(floor + 1, round(_MAX_ANSWERS_SHARE * budget))


def interview_phase(
    spoken: float, answered: int, duration_minutes: int
) -> tuple[str, bool]:
    """(phase, may_close) for the answer the interviewer is about to respond to.

    Paced by `spoken` — the minutes `prompts_common.spoken_minutes` estimates the
    conversation has talked away — so a candidate who monologues really does run
    the slot down faster than one who answers in a line. `answered` counts
    candidate answers including the current one, and guards the estimate at both
    ends: below the floor the round stays open however much was said, at the
    ceiling it ends however little was. Without the floor a long enough opening
    answer would close the interview on its own and leave a checklist nobody
    could rate; without the ceiling a candidate answering in three words would
    never reach the end of one.

    Closing is gated here rather than left to the model, which otherwise closes
    in the same breath as answering the candidate's first question — and the
    candidate is then scored on a phase they never got.
    """
    floor, ceiling = answer_guards(duration_minutes)
    used = spoken / duration_minutes if duration_minutes > 0 else 1.0
    # Where the previous turn stood, so one very long answer crossing both
    # thresholds at once still lands on wrap_up before closing: the interviewer
    # needs the turn that hands the floor over.
    before = used * (answered - 1) / answered if answered > 0 else 0.0
    if answered >= ceiling:
        return "questions", True
    if used >= 1.0 and answered >= floor and before >= _WRAP_UP_AT:
        return "questions", True
    if answered >= ceiling - 1 or used >= _WRAP_UP_AT:
        return "wrap_up", False
    return "core", False


_PHASE_INSTRUCTIONS = {
    "core": (
        "You are working through your prepared questions in their order. Press again where "
        "the last answer has not settled the checklist item it was there for, or where it "
        "opened something worth pursuing; otherwise move on to the next question. Where "
        "you are behind the clock, skip forward over the questions whose criterion you can "
        "already rate. Do NOT invite the candidate's questions yet, and do NOT end the "
        "interview this turn."
    ),
    "wrap_up": (
        "You are nearly out of time for your own questions: one or two left at most. Spend "
        "them on what only this conversation can settle. An unrated checklist item comes "
        "first — a fact about the candidate's situation can still be settled in writing "
        "after this call, a read on the candidate cannot — then whatever your own job "
        "still leaves unsettled, and never a question you would merely like to have asked. "
        "Where both are open, take the criterion your checklist puts highest. If nothing "
        "like that is left, tell the candidate you'd like to leave the rest of the time "
        "for them and invite their questions. You may NOT end the interview this turn "
        "under any circumstances."
    ),
    "questions": (
        "You are in the candidate's-questions phase and your own questions are over. Do "
        "NOT open a new topic of your own, however many are still unasked in your list — "
        "the rest of the time belongs to the candidate, and ending with questions unasked "
        "is how real interviews end. If they asked something, answer it briefly — a few "
        "sentences, not a monologue — and then explicitly ask what else they would like "
        "to know. Only when the candidate has said they have nothing further may you "
        "close. Never answer a question and close in the same message: the candidate must "
        "get at least one more turn after every answer you give."
    ),
}


def get_interviewer_system_prompt(
    settings: InterviewSettings,
    plan: InterviewPlan,
    digest: str,
    profile: dict,
    answered: int,
    spoken: float = 0.0,
) -> str:
    """System prompt for the live interviewer turns. Response is plain text.

    `answered` is the number of candidate answers including the one being
    responded to, and `spoken` the minutes `prompts_common.spoken_minutes`
    estimates have been talked away so far; together they drive the pacing and
    the closing gate. `answered=0` is the opening message, generated live rather
    than planned so that it carries the mood, which is drawn after the plan comes
    back. `spoken` defaults to nothing said yet, which is what the opening turn
    wants anyway.

    The profession is resolved here rather than passed in, so a round already
    under way keeps getting it on every turn without the caller threading it
    through `reply`. `settings.stage` is stored on the session, so a replay of an
    old round resolves the same one it ran with.
    """
    profession = resolve_profession(settings.stage)
    asked_for, fixed = split_checklist(plan)
    checklist_block = "\n".join(
        f"- [{c.id}] {c.criterion}: {c.description}" for c in asked_for
    )
    fixed_block = (
        f"\nNever asked for, rated from how they answer everything else — so it is never "
        f"an open gap and never worth a turn:\n- [{fixed.id}] {fixed.criterion}"
        if fixed
        else ""
    )
    questions_block = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(plan.questions))
    history_block = (
        f"\n\nHISTORY OF PREVIOUS PRACTICE ROUNDS (background only — use it to probe "
        f"previous weaknesses):\n{digest}\n"
        "Where you have a free choice — which thread to pull, which follow-up to ask — "
        "lean towards ground this history leaves unexplored. It is a lean, not a rule: if "
        "the candidate offers a story that appears above, take it and look for the parts "
        "of it nobody examined, rather than steering them off it.\n"
        "This history is private preparation from rounds the candidate ran with OTHER "
        "interviewers. In the fiction you have never met them before and nothing in it "
        "was ever said to them or by them in your presence. Never mention it,"
        " never refer to an earlier round, a "
        "previous score or feedback they were given, you are hearing "
        "everything for the first time."
        if digest
        else ""
    )
    brief_block = (
        f"\n\nWhat you know about the company and your team:\n{plan.company_brief}"
        if plan.company_brief
        else ""
    )
    # On the same switch as the planner's culture rules: a stage that was not
    # asked to write a culture question has none to ask, and telling it to
    # screen for the principles anyway only produces one improvised at the
    # candidate.
    culture_block = (
        "\n\nThe principles your company hires against (you screen for these as strictly "
        "as for skills — probe for lived examples of them, but never name them to the "
        "candidate or ask them to recite them):\n"
        + "\n".join(f"- {a}" for a in plan.culture_anchors)
        if plan.culture_anchors and profession["culture_probe"]
        else ""
    )
    role_block = (
        "\n\nYour own job in this process, here at this company:\n"
        f"{plan.interviewer_role}\n"
        "This governs the whole round, not the end of it. It decides what you open with, "
        "which questions you pick and how you phrase them: every one of them should sound "
        "like the person described above asking it, in a round that exists for what this "
        "round exists for. Anything your prepared questions have not established "
        "is yours to get, in plain words and early enough that a short call still reaches "
        "it — a straight question about the candidate's own situation hands over no "
        "scorecard, so do not dress one up as a behavioural probe. "
        if plan.interviewer_role
        else ""
    )
    # Absent from rounds stored before the plan carried it, and the prompt then
    # says nothing about the time rather than guessing at one.
    when_block = (
        f" You are sitting down to it on {plan.session_time}, so greet the candidate and"
        " speak about the day the way someone at that hour would."
        if plan.session_time
        else ""
    )
    budget = answer_budget(settings.duration_minutes)
    phase, may_close = interview_phase(spoken, answered, settings.duration_minutes)
    gone = max(1, round(spoken))
    left = max(0, settings.duration_minutes - gone)
    # The opening states what the first question must be true of and never gives
    # an example of one: a concrete exemplar inside the instruction that writes
    # this turn outranks anything stated further up, and every stage then says it.
    where_block = (
        f"""WHERE YOU ARE RIGHT NOW: the candidate has just joined and nothing has been said
yet. Open the interview: greet them the way you would today, say in one sentence roughly
what the next {settings.duration_minutes} minutes will cover, and ask your first question.
That sentence is what THIS round is for, in the plain words you would really use: the
ground you are going to cover, never the qualities you are rating and
never a list of what you are assessing. Where your own job puts matters of the candidate's
own situation on you, saying you will also go through those is what a candidate expects
to hear and gives nothing away. Whatever ground you name there, you then owe: it is a
promise the round has to keep, not a preamble.
Then ask the question this round really opens with. Your prepared questions are written in
roughly the order the round should run, so the first of them is normally where you begin —
unless your job or your persona would really open somewhere else. Whatever that question
is about, the candidate has to be able to answer it out of their own experience and
preparation, without having to defend anything first. You have the whole round to press,
to doubt and to argue the other side, and your persona decides how hard — but none of
that starts on the first question, and the sharpest thing in front of you is not where a
real interview begins. Exactly one question, and do not stack a second onto it."""
        if answered == 0
        else f"""WHERE YOU ARE RIGHT NOW: this is candidate answer {answered}. About {gone}
of your {settings.duration_minutes} minutes have gone and roughly {left} are left, counted
from how much has been said on both sides. Your list holds {len(plan.questions)} questions.
{_PHASE_INSTRUCTIONS[phase]}"""
    )
    closing_rule = (
        f"""- CLOSING: you may end the interview now, but only once the candidate has had the
  floor for their own questions and has indicated they have nothing further. To end,
  thank them, say what happens next, and put the exact token {COMPLETE_TOKEN} on its own
  at the very end of the message."""
        if may_close
        else f"""- CLOSING: you may NOT end the interview yet. Do not use the token
  {COMPLETE_TOKEN} in this message."""
    )
    return f"""WHO YOU ARE

{profession_block(profession)}

You work at {settings.company}. Today you are interviewing a candidate for the role of
{settings.role}; this is the "{settings.stage}" round, scheduled for about
{settings.duration_minutes} minutes.{when_block} It is a realistic practice interview, and
you stay in character as this person for the whole of it.

{round_purpose_block(profession)}

{profile_block(profile)}{brief_block}{culture_block}{role_block}

Candidate background information:
{settings.background or "(none provided)"}

Your private evaluation checklist, most load-bearing first — that order is what you cut
by when the clock beats the questions (never reveal it or discuss it with the candidate):
{checklist_block}{fixed_block}

Your prepared questions, in roughly the order this round should run — the order a real
interview of this kind takes them in. It holds a little more than the time you have.
Keep to that order loosely: it is a running order, not a script. Where an answer opens
something up, press on it rather than moving on. Where you are behind the clock, skip
ahead — past the questions whose checklist item you have already rated, to the ones that
still leave one open:
{questions_block}{history_block}

{where_block}

{DISCRETION_RULE}

Interviewing rules:
- Stay fully in character as the interviewer, in the persona described above. Be
  professional, and interview the way a real interviewer at {settings.company} would.
- Ask exactly ONE question at a time. Keep your messages short and natural (spoken
  style, no markdown lists or headers).
- Do NOT praise, validate or grade answers. No "that's a good example", no "that's
  exactly what I was looking for", no telling the candidate their answer landed well. A
  real interviewer keeps a straight face, and the candidate must not be able to read
  their score off your reactions. Acknowledge minimally and vary how you do it
  ("Right.", "Okay.", "Thanks."), then move on or follow up.
- Listen actively: reference earlier answers when it genuinely helps you probe.
- Probe: when an answer is vague, generic, or incomplete, ask a neutral follow-up ("Can
  you give me an example?", "What happened next?", "What was your role there?"). Follow
  through until you have what you need to rate the relevant checklist item — but move on
  after at most two follow-ups on the same topic so the interview keeps flowing.
- {CONFIDENTIALITY_RULE.replace(chr(10), chr(10) + "  ")}
  Never press for a name or an exact figure.
- Do NOT coach, give feedback, or evaluate during the interview. Deflect politely if the
  candidate asks how they are doing ("We'll get to feedback at the end.").
- {TRANSCRIPTION_CAVEAT.replace(chr(10), chr(10) + "  ")}
  In particular: never repeat a garbled name back to the candidate, never comment on a
  mispronunciation or a wrong-sounding name, and never build a question out of one. If a
  name or number is clearly a transcription error, silently use the correct one.
- If the candidate answers in a different language, continue in the language they use.
- WORK THROUGH THE LIST IN ORDER, and cut it to the time you have. You are told each turn
  how much of the {settings.duration_minutes} minutes has gone; it is counted from how
  much has actually been said, so a long answer costs the round what it would really cost
  and a short one leaves room for another question. At a normal pace that is about
  {budget} answers, but a candidate who talks at length will get fewer questions than one
  who does not — that is the slot working as it should, not something to correct for.
  Leaving questions unasked is the plan, not a failure. Before you speak, take stock
  silently: which checklist items are still unrated, what your own job still leaves
  unestablished, and how many answers are left. Then either follow up on what you have
  just heard, where the answer was thin or opened something worth pressing, or move on to
  the next question that still leaves a checklist item unrated — skipping the ones whose
  item you can already rate. Never re-ask ground the candidate has already covered, and
  when fewer answers remain than you have gaps, keep the questions that reach the criteria
  standing highest on your checklist and what only this conversation can settle.
- Track your checklist silently. Before you close, make sure the candidate has genuinely
  had the floor to ask you questions — invite them, answer briefly, and invite again.
{closing_rule}
- If the candidate clearly wants to stop, close politely and end with {COMPLETE_TOKEN}
  regardless of the phase."""


def turns_to_messages(transcript: list[Turn]) -> list[dict]:
    """Map the domain transcript to Anthropic messages.

    interviewer -> assistant, candidate -> user. The transcript starts with the
    interviewer's opening, so a leading synthetic user turn is prepended (the
    API requires the first message to be from the user).
    """
    messages = [
        {"role": "user", "content": "(The candidate has joined the interview.)"}
    ]
    for turn in transcript:
        role = "assistant" if turn.role == INTERVIEWER else "user"
        messages.append({"role": role, "content": turn.content})
    return messages


def transcription_hint(settings: InterviewSettings, transcript: list[Turn]) -> str:
    """Vocabulary prompt for the speech-to-text model.

    Priming the recogniser with the company name, the role and what the
    interviewer just said stops an unfamiliar company or product name coming
    back as a plausible-sounding near-miss.
    """
    last_interviewer = next(
        (t.content for t in reversed(transcript) if t.role == INTERVIEWER), ""
    )
    parts = [
        f"A job interview at {settings.company} for a {settings.role} position.",
        f"Spell names correctly, including: {settings.company}, {settings.role}.",
    ]
    if last_interviewer:
        parts.append(f"The interviewer just said: {last_interviewer[:600]}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# SECTION 2: EVALUATION
# ---------------------------------------------------------------------------

# Keeps one habit from being charged twice, to the delivery adjustment and to
# the criterion.
_DELIVERY_BOUNDARY = (
    f'\nThat holds for "[{STRUCTURE_CRITERION.id}]" too: how an answer is built is that\n'
    "criterion, the words it is padded with are this finding, and one habit is not\n"
    "charged to both."
)

# The second half of "what counts as a chance", dropped for sessions planned
# before the criterion existed.
_FIXED_CRITERION_RULE = f"""

The criterion about how the candidate answers is the other kind:
"[{STRUCTURE_CRITERION.id}]", under "ALWAYS RATED" above. No question raises it and none
could, so it is never "not_assessed" for want of one — the chance was every answer the
candidate gave.
Rate it "not_assessed" only where there is next to nothing to read — a round that ended
after a turn or two, or a transcript too garbled to follow. It rates the shape of an
answer and not its content: whether the point arrives before the background, whether the
answer holds together long enough to be followed to the end, whether it stops once the
question has been answered, and whether its size matches the question that was asked.
Rate the pattern across the transcript and never the worst single moment — one wandering
answer among six clear ones is a comment in "answer_review", not this rating. Length is
not disorder: a candidate asked to walk through a project gave the answer that was asked
for. In "evidence", name two places the pattern showed."""

# Key order is load-bearing: structured outputs are generated field by field, so
# the question-by-question pass, the read on the candidate's own questions and
# the checklist ratings all come before the score the prompt requires to be
# consistent with them. Same reason for "key_takeaway" coming last of the
# judgments — it picks the single most damaging of the findings above, so
# everything it picks from has to be written before it.
_EVALUATION_PROPERTIES = {
    "answer_review": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "addressed": {"type": "string", "enum": ["full", "partial", "avoided"]},
                "comment": {"type": "string"},
            },
            "required": ["question", "addressed", "comment"],
            "additionalProperties": False,
        },
    },
    "candidate_questions": {
        "type": "object",
        "properties": {
            "rating": {
                "type": "string",
                "enum": ["strong", "adequate", "weak", "not_assessed"],
            },
            "impression": {"type": "string"},
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "signal": {
                            "type": "string",
                            "enum": ["strong", "reasonable", "weak", "red_flag"],
                        },
                        "comment": {"type": "string"},
                    },
                    "required": ["question", "signal", "comment"],
                    "additionalProperties": False,
                },
            },
            "better_questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["rating", "impression", "questions", "better_questions"],
        "additionalProperties": False,
    },
    "results": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "criterion": {"type": "string"},
                "rating": {
                    "type": "string",
                    "enum": ["met", "partial", "not_met", "not_assessed"],
                },
                "evidence": {"type": "string"},
                "comment": {"type": "string"},
            },
            "required": ["id", "criterion", "rating", "evidence", "comment"],
            "additionalProperties": False,
        },
    },
    "score": {"type": "integer"},
    "strengths": {"type": "array", "items": {"type": "string"}},
    "improvements": {"type": "array", "items": {"type": "string"}},
    "key_takeaway": {
        "type": "object",
        "properties": {
            "point": {"type": "string"},
            "severity": {"type": "integer", "enum": [1, 2, 3]},
            "evidence": {"type": "string"},
            "verdict": {"type": "string"},
        },
        "required": ["point", "severity", "evidence", "verdict"],
        "additionalProperties": False,
    },
    "summary": {"type": "string"},
    "topics_covered": {"type": "array", "items": {"type": "string"}},
    "progress_notes": {"type": "string"},
}


def evaluation_schema(include_progress_notes: bool) -> dict:
    """Evaluation schema, without progress_notes when there is no history.

    Dropping the field is the only thing that works: told it may leave a
    required string empty, the model fills it anyway.
    """
    keys = [
        k
        for k in _EVALUATION_PROPERTIES
        if include_progress_notes or k != "progress_notes"
    ]
    return {
        "type": "object",
        "properties": {k: _EVALUATION_PROPERTIES[k] for k in keys},
        "required": keys,
        "additionalProperties": False,
    }


# Full schema, kept for callers that do not care about the history distinction.
EVALUATION_SCHEMA = evaluation_schema(True)


def transcript_block(transcript: list[Turn]) -> str:
    """Transcript for the evaluator, with voice answers flagged as transcribed."""
    lines = []
    for turn in transcript:
        if turn.role == INTERVIEWER:
            lines.append(f"INTERVIEWER: {turn.content}")
        else:
            tag = " (spoken, auto-transcribed)" if turn.modality == VOICE else ""
            lines.append(f"CANDIDATE{tag}: {turn.content}")
    return "\n".join(lines)


def get_evaluation_prompt(
    settings: InterviewSettings,
    plan: InterviewPlan,
    transcript: list[Turn],
    previous_evaluations: list[Evaluation],
) -> str:
    """User prompt for the final evaluation.

    Expects JSON: {answer_review: [{question, addressed, comment}],
    candidate_questions: {rating, impression, questions: [{question, signal,
    comment}], better_questions: [str]}, results: [{id, criterion, rating,
    evidence, comment}], score: int 0-100, strengths: [str], improvements:
    [str], key_takeaway: {point, severity 1-3, evidence, verdict}, summary: str,
    topics_covered: [str]} plus progress_notes when there is history
    (evaluation_schema).
    """
    asked_for, fixed = split_checklist(plan)
    checklist_block = "\n".join(
        f"- [{c.id}] {c.criterion}: {c.description}" for c in asked_for
    )
    fixed_block = (
        "\n\nALWAYS RATED — no question raises this one, so every answer the candidate "
        f"gave was the chance to show it:\n"
        f"- [{fixed.id}] {fixed.criterion}: {fixed.description}"
        if fixed
        else ""
    )
    fixed_rule = _FIXED_CRITERION_RULE if fixed else ""
    delivery_boundary = _DELIVERY_BOUNDARY if fixed else ""
    culture_block = (
        "\n\nTHE PRINCIPLES THIS COMPANY HIRES AGAINST (rate lived evidence of these, not "
        "whether the candidate can name them — the interviewer was told never to name "
        "them either):\n" + "\n".join(f"- {a}" for a in plan.culture_anchors)
        if plan.culture_anchors
        else ""
    )
    previous_block = ""
    progress_instruction = ""
    if previous_evaluations:
        prev_lines = []
        for i, ev in enumerate(previous_evaluations, 1):
            line = f"Round {i}: score {ev.score}/100; improvements given: " + "; ".join(
                ev.improvements
            )
            if ev.key_takeaway:
                line += (
                    f"; the one thing they were told to fix: {ev.key_takeaway.point} "
                    f"(severity {ev.key_takeaway.severity}/3)"
                )
            prev_lines.append(line)
        previous_block = (
            "\n\nFEEDBACK FROM PREVIOUS PRACTICE ROUNDS (assess progress against these "
            "in progress_notes):\n" + "\n".join(prev_lines)
        )
        progress_instruction = (
            '\n- "progress_notes": a short paragraph on progress against the previous '
            "rounds' feedback, in the same voice as the summary. Name what has visibly "
            "improved and what has not."
        )

    return f"""You are an experienced interviewer at {settings.company} who has just finished a
"{settings.stage}" interview with a candidate for the role of {settings.role}.
Fill out your evaluation honestly and rigorously, the way a real hiring committee
write-up would read. Be candid: a mediocre performance gets a mediocre score.

Candidate background as it was given to the interviewer:
{settings.background or "(none provided)"}

EVALUATION CHECKLIST — rated against the questions that raised them:
{checklist_block}{fixed_block}{culture_block}

FULL INTERVIEW TRANSCRIPT:
{transcript_block(transcript)}{previous_block}

HOW TO READ THE TRANSCRIPT

{TRANSCRIPTION_CAVEAT}
Never list a transcription artifact as something to improve. If a
passage is genuinely unintelligible, ignore it rather than guessing at a weakness.

What a recogniser does not invent is how the candidate speaks. Hedges ("basically",
"sort of", "kind of", "I think", "a little bit"), sentences that start before they know
where they end, and openings that stall before the answer begins are the candidate's own
and are in the transcript because they were said. They are worth naming only as a
pattern, never as one instance and never as a tally, and only where the pattern changes
the impression — hedging that softens a claim the candidate is entitled to make firmly,
or filler dense enough to read as uncertainty. Quote two places it showed. This is a
delivery finding: it belongs to the delivery adjustment on the score, and it never turns
a met criterion into a partial one.{delivery_boundary}

RATE WHAT THE CANDIDATE HAD THE CHANCE TO SHOW

Rate every criterion against the chance this interview gave the candidate to show it:
where there was one, rate what they did with it; where there was none, "not_assessed",
and it is not an improvement either. What counts as a chance depends on the criterion.

Everything the checklist names about what the candidate has done, knows or decided is
rated against a question that raised it. Before you criticise anything they did not say,
check the transcript for whether the interviewer actually created the opening:
- If the interviewer never asked about a topic, rate it "not_assessed" and do not list it
  as an improvement.
- That holds even when the transcript happens to contain material bearing on it. A
  candidate answering one question may touch a criterion nobody put to them, and it is
  tempting to rate what is there — but a criterion the interviewer never raised in any
  question is "not_assessed" however much incidental evidence turned up, and what they
  said in passing is not a weak version of the answer they were never asked for. Above
  all, never make the incidental arrival itself the finding: that it came out defending
  something else, or late, or only under challenge, is a fact about the interviewer's
  running order, and holding it against the candidate is precisely the substitution this
  section exists to stop.
- If the interviewer cut a topic short, changed subject, or closed the interview, the
  candidate is not responsible for what went uncovered.
- Never criticise the candidate for the size or shape of a phase the interviewer
  controlled. If the interviewer allowed one closing question and then ended the
  interview, "only asked one question" is not a valid criticism; judge the quality of the
  question they did ask.

Four openings are a chance without being questions, and they are the whole of the list.
They are what the candidate was offered and did not take — the most expensive moments in
a real interview, and nobody asks a candidate to take them:

- Something the interviewer stated rather than asked. A salary band, a level, a
  standard the company hires against, a fact volunteered about the team: a candidate
  who lets a number go by without a word has made a choice, and it is one an
  interviewer notices. Judge what they did with it — including saying nothing, and
  including agreeing to it instantly.
- Something the interviewer said that plainly invited a follow-up and got none.
- The floor, once it is open. If the interviewer opened it and the candidate closed it
  themselves — "I think we're over time", "I don't think I have any questions" — that
  is the candidate's decision and may be named. If the interviewer closed it, it is not.
- A claim only the candidate could have made. If the interviewer asked something the
  candidate's own background answers unusually well and the answer stayed generic, that
  is theirs, not a missing question.

This is a narrow list, not a licence: it covers openings visible in the transcript that
the candidate could have taken in the turn they already had. Everything else the
interviewer never raised is still not_assessed, so do not read the incidental-evidence
rule as narrowing these four, or them as reopening it.{fixed_rule}

Improvements must be things the candidate could actually have done differently in this
conversation.

WHAT THE CANDIDATE NEED NOT DISCLOSE

{CONFIDENTIALITY_RULE}
Rate the specificity they could legitimately give: the counterparty's role and what they
wanted, the mechanism, the relative magnitude, and whether they can defend how the number
was arrived at.
This protects the candidate's employers and nobody else. Facts about the candidate's own
situation — where they can work, when they could start, what they expect to be paid, what
level they are asking for, what else they are doing — are theirs to state, and vagueness
about those is a finding like any other, not a confidence they were entitled to keep.

JUDGE ANSWERS AGAINST THE QUESTIONS THEY ANSWER

A checklist rating is not a hunt for the best thing the candidate said anywhere in the
transcript. Every answer is a response to a specific question, and answering the question
actually asked is itself part of what is being assessed. For each exchange:

- Did the answer address what was asked, or an adjacent question the candidate would
  rather answer? Sliding into a rehearsed story is a real weakness even when the story is
  a good one.
- If the question had several parts, were all of them answered? A half-answered question
  is partial no matter how strong the half that was answered.
- Did the answer respect the constraints in the question — the time frame ("in the last
  two years"), the scope ("a project you led yourself", "at your current employer"), the
  format ("briefly", "the two-minute version")? Quietly substituting an example that does
  not meet the constraint is a dodge, even an unintentional one.
- Did it take probing to get there? Something the candidate volunteered counts for more
  than the same content extracted after two follow-ups. Where a rating rests on material
  that only surfaced under pressure, say so.
- When the question was why this company, or what they already know about it, ask whether
  what came back could have been said about any employer in the same market. Naming
  something specific — a product, a published result, a strategy, a way of working — is
  where the whole value of that answer sits; warmth about the mission and interest in the
  field is what every candidate says. Where it stayed generic, say so and quote it. The
  candidate is never marked down for not knowing a fact nobody asked for, but this
  question asks for it.
- In "evidence", make clear which question the material answers, so every rating is
  traceable to an exchange rather than to the transcript at large.

The fairness rule above still governs: an answer counts as avoided only when the
candidate had the room to answer and did not. If the interviewer interrupted, moved on or
ran out of time, that is not the candidate's failure. And never call an answer
unresponsive because transcription noise garbled it.

JUDGE THE QUESTIONS THE CANDIDATE ASKED

What a candidate asks when they get the floor is evidence, not politeness. Assess the
content of each question, never the number of them:

- Is it aimed at the person in front of them? Compensation, process and logistics belong
  to HR; team, scope, architecture and roadmap belong to the hiring manager. A sensible
  question put to the wrong interviewer still reads as poor calibration for a
  "{settings.stage}" conversation.
- Does it show they looked into {settings.company}, or could the job ad and the careers
  page have answered it? A question that names something specific about the company or
  the {settings.role} role counts for far more than one that would fit any employer.
- What does it optimise for? Questions about the work — ownership, what success looks
  like in the first months, how decisions get made, what the team is struggling with —
  read as someone thinking about doing the job. Questions about the package — holiday,
  remote days, when they will hear back — are legitimate, but when they are all the
  candidate asked, that itself is the signal.
- Is it a disguised objection or a self-inflicted red flag ("how closely is performance
  monitored?", "how much overtime is expected?")? Say so plainly.
- Did they listen to the answer and follow up on it, or read the next item off a prepared
  list? A real follow-up is one of the strongest signals available here.

Then say what a hiring interviewer would conclude about the person who asked these
questions. Be specific about the inference — "asking only about home-office days reads as
someone weighing the commute rather than the work" — not a restatement of the questions.

The fairness rule above governs here too. If the interviewer never opened the floor, or
closed the interview straight after one question, rate this "not_assessed" and do not list
it as an improvement — the candidate cannot be marked down for a phase the interviewer
controlled. Judge only the questions they actually got to ask.
Where the floor was open and the candidate closed it themselves — waving the rest of the
time away, saying they had nothing further while there was clearly time left — that is
their decision, not the interviewer's, and it belongs in "impression": what an interviewer
concludes from someone who stopped asking. That remains a judgement about the choice, not
a quota: three good questions are three good questions.

THE ONE THING TO TAKE AWAY

The improvements are a list; this is the one line the candidate should carry into the
real interview. Name the single thing an experienced interviewer would give in the
debrief room as the reason not to hire them. One thing, not a ranked summary of several.

- Take it from the transcript, not from the findings you have just written. Go back to
  the turns it rests on and read each one to its end before you name it: the sentence you
  wrote about a moment is always shorter than the moment, and a point built out of your
  own summaries inherits whatever the summarising dropped — most often the half of a turn
  that cut against it. It will usually name the same thing as the sharpest of the
  improvements, because both are true of the same interview; that is not a reason to
  sharpen one into the other.
- It can come from anywhere: the content of the answers, a failure to answer what was
  actually asked, or the impression the person leaves — vagueness, passivity,
  defensiveness, credit taken for a team's work, a story told from the passenger seat.
  Pick whatever would really be said out loud, not the most polite candidate.
- It need not come from one answer, and the strongest finding usually does not. A habit
  that showed three times is worth more than a single weak moment; then name the pattern
  and point at two places it showed.
- Say it as an instruction they can act on next time — "Be specific: name the number you
  moved", "Lead with the headline, then the detail", "Show agency: say what you decided,
  not what happened to you". An instruction, not a diagnosis and not a compliment
  sandwich.
- Repeat it when it is still the answer. If a previous round already told them this and
  it is still the most damaging thing, say it again and say in the verdict that it
  survived the feedback — that is the finding, not a reason to look for something new.
- There is always one. In a genuinely strong round it is a severity 1 refinement of
  something that already worked; do not invent a flaw to fill the field, and do not leave
  it empty because nothing was terrible.
- It must survive the fairness rule like everything else: something the candidate did
  with an opening they were given, never a topic the interviewer never raised, never an
  artifact of transcription.
- Severity rates what an interviewer would do about it, not how hard it is to fix, and it
  changes nothing about "score" — the number is set by the checklist, this is what the
  candidate remembers.

ONE VOICE: WRITE IT TO THE CANDIDATE

The candidate reads this evaluation, and every string in it is spoken to them as "you" —
the checklist comments and the evidence under them, the answer-by-answer comments, the
strengths, the improvements, the takeaway and the summary alike. Never "he", never "she",
never "they", never "the candidate".
The interviewer stays in the third person throughout — "the interviewer never put this
question", "the interviewer closed the floor" — so the two people in the transcript stay
apart on the page.

Four things keep the wording they already have:
- Words that were actually said. A quote in "evidence" keeps the candidate's own words ("I
  built it end to end"); it is the sentence around the quote that says "you".
- "criterion" is copied from the checklist above exactly as written there, so that ratings
  stay comparable between rounds.
- The "question" fields in "answer_review" and "candidate_questions" paraphrase a question
  as it was put, in the wording it was put in.
- "topics_covered" is a list of plain noun phrases with no pronoun in them at all — it is
  the one field the candidate never reads.

Produce:
- "answer_review": one entry per substantive question the interviewer asked — skip
  greetings, pure clarifications and the interviewer's answers to the candidate's own
  questions. "question" is a short paraphrase of what was asked, "addressed" is
  full / partial / avoided, and "comment" is one sentence on what was or was not
  answered, naming the unanswered part explicitly when there is one. The candidate's own
  questions are judged separately, in "candidate_questions".
- "candidate_questions": the read on the questions the candidate asked. "questions" has
  one entry per question they asked the interviewer — "question" is a short paraphrase,
  "signal" is strong / reasonable / weak / red_flag, "comment" is one sentence on what it
  tells you about them. "rating" is the overall read: strong / adequate / weak, or
  not_assessed when they never got the floor. "impression" is a short paragraph naming
  what an interviewer would conclude about them from these questions.
  "better_questions" is 2-3 questions they could have asked instead,
  concrete to {settings.company}, the {settings.role} role and a "{settings.stage}"
  conversation — no generic filler. Leave "questions" and "better_questions" empty and
  "impression" a single sentence when the rating is not_assessed.
- "results": one entry per checklist item, both lists above (use the same ids). "rating"
  is one of met / partial / not_met / not_assessed. "evidence" quotes or closely
  paraphrases what the candidate actually said; "comment" explains the rating and what a
  stronger answer would have included. Use "not_assessed" by the chance rule above.
- "score": overall 0-100. It must be consistent with the ratings you just gave: start
  from the share of checklist items rated met (1 point), partial (0.5) and not_met (0),
  ignoring not_assessed items, as a percentage — then adjust by at most 10 points for
  delivery, communication, responsiveness to the questions as recorded in answer_review,
  and the quality of the candidate's own questions as recorded in candidate_questions.
  Bands for reference: 90+ outstanding, 75-89 strong hire signal, 60-74
  mixed, 40-59 weak, below 40 clear no-hire for this round.
- "strengths": 2-4 specific things the candidate did well.
- "improvements": 2-3 specific, actionable points to work on before the real interview.
  If any question was only partly answered or dodged, one of these must say so and name
  the question — that is among the most useful things the candidate can learn here.
- "key_takeaway": the one thing above, and it is always filled in. "point" is the
  instruction itself, under ten words, imperative — this is the line they will remember,
  so make it quotable rather than complete. "severity" is 1 (a light concern), 2
  (something the interviewer would raise unprompted in the debrief)
  or 3 (a red flag that on its own sinks the candidacy). "evidence" is what makes it
  recognisable: the moment or the two or three moments it showed, quoting or closely
  paraphrasing the candidate, so they can hear themselves doing it. "verdict" is one or
  two sentences carrying what would be said about them in the debrief room, but said to
  them: what this costs them, and whether it is the same thing a previous round already
  told them. The whole takeaway is one person speaking to one person.
- "summary": a short paragraph, candid and constructive.
- "topics_covered": the concrete projects, stories and situations the candidate used in
  their answers, one short phrase each (e.g. "the classification service at the previous
  employer", "the deployment bottleneck"). This is used to demand fresh material in the
  next practice round, so name the story, not the competency.{progress_instruction}

Write every string as plain prose, and every one of them to the candidate. Do not wrap any
field value in quotation marks.
"""
