"""Everything written before the interview starts: the plan, and the research it reads.

`get_plan_prompt` designs the round — the interviewer's own job at this company,
the scorecard, the questions in the order the round should run them, and the mood
pool this round's mood is drawn from. `get_company_research_prompt` writes the
files under `data/context/<company>/` that the planner later reads back out.
Running the round the plan describes, and rating it afterwards, is
`prompts_interview`; the round profile and the rules both halves obey are
`prompts_common`.

Each structured prompt's docstring documents the exact JSON contract expected
back; the matching schema constant is enforced server-side via structured
outputs, so no client-side JSON repair is needed.
"""

import random
from datetime import datetime

from src.core.prompts_common import (
    CONFIDENTIALITY_RULE,
    DISCRETION_RULE,
    STRUCTURE_CRITERION,
    answer_budget,
    curveball_domains,
    dedupe,
    profession_block,
    profile_block,
    resolve_profession,
    round_purpose_block,
)
from src.models.interview import InterviewSettings

# ---------------------------------------------------------------------------
# SECTION 1: INTERVIEW PLANNING
# ---------------------------------------------------------------------------

_PLAN_PROPERTIES = {
    "company_brief": {"type": "string"},
    "culture_anchors": {"type": "array", "items": {"type": "string"}},
    "interviewer_role": {"type": "string"},
    "mood_adjectives": {"type": "array", "items": {"type": "string"}},
    "checklist": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "criterion": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["id", "criterion", "description"],
            "additionalProperties": False,
        },
    },
    "curveball": {"type": "string"},
    "questions": {"type": "array", "items": {"type": "string"}},
}


def plan_schema(include_curveball: bool) -> dict:
    """Plan schema, minus the pieces this stage has no use for.

    Dropping the field is the only thing that works, exactly as with
    `progress_notes` in `prompts_interview.evaluation_schema`: told it may leave
    a required string empty, the model writes a scenario anyway, and the planner
    is then invited to spend a question on it. The matching bullet leaves the
    prompt with it (see `get_plan_prompt`).

    `required` names every property present, which is why a flag cannot simply
    drop the field from `required`: `additionalProperties` is False, so a
    property the schema declares is a property the model must return.
    """
    dropped = set()
    if not include_curveball:
        dropped.add("curveball")
    keys = [k for k in _PLAN_PROPERTIES if k not in dropped]
    return {
        "type": "object",
        "properties": {k: _PLAN_PROPERTIES[k] for k in keys},
        "required": keys,
        "additionalProperties": False,
    }


PLAN_SCHEMA = plan_schema(True)

_MAX_PAST_QUESTIONS = 24
_MAX_PAST_TOPICS = 20


def history_digest(previous: list) -> str:
    """Summarize previous sessions for the same settings into prompt context.

    `previous` is a list of InterviewSession (oldest first). Returns "" when
    there is no history. Deliberately carries *topics and stories*, not just
    question wording: rephrasing a question does not stop the candidate
    re-telling the same project for the fifth time.

    Purely descriptive — it states what happened in earlier rounds and nothing
    else. What to DO with it belongs to whichever prompt embeds it, because the
    two consumers need opposite things: the planner may act on the history
    openly, while the live interviewer must never let on that it exists. An
    instruction phrased in here reaches both, and the interviewer then says it
    out loud to a candidate it has, in the fiction, never met.
    """
    if not previous:
        return ""
    asked: list[str] = []
    topics: list[str] = []
    scores: list[str] = []
    for session in previous:
        asked.extend(session.plan.questions)
        if session.evaluation:
            topics.extend(session.evaluation.topics_covered)
            scores.append(
                f"round on {session.created_at[:10]}: {session.evaluation.score}/100"
            )

    latest = previous[-1].evaluation
    lines = [
        f"The candidate has practiced this interview {len(previous)} time(s) before."
    ]
    if scores:
        lines.append("Previous scores: " + "; ".join(scores))
    if asked:
        lines.append("Questions asked in previous rounds:")
        lines.extend(f"- {q}" for q in dedupe(asked, _MAX_PAST_QUESTIONS))
    if topics:
        lines.append("Stories and situations the candidate covered in previous rounds:")
        lines.extend(f"- {t}" for t in dedupe(topics, _MAX_PAST_TOPICS))
    if latest:
        weak = [
            f"{r.criterion} (rated {r.rating})"
            for r in latest.results
            if r.rating in ("partial", "not_met")
        ]
        if weak:
            lines.append("Weakest areas in the most recent round:")
            lines.extend(f"- {w}" for w in weak)
        if latest.improvements:
            lines.append("Improvement points from the most recent round's feedback:")
            lines.extend(f"- {i}" for i in latest.improvements)
    return "\n".join(lines)


def core_checklist_block(previous: list) -> str:
    """The previous round's checklist, so the rubric stays comparable across rounds.

    Minus the fixed criterion: shown it, the planner writes its own version of it
    and the round is scored on the same thing twice.
    """
    if not previous:
        return ""
    carried = [c for c in previous[-1].plan.checklist if c.id != STRUCTURE_CRITERION.id]
    if not carried:
        return ""
    items = "\n".join(f"- [{c.id}] {c.criterion}" for c in carried)
    return (
        "\n\nCHECKLIST USED IN THE PREVIOUS ROUND:\n"
        f"{items}\n"
        "Keep the 3-4 most central of these as-is, with the SAME ids and criteria wording, "
        "so scores stay comparable between rounds. Replace the remainder with different "
        "criteria that suit this round's persona and scenario."
    )


def _plan_sizes(duration_minutes: int) -> tuple[int, int, int, int]:
    """Checklist size for the slot, and the running order that fills it.

    The checklist scales with the duration: roughly one criterion per 4-6
    minutes is what a round has time to actually assess, and it is the rubric
    the score is anchored to.

    The questions are a running order for this slot, sized by the same
    `answer_budget` the live round is paced by — about three minutes each — plus
    a couple the interviewer can drop when an answer is worth pressing on. The
    floor keeps the pool at least as long as the checklist: a criterion no
    question can reach is "not_assessed" by construction, whatever the candidate
    says.
    """
    items_lo = max(3, min(7, duration_minutes // 6))
    items_hi = max(4, min(8, duration_minutes // 4))
    questions_lo = max(items_hi, answer_budget(duration_minutes))
    questions_hi = questions_lo + 2
    return items_lo, items_hi, questions_lo, questions_hi


def get_plan_prompt(
    settings: InterviewSettings,
    digest: str,
    profile: dict,
    core_block: str = "",
    used_scenarios: list[str] | None = None,
    now: datetime | None = None,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    """User prompt for interview-plan generation, and the hour it was written for.

    Expects JSON: {company_brief: str, culture_anchors: [str],
    interviewer_role: str, mood_adjectives: [str],
    checklist: [{id, criterion, description}], curveball: str,
    questions: [str]} — `plan_schema`, whose `curveball` is present only for the
    stages that want one. This prompt and that schema are gated by the same
    `resolve_profession` lookup, so a caller building the schema must resolve it
    from the same `settings.stage`.

    `now` is the wall-clock time of this practice session and is what the mood
    pool is written from. `rng` permutes the curveball domains (see
    `curveball_domains`). Both are injectable for tests.

    The time is returned as it was rendered, because the live interviewer has to
    greet and pace the round by the same clock the moods were reasoned from.
    """
    now = now or datetime.now()
    when = f'{now.strftime("%A")} at {now.strftime("%H:%M")}'
    profession = resolve_profession(settings.stage)
    wants_curveball = profession["curveball"]
    wants_culture = profession["culture_probe"]
    history_block = (
        f"\n\nHISTORY OF PREVIOUS PRACTICE ROUNDS:\n{digest}\n"
        "The exploratory questions must be new: do not reuse the ones listed above or "
        "merely rephrase them, reach the same competencies from a genuinely different "
        "angle, and do not send the candidate back to a story they have already told. "
        "The standing questions are the exception. Where a question exists to establish "
        'something "interviewer_role" below puts on this interviewer, a real interviewer '
        "asks it of everyone, every time, in much the same words — repeating one of those "
        "is correct and costs the round nothing. Freshness is a rule about the questions "
        "whose whole value is being unexpected. Build this round so the weakest areas "
        "above get probed hard and the candidate has a real chance to show improvement "
        "on them."
        if digest
        else ""
    )
    scenario_block = (
        "\n\nSITUATIONAL SCENARIOS USED IN RECENT ROUNDS.\nThe one you write below must be"
        " clearly different — a different kind of situation, not a\nreworded version:\n"
        + "\n".join(f"- {s}" for s in used_scenarios)
        if used_scenarios and wants_curveball
        else ""
    )
    context_block = (
        "\n\nCOMPANY RESEARCH NOTES (see the warning below):\n"
        + settings.company_context
        if settings.company_context
        else ""
    )
    items_lo, items_hi, questions_lo, questions_hi = _plan_sizes(
        settings.duration_minutes
    )
    curveball_bullet = (
        f"""6. "curveball" — one situational scenario for this round: a concrete, realistic
   dilemma somebody in THIS role at THIS company could actually land in, pitched at the
   level this stage hires for. One sentence, written as a situation the candidate is put
   into, never as the quality you want to measure. It does not have to be technical — take
   it from whatever this job is actually made of, which may be
   {curveball_domains(rng)}.
   It must be something a standard behavioural question would not reach, and it must not
   repeat any scenario listed under the recent rounds above.
"""
        if wants_curveball
        else ""
    )
    questions_number = 7 if wants_curveball else 6
    curveball_question_rule = (
        """
   - The scenario you wrote in "curveball" may be one of these questions, phrased in this
     company's own terms, placed at the point in the round where it belongs."""
        if wants_curveball
        else ""
    )
    culture_anchor_use = (
        " Screening against them is\n   part of this round's job, so the checklist and the"
        " questions below both have to."
        if wants_culture
        else " Screening against them belongs to\n   the rounds this company runs for it,"
        " which is not this one — collect them for the\n   brief, and do not build this"
        " round's checklist or questions around them."
    )
    culture_criterion_rule = (
        """ When
   culture_anchors is non-empty, at least one criterion must assess lived fit against
   those principles, named in the company's own language — this is a criterion, not a
   question, so it may be explicit here."""
        if wants_culture
        else ""
    )
    culture_question_rule = (
        """
   - When culture_anchors is non-empty at least one question must probe a lived example
     against one of those principles — without naming the principle or the framework.
     Ask for the situation the principle would show up in and let the candidate reveal
     whether they work that way."""
        if wants_culture
        else ""
    )
    optional_question_rules = curveball_question_rule + culture_question_rule
    prompt = f"""WHO THE INTERVIEWER IS

{profession_block(profession)}

Everything below is preparation for that person. The checklist, the questions and what
they volunteer are all theirs to use, so each one has to be something that professional
would really ask, and really know.

Design a realistic practice job interview.

Company: {settings.company}
Role: {settings.role}
Interview stage: {settings.stage}
Scheduled duration: about {settings.duration_minutes} minutes
When it is taking place: {when}

{round_purpose_block(profession)}

{profile_block(profile)}

Background information provided by the candidate (CV notes, the job ad, personal focus
areas):
{settings.background or "(none provided)"}{context_block}{history_block}{core_block}

IMPORTANT — the company research notes above are the candidate's OWN preparation
material. They are a record of what the candidate has read and the answers they intend
to give. Use them only for realistic factual colour about the company, the team and its
products. Do NOT build questions designed to let the candidate recite them, and do NOT
treat them as the correct answers to this interview.
The plain question about what the candidate knows or thinks of {settings.company} is not
one of those, and is not a leak: it supplies nothing, it is asked in real interviews of
this kind constantly, and whether anything specific comes back is precisely the
measurement. What is forbidden is feeding them the material — naming a product, a
principle or a result from the notes inside the question and asking what they make of it.

{DISCRETION_RULE}

WHAT THE CANDIDATE NEED NOT DISCLOSE

{CONFIDENTIALITY_RULE}
No criterion and no question needs a leak to satisfy it. This protects the candidate's
employers, not the candidate: facts about their own situation is not a breach of it.{scenario_block}

Produce:
1. "company_brief" — 8-12 short lines of company and team facts a real interviewer at
   {settings.company} would carry in their head: what the business does and how it makes
   money, what this team owns, the metrics they are judged on, the products or systems
   involved, and how they work. Facts only. Exclude everything that is candidate-side
   preparation (talking points, questions-to-ask lists, role-fit self-assessment,
   relocation notes). This brief — not the raw notes — is what the live interviewer will
   see, so it must stand on its own.
2. "culture_anchors" — if the material names the company's own values framework (an
   "X Formula", leadership principles, operating principles, ways of working, a culture
   memo), list its actual named principles in the company's own words, one per entry.
   Return an empty list only when there genuinely is no such framework. Companies that
   publish one screen against it as strictly as against skills.{culture_anchor_use}
3. "interviewer_role" — the professional described at the top of this prompt, as they
   actually exist at {settings.company}.
   "WHO THE INTERVIEWER IS" is the profession as it is practised everywhere; this is that
   same person in THIS company, for THIS role — named, concrete, and specific enough that
   the checklist and the questions can be built from it. Give them the title this company
   would really give them, never just "the interviewer", and write 4-6 short lines in
   their voice: the job they are doing in this process.
4. "mood_adjectives" — exactly 10 single-word adjectives for
   how THIS interviewer feels at this exact moment. Reason from the time given above and
   what that hour plausibly means for their day — first thing on a Monday, the last slot
   before the weekend, straight out of a long meeting, an evening call that has already
   overrun — and from the persona described above: these must be moods that person could
   credibly be in. Requirements:
   - No two may mean the same thing. Ten shades of "engaged" is a failed answer; the ten
     should read as ten genuinely different days.
   - They must not all be positive. Real interviewers turn up tired, distracted, impatient,
     sceptical or preoccupied, and an interview with one of those is still a professional
     interview. Several of the ten must be moods of that kind.
   - Single words, no explanations.
5. "checklist" — {items_lo} to {items_hi} evaluation criteria a real interviewer at this
   stage at this company would use as a scorecard for this role. Give each a short
   stable id (e.g. "motivation", "system_design"), a concise criterion, and a one-sentence
   description of what a strong answer demonstrates. Assess what THIS round is for, as
   described above.
   The scorecard answers to both halves of the job you described in "interviewer_role":
   the qualities this specific stage judges, and — how the
   candidate handles them. {culture_criterion_rule}
   ORDER THEM BY IMPORTANCE, the most load-bearing criterion for this round first. That
   order does two jobs downstream: it decides which criteria the questions below have to
   reach twice, and it is what the interviewer drops questions by when the clock runs
   short. Judge it against the job you wrote in "interviewer_role" — highest is what that
   job cannot close the call without a read on.
   One criterion is not yours to write. Every round is also scored on
   "[{STRUCTURE_CRITERION.id}] {STRUCTURE_CRITERION.criterion}", which is appended to your
   checklist automatically, so write no criterion of your own for how clearly, how
   concisely or how coherently the candidate answers — yours are {items_lo} to {items_hi}
   on top of it, on what this round is actually about. It needs no question either: it is
   rated from how the candidate answers the questions you do write.
{curveball_bullet}{questions_number}. "questions" — {questions_lo} to {questions_hi}
   questions: the running order for this round. That is about one question per three
   minutes of the {settings.duration_minutes}-minute slot, plus a couple spare — the
   interviewer keeps to your order loosely, presses where an answer opens something up,
   and drops the spares when that costs it time. Cover the ground from different angles,
   and tailor the questions to the company, role, and the candidate's background.
   ORDER THEM AS THE CONVERSATION WOULD RUN — the order a real interview of this kind
   takes them in, not a ranking. What opens the round, what its middle is made of, and what
   belongs near the end. Put the standing questions this interviewer's job requires where
   that profession would really put them: a recruiter settles the practical matters early,
   a hiring manager comes to them at the close. Open on ground the candidate can answer out
   of their own experience, and leave the sharpest question for a point where the
   conversation has warmed up enough to carry it.
   COVER THE CHECKLIST, in the order of importance you just gave it. Every criterion you
   wrote needs at least one question that plainly gives the candidate the chance to show
   it, and the criteria standing highest need a second route to them from a different
   angle — a candidate with no material for the first still gets a fair run at what this
   round most needs to know. A criterion no question reaches cannot be rated at all,
   however well the candidate does elsewhere.
   Requirements:
   - Every matter "interviewer_role" says this interviewer cannot close the call without
     having established needs a question here that plainly establishes it. Ask those in
     ordinary words, one short sentence each: a straight question about the candidate's
     own situation hands over no scorecard, so do not dress it up as a behavioural probe,
     and do not leave it to be improvised on the day.
   - Each entry must be ONE question. Avoid using multi-part questions or bundling things
    with "and" — follow-ups happen live in the interview.{optional_question_rules}
   - No question may restate the criterion it tests. Follow the discretion rule above.
   - Write them the way this round's persona would ask them.

Before you finish, reread the questions as the candidate would. If they could reconstruct
the checklist from them, or if any of them hands over the shape of the answer you want,
rewrite it. The plain questions this job requires are the exception: they are meant to
look exactly like what they are.

Write every string as plain prose. Do not wrap any field value in quotation marks.
"""
    return prompt, when


# ---------------------------------------------------------------------------
# SECTION 2: COMPANY RESEARCH
# ---------------------------------------------------------------------------
#
# This produces the files the *planner* later reads back out of
# data/context/<company>/ — it does not touch a live interview.
#
# Nothing here looks at the candidate: no CV, no fit score, no ranking of roles
# against a profile. A role or focus in the brief narrows *what is researched*
# and never turns the report into an assessment of the reader. `get_plan_prompt`
# already has to warn the planner that the context files are the candidate's own
# preparation material and not the answer key, and every candidate-side sentence
# in a research file is a sentence that warning has to survive.

COMPANY_RESEARCH_BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "role": {"type": "string"},
        "focus": {"type": "string"},
    },
    "required": ["company", "role", "focus"],
    "additionalProperties": False,
}


def get_research_brief_prompt(brief: str) -> str:
    """User prompt that splits a free-text research request into its parts.

    Expects JSON: {company: str, role: str, focus: str}
    (COMPANY_RESEARCH_BRIEF_SCHEMA).

    The UI asks for one text box, but the report has to be filed under a company
    folder, so the company name has to come back out of the prose. `role` and
    `focus` are "" when the request does not mention them.
    """
    return f"""Read this research request and pull out what it is asking for.

REQUEST:
{brief}

Produce:
1. "company" — the name of the company to research, as the company writes it
   itself (correct the capitalisation and spelling of an obvious misspelling, and
   strip legal suffixes like GmbH, N.V., Inc. unless the company is normally
   referred to with them). If the request names no company at all, return an
   empty string rather than guessing.
2. "role" — the job title the research should be angled at, or "" if none is
   named. Take the title only; leave requirements and seniority prose in "focus".
3. "focus" — everything else the request asks for, in one or two sentences:
   topics to dig into, a team or product to concentrate on, a location, a link to
   read, questions to answer. Include a pasted job advert as a short summary of
   what it reveals about the team and the work, not verbatim. "" if the request
   is nothing but a company name.

Write every string as plain prose. Do not wrap any field value in quotation marks.
"""


COMPANY_RESEARCH_SYSTEM = """\
You are a research analyst preparing a briefing on a company for someone who is about to
interview there. You have web search and web fetch; use them properly rather than writing
from memory.

- Go to primary sources first: the company's own site, careers page, engineering blog,
  press releases, financial reporting, published papers and open-source repositories.
  Then the secondary ones: employee reviews (kununu is the strong one for German-speaking
  companies, Glassdoor elsewhere), levels.fyi, Reddit, trade press.
- Prefer recent material, and say how recent a claim is when it could go stale — funding,
  headcount, leadership, strategy and compensation all move.
- Write the report in English no matter what language the sources are in, translating
  quotes rather than dropping them.
- Keep the URL of anything you use; every factual claim of substance should be traceable.
- Distinguish what you found from what you inferred. "No public information on this" is a
  useful sentence; an invented specific is a harmful one. Never invent a number, a name, a
  date or a link.
- Report what you find, including what is unflattering. A briefing that reads like the
  company's own recruiting page has failed."""


def get_company_research_prompt(
    company: str,
    role: str = "",
    focus: str = "",
    now: datetime | None = None,
) -> str:
    """User prompt for a company research report. Returns Markdown, not JSON.

    `now` anchors "recent" — a model reasoning about the last 6-12 months needs
    to be told when it is. Injectable for tests.
    """
    now = now or datetime.now()
    role_block = (
        "\n\nANGLE: this briefing will be read before an interview for the role of"
        f"\n{role}. Let that decide which part of the business, which org and which"
        "\ntechnology you go deep on, and which compensation bands you look up. It does"
        "\nNOT change what kind of document this is: see the boundary below."
        if role
        else ""
    )
    focus_block = f"\n\nASKED FOR SPECIFICALLY:\n{focus}" if focus else ""
    return f"""Research {company} and write an interview-preparation briefing on the company.

Today is {now.strftime("%d %B %Y")}. "Recent" means the last 6-12 months from that
date.{role_block}{focus_block}

THE BOUNDARY OF THIS DOCUMENT

This is a briefing about a company, not an assessment of a candidate. You have not been
given a CV and must not ask for one or imagine one.

- Never write in the second person, and never refer to "the candidate", "the reader",
  "your background" or "your experience".
- No fit scores, no "X/10", no bridges between someone's experience and the job, no gap
  or red-flag analysis, no advice on how to play the interview.
- Do not rank several open roles against anybody. You may describe what roles exist and
  what they involve — that is a fact about the company.
- Every sentence must be a claim about {company}, its market, its people or its process.

WHAT TO RESEARCH

The product and how it earns money; who buys it and why; the named competitors and where
this company genuinely wins or loses against them; recent news, funding, results and
strategy shifts; the values the company publishes and what employees report instead; the
engineering and research footprint — publications, open source, conference presence, the
technical leadership and what they say in public; how the company hires and what its
process is reported to be; what its roles pay.

THE REPORT

Write Markdown, starting with a level-1 heading and nothing before it. Use these sections:

# {company} — Company Briefing
*Researched: {now.strftime("%d %B %Y")}*{" · *Angle: " + role + "*" if role else ""}

## At a glance
3-6 bullets: what they do, size, where they are, founded or funding stage, and anything
someone walking into an interview there would be embarrassed not to know.

## The business
What a customer actually buys and what it does for them — concretely enough to be
explained out loud without hand-waving. How the money is made. The market and the
named competitors, with the reason this company wins or loses against each, not just the
positioning. Recent developments worth being aware of.

## Culture and values
What working there is reported to be like. The company's stated values against what
employees actually say, as patterns rather than single reviews. If the company has its own
named framework — an "X Formula", leadership principles, operating principles, a culture
memo — quote its actual principles in the company's own words, because companies that
publish one screen candidates against it. Then say what the framework reveals about what
they truly reward: speed against rigour, invention against reliability, cooperation
against internal competition, growth at any cost against something else. Name what is
conspicuously absent from it too. What makes someone successful there, and what gets
people stuck.

## Reputation — technical, research, general
Where they sit in the competitive landscape and how selective they are said to be. Their
innovation record and R&D footprint: papers, open source, engineering blog, conference
presence. The notable technical leaders and their public positions. Be honest about tier —
"solid but not frontier" is a legitimate finding and a useful one.

## Hiring and interview process
The stages this company actually runs for roles like this, in order, and what each one is
reported to test. Typical timeline. What candidates report about the experience, including
the complaints. Anything unusual about how they assess.

## Compensation
Public bands for this kind of role at this company and location, with the source for each.
Base against total, and how equity or bonus is structured. Say plainly when the data is
thin or is extrapolated from a neighbouring level or city.

## Questions to ask them
2-4 questions someone could ask in the interview that show real knowledge of this company.
Each must be tied to something concrete in the sections above — a strategy shift, a
technical choice, a tension in the values. No generic questions; "what is the culture
like?" is a wasted line.

## Sources
The URLs behind the claims above, grouped so a reader can go deeper on one section.

Drop a section entirely if the research turned up nothing real for it, and say so in one
line. A short honest briefing beats a complete-looking padded one. State the important
finding in each section directly, in the first sentence — never withhold it as a tease.
"""
