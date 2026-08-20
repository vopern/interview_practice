"""Domain model for interview sessions.

Plain dataclasses with JSON round-trip serialization. These define the on-disk
schema (and the future DB/API schema) — keep them free of any I/O or framework
dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

INTERVIEWER = "interviewer"
CANDIDATE = "candidate"

# How a candidate turn was produced. Voice answers go through speech-to-text and
# therefore carry transcription errors, which the evaluator must not hold
# against the candidate — see prompts_interview.TRANSCRIPTION_CAVEAT.
TEXT = "text"
VOICE = "voice"

STAGES = [
    "HR Screen",
    "Hiring Manager",
    "Technical",
    "Behavioral",
    "Case Study",
    "Final Round",
]

RATINGS = ["met", "partial", "not_met", "not_assessed"]

# How fully an answer responded to the question it was given.
ADDRESSED = ["full", "partial", "avoided"]

# What one question the candidate asked at the end signals to the interviewer.
CANDIDATE_QUESTION_SIGNALS = ["strong", "reasonable", "weak", "red_flag"]

# The overall read on the candidate's own questions. "not_assessed" covers the
# case where the interviewer never actually opened the floor — see the fairness
# rule in prompts_interview.get_evaluation_prompt.
CANDIDATE_QUESTIONS_RATINGS = ["strong", "adequate", "weak", "not_assessed"]

# How damaging the one thing the candidate should take away is — how loudly an
# interviewer would raise it in the debrief, not how hard it is to fix.
TAKEAWAY_SEVERITIES = [1, 2, 3]

# Weight per rating when deriving the rubric-implied score (not_assessed items
# are excluded from the average rather than counted as failures).
RATING_WEIGHTS = {"met": 1.0, "partial": 0.5, "not_met": 0.0}


def _unquote(text: str) -> str:
    """Drop quotation marks wrapping a whole field value.

    Models return prose fields wrapped in quotes, which then show up verbatim in
    the UI. Only for fields that are never legitimately one quotation: `evidence`
    is frequently a single verbatim quote of the candidate and keeps its marks.
    """
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] in '"“' and stripped[-1] in '"”':
        return stripped[1:-1].strip()
    return text


@dataclass
class InterviewSettings:
    company: str
    role: str
    stage: str
    background: str = ""
    duration_minutes: int = 30
    # Static research files from data/context/<company>/. Kept apart from
    # `background` on purpose: `background` is what the interviewer would
    # legitimately hold (CV, job ad), while this is the candidate's own prep
    # material and must not be treated as a script of correct answers.
    company_context: str = ""

    @property
    def slug(self) -> str:
        # Deliberately excludes duration_minutes: a shorter or longer round of
        # the same interview shares one practice history.
        raw = f"{self.company}_{self.role}_{self.stage}".lower()
        return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", raw)).strip("_")

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "role": self.role,
            "stage": self.stage,
            "background": self.background,
            "duration_minutes": self.duration_minutes,
            "company_context": self.company_context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InterviewSettings":
        return cls(
            company=d["company"],
            role=d["role"],
            stage=d["stage"],
            background=d.get("background", ""),
            duration_minutes=int(d.get("duration_minutes", 30)),
            company_context=d.get("company_context", ""),
        )


@dataclass
class ChecklistItem:
    id: str
    criterion: str
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "criterion": self.criterion,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChecklistItem":
        return cls(
            id=str(d["id"]),
            criterion=d["criterion"],
            description=d.get("description", ""),
        )


@dataclass
class InterviewPlan:
    checklist: list[ChecklistItem]
    questions: list[str]
    # Company/team facts distilled once at plan time. The live interviewer gets
    # this instead of the raw context files, which are far too long to re-send
    # on every turn.
    company_brief: str = ""
    # The company's own named values/principles (a published leadership-
    # principles list, a culture memo...). Companies that publish one screen
    # against it explicitly, so the interview has to as well.
    culture_anchors: list[str] = field(default_factory=list)
    # This company's own version of the professional running the stage: their
    # real title here, and what they are accountable for establishing before
    # passing the candidate on. The job, as distinct from the manner the
    # archetype gives them. Carried to the live interviewer, so improvised
    # follow-ups and end-of-round triage serve it too. Declared after the
    # required fields because the earlier ones are constructed positionally.
    interviewer_role: str = ""
    # The generic halves of the same thing, rendered from
    # `prompts_common.INTERVIEWER_PROFESSIONS`: who runs a round of this stage
    # anywhere, and what a round of it is for. Snapshotted here rather than
    # re-resolved on display, like the archetype's `style` in `round_profile` —
    # editing the constant must not change what a stored round is described as.
    interviewer_profession: str = ""
    round_purpose: str = ""
    session_time: str = ""

    def to_dict(self) -> dict:
        return {
            "checklist": [c.to_dict() for c in self.checklist],
            "questions": list(self.questions),
            "company_brief": self.company_brief,
            "culture_anchors": list(self.culture_anchors),
            "interviewer_role": self.interviewer_role,
            "interviewer_profession": self.interviewer_profession,
            "round_purpose": self.round_purpose,
            "session_time": self.session_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InterviewPlan":
        return cls(
            checklist=[ChecklistItem.from_dict(c) for c in d.get("checklist", [])],
            questions=list(d.get("questions", [])),
            company_brief=d.get("company_brief", ""),
            culture_anchors=list(d.get("culture_anchors", [])),
            # Some sessions on disk carry the older key names. Missing keys must
            # never raise: `_load_file` skips a file that does, which would drop
            # the round from the History tab and from every later round's digest.
            interviewer_role=d.get("interviewer_role", d.get("interviewer_remit", "")),
            interviewer_profession=d.get("interviewer_profession", ""),
            round_purpose=d.get("round_purpose", d.get("stage_remit", "")),
            session_time=d.get("session_time", ""),
        )


@dataclass
class Turn:
    role: str  # INTERVIEWER or CANDIDATE
    content: str
    modality: str = TEXT  # TEXT or VOICE (candidate turns only)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "modality": self.modality}

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(
            role=d["role"], content=d["content"], modality=d.get("modality", TEXT)
        )


@dataclass
class ChecklistResult:
    id: str
    criterion: str
    rating: str  # one of RATINGS
    evidence: str = ""
    comment: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "criterion": self.criterion,
            "rating": self.rating,
            "evidence": self.evidence,
            "comment": self.comment,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChecklistResult":
        return cls(
            id=str(d["id"]),
            criterion=d["criterion"],
            rating=d.get("rating", "not_assessed"),
            evidence=d.get("evidence", ""),
            comment=d.get("comment", ""),
        )


@dataclass
class AnswerReview:
    """How one answer fared against the question it was actually asked.

    Separate from `ChecklistResult`: a criterion can be evidenced by anything
    anywhere in the transcript, but answering the question in front of you is
    its own skill.
    """

    question: str
    addressed: str  # one of ADDRESSED
    comment: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "addressed": self.addressed,
            "comment": self.comment,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnswerReview":
        return cls(
            question=d.get("question", ""),
            addressed=d.get("addressed", "full"),
            comment=d.get("comment", ""),
        )


@dataclass
class CandidateQuestion:
    """One question the candidate asked the interviewer, and what it signalled."""

    question: str
    signal: str  # one of CANDIDATE_QUESTION_SIGNALS
    comment: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "signal": self.signal,
            "comment": self.comment,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateQuestion":
        return cls(
            question=d.get("question", ""),
            signal=d.get("signal", "reasonable"),
            comment=d.get("comment", ""),
        )


@dataclass
class CandidateQuestionsReview:
    """The read on the questions the candidate asked at the end.

    What a candidate chooses to ask is evidence in its own right: what they
    researched, what they optimise for, and who they think they are talking to.
    `impression` is the inference a real interviewer would draw from it.
    """

    rating: str  # one of CANDIDATE_QUESTIONS_RATINGS
    impression: str = ""
    questions: list[CandidateQuestion] = field(default_factory=list)
    # Questions that would have landed better, specific to this company and role.
    better_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rating": self.rating,
            "impression": self.impression,
            "questions": [q.to_dict() for q in self.questions],
            "better_questions": list(self.better_questions),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateQuestionsReview":
        return cls(
            rating=d.get("rating", "not_assessed"),
            impression=_unquote(d.get("impression", "")),
            questions=[CandidateQuestion.from_dict(q) for q in d.get("questions", [])],
            better_questions=list(d.get("better_questions", [])),
        )


@dataclass
class KeyTakeaway:
    """The single thing to fix before the real interview.

    One object rather than another entry in `improvements`, because it is a
    choice: the one line an interviewer would give in the debrief as the reason
    not to hire. `severity` is how loudly it would be raised, not how hard it is
    to fix, and it feeds nothing but the UI's callout colour — the score comes
    from the checklist alone (see `Evaluation.rubric_score`).
    """

    point: str  # the instruction itself, imperative and short enough to quote
    severity: int  # one of TAKEAWAY_SEVERITIES: 1 light concern .. 3 red flag
    # What makes it recognisable — often a pattern across answers rather than a
    # single moment, and frequently a verbatim quote, hence never _unquote'd.
    evidence: str = ""
    # The debrief-room reasoning, including whether an earlier round already
    # said this and the candidate did it again.
    verdict: str = ""

    def to_dict(self) -> dict:
        return {
            "point": self.point,
            "severity": self.severity,
            "evidence": self.evidence,
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeyTakeaway":
        try:
            severity = int(d.get("severity", 2))
        except (TypeError, ValueError):
            severity = 2
        return cls(
            point=_unquote(d.get("point", "")),
            severity=min(max(severity, 1), 3),
            evidence=d.get("evidence", ""),
            verdict=_unquote(d.get("verdict", "")),
        )


@dataclass
class Evaluation:
    score: int
    results: list[ChecklistResult]
    strengths: list[str]
    improvements: list[str]
    summary: str
    progress_notes: str = ""
    # Per-question responsiveness: did each answer address what was asked.
    answer_review: list[AnswerReview] = field(default_factory=list)
    # Stories and situations the candidate actually used, so the next round can
    # demand fresh material instead of only avoiding the old question wording.
    topics_covered: list[str] = field(default_factory=list)
    # The questions the candidate asked at the end, judged on content. None for
    # sessions stored before this was evaluated.
    candidate_questions: Optional[CandidateQuestionsReview] = None
    # The one thing to fix before the real interview. None for sessions stored
    # before this was evaluated; the UI skips the section entirely then.
    key_takeaway: Optional[KeyTakeaway] = None

    def rubric_score(self) -> Optional[int]:
        """Score implied by the checklist ratings alone, or None when it says little.

        What the model's drifting holistic score is anchored to (see
        `interview_manager._anchor_score`). Returns None when fewer than half the
        criteria were actually assessed: too small a sample to pin a score to,
        and averaging over it would reward a candidate for topics the
        interviewer never raised.
        """
        weights = [
            RATING_WEIGHTS[r.rating] for r in self.results if r.rating in RATING_WEIGHTS
        ]
        if not weights or len(weights) * 2 < len(self.results):
            return None
        return round(100 * sum(weights) / len(weights))

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "results": [r.to_dict() for r in self.results],
            "strengths": list(self.strengths),
            "improvements": list(self.improvements),
            "summary": self.summary,
            "progress_notes": self.progress_notes,
            "topics_covered": list(self.topics_covered),
            "answer_review": [a.to_dict() for a in self.answer_review],
            "candidate_questions": (
                self.candidate_questions.to_dict() if self.candidate_questions else None
            ),
            "key_takeaway": (
                self.key_takeaway.to_dict() if self.key_takeaway else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Evaluation":
        return cls(
            score=int(d["score"]),
            results=[ChecklistResult.from_dict(r) for r in d.get("results", [])],
            strengths=list(d.get("strengths", [])),
            improvements=list(d.get("improvements", [])),
            summary=_unquote(d.get("summary", "")),
            progress_notes=_unquote(d.get("progress_notes", "")),
            topics_covered=list(d.get("topics_covered", [])),
            answer_review=[
                AnswerReview.from_dict(a) for a in d.get("answer_review", [])
            ],
            candidate_questions=(
                CandidateQuestionsReview.from_dict(d["candidate_questions"])
                if d.get("candidate_questions")
                else None
            ),
            key_takeaway=(
                KeyTakeaway.from_dict(d["key_takeaway"])
                if d.get("key_takeaway")
                else None
            ),
        )


@dataclass
class InterviewSession:
    id: str
    created_at: str
    settings: InterviewSettings
    plan: InterviewPlan
    transcript: list[Turn] = field(default_factory=list)
    evaluation: Optional[Evaluation] = None
    # The persona/scenario/pressure drawn for this round. Stored rather than
    # recomputed: it is a random draw, so the live interviewer would otherwise
    # change personality between turns, and later rounds could not avoid it.
    round_profile: dict = field(default_factory=dict)
    # Which stored rounds this one is allowed to see. Both are recorded rather
    # than passed around because `reply` and `finish` re-read the history on
    # every call and must resolve the same scope the round was started with.
    #
    # The id of the session this round replays, and the cut-off for its history:
    # a replay sees what its source saw, never the source itself.
    replay_of: Optional[str] = None
    # False for a round deliberately started as if nothing had been practiced.
    use_history: bool = True

    @classmethod
    def new(
        cls,
        settings: InterviewSettings,
        plan: InterviewPlan,
        round_profile: Optional[dict] = None,
        replay_of: Optional[str] = None,
        use_history: bool = True,
    ) -> "InterviewSession":
        now = datetime.now()
        return cls(
            id=now.strftime("%Y%m%d_%H%M%S"),
            created_at=now.isoformat(timespec="seconds"),
            settings=settings,
            plan=plan,
            round_profile=round_profile or {},
            replay_of=replay_of,
            use_history=use_history,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "settings": self.settings.to_dict(),
            "plan": self.plan.to_dict(),
            "transcript": [t.to_dict() for t in self.transcript],
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "round_profile": dict(self.round_profile),
            "replay_of": self.replay_of,
            "use_history": self.use_history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InterviewSession":
        return cls(
            id=d["id"],
            created_at=d["created_at"],
            settings=InterviewSettings.from_dict(d["settings"]),
            plan=InterviewPlan.from_dict(d["plan"]),
            transcript=[Turn.from_dict(t) for t in d.get("transcript", [])],
            evaluation=(
                Evaluation.from_dict(d["evaluation"]) if d.get("evaluation") else None
            ),
            round_profile=dict(d.get("round_profile", {})),
            replay_of=d.get("replay_of"),
            use_history=bool(d.get("use_history", True)),
        )
