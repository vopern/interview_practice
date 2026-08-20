from src.models.interview import (
    CandidateQuestion,
    CandidateQuestionsReview,
    ChecklistItem,
    ChecklistResult,
    Evaluation,
    InterviewPlan,
    InterviewSession,
    InterviewSettings,
    KeyTakeaway,
    Turn,
)


def make_session() -> InterviewSession:
    settings = InterviewSettings(
        "Acme Corp", "Data Scientist", "HR Screen", "10 years of Python"
    )
    plan = InterviewPlan(
        checklist=[ChecklistItem("motivation", "Motivation", "Knows why Acme")],
        questions=["Why Acme?", "Tell me about yourself"],
        interviewer_role="You screen for the rockets team. Know when they can start.",
    )
    session = InterviewSession.new(settings, plan)
    session.transcript = [
        Turn("interviewer", "Welcome! Why Acme?"),
        Turn("candidate", "Because rockets."),
    ]
    session.evaluation = Evaluation(
        score=72,
        results=[
            ChecklistResult("motivation", "Motivation", "partial", "rockets", "thin")
        ],
        strengths=["concise"],
        improvements=["give examples"],
        summary="Mixed round.",
        progress_notes="",
        candidate_questions=CandidateQuestionsReview(
            rating="weak",
            impression="You asked only about the package.",
            questions=[
                CandidateQuestion("How many vacation days?", "weak", "Perks only."),
            ],
            better_questions=["What does success look like in the first 90 days?"],
        ),
        key_takeaway=KeyTakeaway(
            point="Be specific: name the number you moved",
            severity=2,
            evidence='"we improved things a lot" — twice, no figure either time',
            verdict="Round two told you the same thing.",
        ),
    )
    return session


def test_session_json_roundtrip():
    session = make_session()
    restored = InterviewSession.from_dict(session.to_dict())
    assert restored.to_dict() == session.to_dict()
    assert restored.settings.company == "Acme Corp"
    assert restored.evaluation.score == 72
    assert restored.plan.checklist[0].id == "motivation"
    assert len(restored.transcript) == 2
    questions = restored.evaluation.candidate_questions
    assert questions.rating == "weak"
    assert questions.questions[0].signal == "weak"
    assert questions.better_questions == [
        "What does success look like in the first 90 days?"
    ]
    takeaway = restored.evaluation.key_takeaway
    assert takeaway.point == "Be specific: name the number you moved"
    assert takeaway.severity == 2


def test_plan_without_an_interviewer_loads():
    """Plans stored before these existed predate the fields entirely."""
    plan = InterviewPlan.from_dict({"checklist": [], "questions": ["Why Acme?"]})

    assert plan.interviewer_role == ""
    assert plan.interviewer_profession == ""
    assert plan.round_purpose == ""


def test_who_ran_the_round_survives_a_round_trip():
    """Stored rather than re-resolved from the stage, so editing
    INTERVIEWER_PROFESSIONS cannot change what a round the candidate already sat
    is described as."""
    plan = InterviewPlan.from_dict(
        {
            "checklist": [],
            "questions": [],
            "interviewer_profession": "You are a recruiter.",
            "round_purpose": "What this round was for.",
        }
    )
    restored = InterviewPlan.from_dict(plan.to_dict())

    assert restored.interviewer_profession == "You are a recruiter."
    assert restored.round_purpose == "What this round was for."


def test_a_plan_stored_under_the_old_key_names_still_loads():
    """`interviewer_remit` and `stage_remit` are what every session already on
    disk carries. `_load_file` skips a file that raises and drops a field it does
    not recognise, so without this the stored round loses its interviewer — from
    the History tab and from every later round's digest."""
    plan = InterviewPlan.from_dict(
        {
            "checklist": [],
            "questions": [],
            "interviewer_remit": "You screen for the rockets team.",
            "stage_remit": "What this round was for.",
        }
    )

    assert plan.interviewer_role == "You screen for the rockets team."
    assert plan.round_purpose == "What this round was for."
    # Rewritten under the new names once it is saved again.
    assert plan.to_dict()["interviewer_role"] == "You screen for the rockets team."


def test_evaluation_without_a_key_takeaway_loads():
    """Sessions stored before the takeaway existed still load, without one."""
    session = make_session()
    stored = session.to_dict()
    del stored["evaluation"]["key_takeaway"]

    restored = InterviewSession.from_dict(stored)

    assert restored.evaluation.key_takeaway is None


def test_key_takeaway_keeps_its_quoted_evidence_and_clamps_severity():
    """`point` is prose the model likes to wrap; `evidence` is often one quote."""
    takeaway = KeyTakeaway.from_dict(
        {
            "point": '"Show agency."',
            "severity": "4",
            "evidence": '"the team decided"',
        }
    )
    assert takeaway.point == "Show agency."
    assert takeaway.evidence == '"the team decided"'
    assert takeaway.severity == 3
    assert KeyTakeaway.from_dict({"point": "x", "severity": None}).severity == 2


def test_evaluation_without_candidate_questions_loads():
    """Sessions stored before closing questions were evaluated still load."""
    session = make_session()
    stored = session.to_dict()
    del stored["evaluation"]["candidate_questions"]

    restored = InterviewSession.from_dict(stored)

    assert restored.evaluation.candidate_questions is None
    assert restored.evaluation.score == 72


def test_session_without_history_scope_loads():
    """Sessions stored before replays existed predate both scope fields."""
    session = make_session()
    stored = session.to_dict()
    del stored["replay_of"]
    del stored["use_history"]

    restored = InterviewSession.from_dict(stored)

    assert restored.replay_of is None
    assert restored.use_history is True


def test_history_scope_survives_the_roundtrip():
    session = make_session()
    session.replay_of = "20260101_090000"
    session.use_history = False
    restored = InterviewSession.from_dict(session.to_dict())
    assert restored.replay_of == "20260101_090000"
    assert restored.use_history is False


def test_session_without_evaluation_roundtrip():
    session = make_session()
    session.evaluation = None
    restored = InterviewSession.from_dict(session.to_dict())
    assert restored.evaluation is None


def test_slug_sanitization():
    settings = InterviewSettings(
        "Späti & Co.  GmbH", "ML Engineer (Senior)", "Final Round"
    )
    slug = settings.slug
    assert slug == slug.lower()
    assert " " not in slug and "&" not in slug and "(" not in slug
    # Same settings always map to the same slug
    assert (
        InterviewSettings(
            "Späti & Co.  GmbH", "ML Engineer (Senior)", "Final Round"
        ).slug
        == slug
    )
