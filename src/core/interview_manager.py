"""Business logic: orchestrates planning, interviewing, and evaluation.

Frontend-agnostic — no streamlit imports. A future web API would wrap the
three public methods (start, reply, finish) in HTTP endpoints unchanged.
"""

from dataclasses import replace
from typing import Optional

from src.core import prompts_common, prompts_interview, prompts_preparation
from src.models.interview import (
    CANDIDATE,
    INTERVIEWER,
    TEXT,
    Evaluation,
    InterviewPlan,
    InterviewSession,
    InterviewSettings,
    Turn,
)
from src.services.llm_service import LLMService
from src.storage.interview_storage import InterviewStorage

# How far the model's holistic score may sit from the one its own checklist
# ratings imply. Unanchored, the number drifts between rounds and the progress
# tracking it feeds is meaningless.
SCORE_TOLERANCE = 10


class InterviewManager:
    def __init__(self, llm: LLMService, storage: InterviewStorage):
        self.llm = llm
        self.storage = storage

    def start(
        self,
        settings: InterviewSettings,
        use_history: bool = True,
        replay_of: Optional[InterviewSession] = None,
    ) -> tuple[InterviewSession, int]:
        """Generate a plan (informed by previous rounds) and open the interview.

        Returns (session, number_of_previous_rounds).

        Static context files dropped in data/context/<company>/ are kept in
        `company_context`, separate from the candidate's own background: the
        planner distills them into a short company brief, and only that brief
        reaches the live interviewer.

        Two model calls: the plan, then the opening message. The opening is not
        planned because it has to carry the round's mood, which is drawn from a
        pool the plan call produces.

        `use_history=False` runs the round as if nothing had ever been practiced
        for these settings. Static context files still load — they are the
        candidate's preparation material, not a record of previous rounds.

        `replay_of` re-enters a stored round instead of planning a new one: its
        plan and round profile are reused verbatim, so the planning call does not
        happen at all and the candidate faces the same prepared questions. Only
        the opening is generated, because it is a live turn. See `_history` for
        what such a round is shown of the history.
        """
        if replay_of is None:
            static_context = self.storage.load_company_context(settings.company)
            if static_context:
                settings = replace(settings, company_context=static_context)
        # Skipped when replaying: the source's settings already carry the
        # company_context snapshot, and only the planner — which is not run —
        # ever reads it.
        previous = self._history(
            settings, use_history, replay_of.id if replay_of else None
        )
        digest = prompts_preparation.history_digest(previous)
        if replay_of is not None:
            # Copies, so the new session never aliases the stored one the caller
            # is still holding. The profile already carries the curveball and the
            # mood drawn for that round, which is what a replay wants — and the
            # pool they came from is not persisted, so re-drawing is impossible.
            plan = InterviewPlan.from_dict(replay_of.plan.to_dict())
            profile = dict(replay_of.round_profile)
        else:
            last_score = (
                previous[-1].evaluation.score
                if previous and previous[-1].evaluation
                else None
            )
            profile = prompts_common.pick_round_profile(previous, last_score)
            # Who runs this stage and what it is for. The prompt and the schema
            # are gated by the same lookup: a round not asked for a curveball in
            # the one must not be required to return one by the other.
            profession = prompts_common.resolve_profession(settings.stage)
            plan_prompt, session_time = prompts_preparation.get_plan_prompt(
                settings,
                digest,
                profile,
                prompts_preparation.core_checklist_block(previous),
                prompts_common.recent_scenarios(previous),
            )
            plan_data = self.llm.complete_json(
                system="You design realistic, well-calibrated job interviews.",
                messages=[{"role": "user", "content": plan_prompt}],
                schema=prompts_preparation.plan_schema(profession["curveball"]),
            )
            # The planner writes the round's scenario (it has to fit the role);
            # record it on the profile so the next rounds can be told not to
            # reuse it.
            profile["curveball"] = plan_data.get("curveball", "")
            # Two of the ten moods the planner wrote for this hour. Stored on the
            # profile before the session is built, so it persists and every later
            # turn is conducted in the same mood.
            profile["mood"] = prompts_common.pick_mood(
                plan_data.get("mood_adjectives", [])
            )
            plan = InterviewPlan.from_dict(plan_data)
            # Not asked of the model: it is the hour we told the planner about,
            # not something for it to write.
            plan.session_time = session_time

            plan.checklist.append(replace(prompts_common.STRUCTURE_CRITERION))
            # Snapshotted rather than re-resolved on display: editing the
            # constant must not change what a round already sat is described as.
            plan.interviewer_profession = prompts_common.profession_block(profession)
            plan.round_purpose = prompts_common.round_purpose_block(profession)
        session = InterviewSession.new(
            settings,
            plan,
            profile,
            replay_of=replay_of.id if replay_of else None,
            use_history=use_history,
        )
        session.transcript.append(Turn(INTERVIEWER, self._opening(session, digest)))
        return session, len(previous)

    def _history(
        self,
        settings: InterviewSettings,
        use_history: bool = True,
        before: Optional[str] = None,
    ) -> list[InterviewSession]:
        """The stored rounds a session is allowed to see.

        A replay is cut off strictly before its source: showing it the source
        would put the very questions it is about to ask on the live
        interviewer's "do not reuse" list.
        """
        if not use_history:
            return []
        return self.storage.load_sessions(settings, before=before)

    def _session_history(self, session: InterviewSession) -> list[InterviewSession]:
        """`_history` for a round already under way — same scope on every turn."""
        return self._history(session.settings, session.use_history, session.replay_of)

    def _opening(self, session: InterviewSession, digest: str) -> str:
        """The interviewer's first message, spoken in this round's mood."""
        system = prompts_interview.get_interviewer_system_prompt(
            session.settings, session.plan, digest, session.round_profile, answered=0
        )
        # An empty transcript still yields the synthetic "candidate has joined"
        # user turn the API needs as the first message.
        raw = self.llm.complete(system, prompts_interview.turns_to_messages([]))
        # The prompt forbids closing on the opening turn; strip the token anyway
        # rather than show it to the candidate if the model emits it.
        return raw.replace(prompts_interview.COMPLETE_TOKEN, "").strip()

    def reply(
        self, session: InterviewSession, candidate_text: str, modality: str = TEXT
    ) -> tuple[str, bool]:
        """Process a candidate answer; returns (interviewer_reply, interview_complete)."""
        candidate_turn = Turn(CANDIDATE, candidate_text, modality)
        previous = self._session_history(session)
        digest = prompts_preparation.history_digest(previous)
        answered = sum(1 for t in session.transcript if t.role == CANDIDATE) + 1
        # The round is paced by what has been said, so the answer being handled
        # counts towards the clock it is judged against.
        spoken = prompts_common.spoken_minutes(session.transcript + [candidate_turn])
        _, may_close = prompts_interview.interview_phase(
            spoken, answered, session.settings.duration_minutes
        )
        system = prompts_interview.get_interviewer_system_prompt(
            session.settings,
            session.plan,
            digest,
            # Drawn once at start and carried on the session: re-drawing here
            # would hand the candidate a different interviewer every turn.
            session.round_profile,
            answered,
            spoken,
        )
        # Only mutate the transcript after the LLM call succeeds, so a failed
        # call leaves the session in a retryable state.
        raw = self.llm.complete(
            system,
            prompts_interview.turns_to_messages(session.transcript + [candidate_turn]),
        )
        # The prompt forbids closing before the candidate has had their questions
        # phase; enforced here too, because a model that closes early costs the
        # candidate a criticism they had no way to avoid.
        complete = may_close and prompts_interview.COMPLETE_TOKEN in raw
        reply = raw.replace(prompts_interview.COMPLETE_TOKEN, "").strip()
        session.transcript.append(candidate_turn)
        session.transcript.append(Turn(INTERVIEWER, reply))
        return reply, complete

    def undo_last_answer(self, session: InterviewSession) -> bool:
        """Drop the last candidate answer and the reply it drew; True if anything went.

        Puts the candidate back in front of the question they were just asked, so
        a fluffed or mis-transcribed answer does not have to poison the
        evaluation. Only the transcript changes — the round profile, plan and
        settings stay, and nothing has reached disk yet (sessions are persisted
        in `finish` only). The caller must clear `interview_complete`.
        """
        if len(session.transcript) < 3 or session.transcript[-2].role != CANDIDATE:
            return False
        del session.transcript[-2:]
        return True

    def finish(self, session: InterviewSession) -> Evaluation:
        """Evaluate the interview, attach the result, and persist the session."""
        previous = self._session_history(session)
        previous_evaluations = [s.evaluation for s in previous if s.evaluation]
        data = self.llm.complete_json(
            system="You are a rigorous, fair interview evaluator.",
            messages=[
                {
                    "role": "user",
                    "content": prompts_interview.get_evaluation_prompt(
                        session.settings,
                        session.plan,
                        session.transcript,
                        previous_evaluations,
                    ),
                }
            ],
            schema=prompts_interview.evaluation_schema(bool(previous_evaluations)),
        )
        evaluation = Evaluation.from_dict(data)
        evaluation.score = _anchor_score(evaluation)
        session.evaluation = evaluation
        self.storage.save_session(session)
        return evaluation


def _anchor_score(evaluation: Evaluation) -> int:
    """Clamp the reported score to the band its own checklist ratings support."""
    score = max(0, min(100, evaluation.score))
    implied = evaluation.rubric_score()
    if implied is None:
        return score
    return max(implied - SCORE_TOLERANCE, min(implied + SCORE_TOLERANCE, score))
