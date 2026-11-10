"""Finite task graph and recovery menu. All outputs are recommendations."""

from dataclasses import replace
import uuid

from .evidence import Facts
from .models import (
    Config,
    Decision,
    Event,
    Memory,
    PROMPTS,
    STAGE_SKILL,
    Snapshot,
    Stage,
)


PICK = (Stage.S2, Stage.S6)
CLEAR = ("left_empty", "left_clear", "ledger_resolved")
RETURN = (*CLEAR, "right_stable", "plate_available", "return_start_valid")
ENTRY = {
    Stage.S1: ("target_ready", "truss_on_plate", *CLEAR),
    Stage.S2: ("right_stable", "at_pick_region", "fruit_remaining", *CLEAR),
    Stage.S3: RETURN,
    Stage.S4: ("truss_on_plate", "right_released", *CLEAR, "reposition_benefit"),
    Stage.S5: ("truss_on_plate", "reposition_ready", *CLEAR),
    Stage.S6: ("right_stable", "at_pick_region", "fruit_remaining", *CLEAR),
    Stage.S7: (*RETURN, "no_remaining"),
}


class Supervisor:
    def __init__(
        self,
        episode_id: str,
        now: float,
        config: Config | None = None,
        *,
        simulation=False,
        available_skills=(),
    ):
        self.config = config or Config()
        self.memory = Memory(episode_id, now, stage_since=now, last_pick_progress=now)
        self.simulation = simulation
        self.skills = set(PROMPTS) if simulation else set(available_skills)
        if not self.skills <= set(PROMPTS):
            raise ValueError("unknown skill registration")
        self.last_now = now
        self.clock_id = None
        self.clock_fault = False
        self.observing_since = None
        self.issued = {}
        self.review_latch = None

    def ingest(self, snap: Snapshot, facts: Facts, events: list[Event]):
        m = self.memory
        if snap.episode_id != m.episode_id:
            raise ValueError("episode mismatch")
        if facts.observation_id not in (None, snap.observation_id):
            raise ValueError("facts belong to a different observation")
        if snap.now < self.last_now or self.clock_id not in (None, snap.clock_id):
            self.clock_fault = True
        self.last_now, self.clock_id = snap.now, snap.clock_id
        if snap.invalid_reason(self.config) or self.clock_fault or m.outcome:
            return
        m.damage_detected |= facts.is_true("damage")
        m.drop_detected |= facts.is_true("drop")
        if m.recovery_skill == "disengage_left" and facts.is_true("left_clear"):
            m.recovery_skill = None
        if (
            m.recovery_skill == "deposit_held_fruit"
            and facts.is_true("left_empty")
            and facts.is_true("ledger_resolved")
            and m.separated <= m.deposited
        ):
            m.recovery_skill = None
        for event in events:
            if not event.event_id or not event.references:
                raise ValueError("event requires identity and evidence references")
            if event.observation_id != snap.observation_id:
                raise ValueError("event belongs to a different observation")
            previous = m.events.get(event.event_id)
            if previous:
                # Duplicate event reports may reference a later frame, not a new target/type.
                if (previous.kind, previous.target_id, previous.attempt_id) != (
                    event.kind,
                    event.target_id,
                    event.attempt_id,
                ):
                    raise ValueError("event identity reused with different content")
                continue
            if event.kind in {"SEPARATED", "DEPOSITED"}:
                predicate = (
                    "fruit_separated"
                    if event.kind == "SEPARATED"
                    else "deposit_verified"
                )
                if not event.target_id or not facts.is_true(predicate):
                    continue  # Unconfirmed event can be resubmitted with later evidence.
                if event.kind == "SEPARATED":
                    if event.target_id not in m.separated:
                        m.last_pick_progress = snap.now
                    m.separated.add(event.target_id)
                else:
                    if event.target_id not in m.separated:
                        raise ValueError("deposit requires a tracked separation event")
                    if event.target_id not in m.deposited:
                        m.last_pick_progress = snap.now
                        m.unassociated_failures = 0
                    m.deposited.add(event.target_id)
            elif event.kind in {"ATTEMPT_FAILED", "ATTEMPT_UNRESOLVED"}:
                if not event.attempt_id:
                    raise ValueError("attempt event requires attempt_id")
                if event.attempt_id not in m.attempts:
                    m.attempts.add(event.attempt_id)
                    if event.target_id:
                        if event.target_id in m.deposited:
                            raise ValueError("cannot retry a deposited target")
                        m.failures[event.target_id] = (
                            m.failures.get(event.target_id, 0) + 1
                        )
                    else:
                        m.unassociated_failures += 1
            else:
                raise ValueError(f"unsupported event kind: {event.kind}")
            m.events[event.event_id] = event

    def decide(self, snap: Snapshot, facts: Facts) -> Decision:
        m, cfg = self.memory, self.config

        def decision(
            kind,
            reason,
            *,
            stage=None,
            skill=None,
            required=(),
            outcome=None,
            exit_after_return=False,
        ):
            missing = facts.missing(required)
            if missing and kind in {"TRANSITION", "RECOVER", "FINISH"}:
                kind, reason = "OBSERVE", "prerequisites_not_confirmed: " + reason
            elif (
                skill and kind in {"TRANSITION", "RECOVER"} and skill not in self.skills
            ):
                kind, reason = "NEED_HELP", "skill_not_validated: " + skill
            return Decision(
                uuid.uuid4().hex,
                m.episode_id,
                m.epoch,
                snap.observation_id,
                snap.now,
                snap.now + cfg.decision_ttl_s,
                kind,
                reason,
                stage,
                skill,
                tuple(required),
                missing,
                outcome,
                tuple(
                    sorted(
                        {
                            ref
                            for key in required
                            for ref in facts.references.get(key, ())
                        }
                    )
                ),
                exit_after_return,
            )

        def move(stage, reason, *, extra=(), exiting=False):
            return decision(
                "TRANSITION",
                reason,
                stage=stage,
                skill=STAGE_SKILL[stage],
                required=(*ENTRY[stage], *extra),
                exit_after_return=exiting,
            )

        def observe(reason):
            return decision("OBSERVE", reason)

        def finish():
            resolved = m.separated <= m.deposited
            complete = (
                m.stage == Stage.S7
                and not m.damage_detected
                and not m.drop_detected
                and resolved
                and facts.is_false("damage")
                and facts.is_false("drop")
            )
            outcome = (
                "COMPLETED" if complete else ("PARTIAL" if m.deposited else "FAILED")
            )
            if not resolved or not facts.is_true("ledger_resolved"):
                outcome = "UNRESOLVED"
            return decision(
                "FINISH",
                "cleanup_and_ledger_checked",
                outcome=outcome,
                required=("stow_verified", "truss_on_plate", "right_released", *CLEAR),
            )

        invalid = snap.invalid_reason(cfg)
        if snap.episode_id != m.episode_id:
            raise ValueError("episode mismatch")
        if facts.observation_id not in (None, snap.observation_id):
            raise ValueError("facts belong to a different observation")
        if m.outcome:
            return decision("FINISHED", "episode_already_closed", outcome=m.outcome)
        if invalid or self.clock_fault:
            return decision("STOP", invalid or "clock_discontinuity")
        if facts.is_true("control_fault") or facts.is_true("contact_hazard"):
            return decision("STOP", "control_or_contact_hazard")
        if facts.is_false("execution_consistent"):
            return decision("STOP", "command_behavior_mismatch")
        if not all(facts.is_false(k) for k in ("control_fault", "contact_hazard")):
            return observe("control_and_contact_status_unknown")
        if not facts.is_true("execution_consistent"):
            return observe("execution_status_unknown")
        if facts.conflicts:
            return observe("conflicting_evidence")
        if m.damage_detected or m.drop_detected:
            return decision("NEED_HELP", "damage_or_drop_requires_review")
        if not all(facts.is_false(k) for k in ("damage", "drop")):
            return observe("quality_status_unknown")

        if m.recovery_skill:
            if snap.now - m.recovery_since >= cfg.stage_timeout_s:
                return decision("NEED_HELP", "recovery_timeout")
            return decision("CONTINUE", "await_recovery_result: " + m.recovery_skill)

        episode_expired = snap.now - m.started_at >= cfg.episode_timeout_s
        if episode_expired and m.stage not in (*PICK, Stage.S3, Stage.S7):
            return decision("NEED_HELP", "episode_budget_exhausted")
        if m.stage is None:
            return move(Stage.S1, "start_episode")
        if m.stage not in PICK and snap.now - m.stage_since >= cfg.stage_timeout_s:
            return decision("NEED_HELP", "stage_timeout")

        if m.stage in PICK:
            if facts.is_false("right_stable"):
                return decision("STOP", "right_support_unstable")
            if not facts.is_true("right_stable"):
                return observe("right_support_unknown")
            if facts.is_true("fruit_held_detached"):
                return decision(
                    "RECOVER",
                    "deposit_before_moving_truss",
                    skill="deposit_held_fruit",
                    required=(
                        "fruit_held_detached",
                        "right_stable",
                        "deposit_path_ready",
                    ),
                )
            if not facts.is_true("left_empty"):
                # Contacting an attached fruit is not an empty hand, but may be a valid ongoing pick.
                if not facts.is_false("fruit_held_detached"):
                    return observe("left_hand_contents_unknown")
            exhausted = episode_expired or m.repositions >= cfg.max_repositions
            if (
                snap.target_id
                and m.failures.get(snap.target_id, 0)
                >= cfg.max_total_attempts_per_target
            ):
                return decision("NEED_HELP", "target_attempt_budget_exhausted")
            target_failed = any(
                v - m.round_failure_baseline.get(k, 0) >= cfg.max_failures_per_target
                for k, v in m.failures.items()
            )
            # New approach round gets a local retry window; lifetime counts remain intact.
            stalled = (
                snap.now - max(m.last_pick_progress, m.stage_since) >= cfg.no_progress_s
                or target_failed
                or m.unassociated_failures >= cfg.max_unassociated_failures
            )
            if facts.is_true("no_remaining"):
                if m.separated - m.deposited:
                    return observe("unresolved_separated_fruit")
                return move(Stage.S7, "all_fruit_removed")
            if not facts.is_true("fruit_remaining"):
                return observe("remaining_fruit_unknown")
            if stalled or episode_expired or facts.is_true("reposition_needed"):
                if not facts.is_true("left_clear"):
                    if facts.is_false("left_clear"):
                        return decision(
                            "RECOVER",
                            "disengage_before_return",
                            skill="disengage_left",
                            required=("right_stable", "contact_releasable"),
                        )
                    return observe("left_clear_unknown")
                if m.separated - m.deposited:
                    return observe("unresolved_separated_fruit")
                if not exhausted and not facts.is_true("reposition_benefit"):
                    return decision("NEED_HELP", "no_validated_reposition_benefit")
                return move(
                    Stage.S3,
                    "budget_exit" if exhausted else "reposition_for_progress",
                    exiting=exhausted,
                )
            return decision("CONTINUE", "monitor_current_pick")

        if m.stage in (Stage.S1, Stage.S5):
            if facts.is_true("no_remaining"):
                if m.stage == Stage.S5:
                    return move(Stage.S7, "reassessed_empty_truss")
                return decision("NEED_HELP", "initial_truss_has_no_fruit")
            target = Stage.S2 if m.stage == Stage.S1 else Stage.S6
            if facts.is_true("right_stable") and facts.is_true("at_pick_region"):
                return move(target, "lift_verified")
        elif m.stage == Stage.S3:
            if facts.is_true("truss_on_plate") and facts.is_true("right_released"):
                if m.exit_after_return:
                    return finish()
                if m.repositions >= cfg.max_repositions:
                    return decision("NEED_HELP", "reposition_budget_exhausted_on_plate")
                return move(Stage.S4, "return_verified")
        elif m.stage == Stage.S4:
            if facts.is_true("reposition_ready"):
                return move(Stage.S5, "reposition_verified")
        elif m.stage == Stage.S7:
            if not facts.is_true("no_remaining"):
                return decision("NEED_HELP", "empty_stem_not_confirmed")
            if facts.is_true("stow_verified"):
                return finish()
        return decision("CONTINUE", "await_stage_result")

    def tick(self, snap, facts, events=()):
        self.ingest(snap, facts, list(events))
        result = self.decide(snap, facts)
        if result.kind == "OBSERVE":
            if self.observing_since is None:
                self.observing_since = snap.now
            if snap.now - self.observing_since >= self.config.observation_timeout_s:
                result = replace(
                    result, kind="NEED_HELP", reason="observation_budget_exhausted"
                )
        else:
            self.observing_since = None
        if self.review_latch:
            result = replace(
                result,
                kind=self.review_latch[0],
                reason=self.review_latch[1],
                target_stage=None,
                skill=None,
                outcome=None,
            )
        elif result.kind in {"STOP", "NEED_HELP"}:
            self.review_latch = (result.kind, result.reason)
        self.issued = {
            result.decision_id: result
        }  # Superseded proposals cannot be committed.
        return result

    def resume_simulated(self, stage, snap, facts, *, reviewer, reason):
        """Explicit reviewed restart in a mock only; preserve all episode budgets/results."""
        if not self.simulation:
            raise RuntimeError("real resume disabled")
        if not self.review_latch or not reviewer.strip() or not reason.strip():
            raise ValueError("reviewer and reason required for a latched review")
        if (
            snap.invalid_reason(self.config)
            or snap.episode_id != self.memory.episode_id
            or self.clock_fault
            or snap.now < self.last_now
            or self.memory.outcome
        ):
            raise ValueError(
                "invalid reviewed observation or episode; clock faults require a new session"
            )
        if (
            facts.missing(ENTRY[stage])
            or facts.conflicts
            or not all(
                facts.is_false(k)
                for k in ("control_fault", "contact_hazard", "damage", "drop")
            )
            or not facts.is_true("execution_consistent")
            or self.memory.damage_detected
            or self.memory.drop_detected
        ):
            raise ValueError("reviewed restart prerequisites not confirmed")
        self.memory.stage, self.memory.stage_since = stage, snap.now
        self.memory.epoch += 1
        self.memory.recovery_skill = None
        self.review_latch = None
        self.observing_since = None
        self.issued.clear()
        return {
            "reviewer": reviewer,
            "reason": reason,
            "stage": stage.value,
            "epoch": self.memory.epoch,
            "hardware_executed": False,
        }

    def accept_simulated(self, proposal, snap, facts, *, handoff_confirmed=False):
        """Only a replay/mock can advance commands. No hardware execution implementation."""
        if not self.simulation:
            raise RuntimeError(
                "real execution disabled; implement verified mono handoff first"
            )
        if self.issued.get(proposal.decision_id) != proposal:
            raise ValueError("unknown or superseded proposal")
        if self.review_latch:
            raise ValueError("operator review required")
        if (
            proposal.expected_epoch != self.memory.epoch
            or proposal.episode_id != self.memory.episode_id
        ):
            raise ValueError("obsolete proposal")
        if not proposal.created_at <= snap.now <= proposal.expires_at:
            raise ValueError("expired or future proposal")
        if not handoff_confirmed:
            raise ValueError("mock handoff confirmation required")
        current = self.decide(snap, facts)
        fields = (
            "kind",
            "target_stage",
            "skill",
            "outcome",
            "exit_after_return",
            "required",
        )
        if any(getattr(current, k) != getattr(proposal, k) for k in fields):
            raise ValueError("conditions changed; replan")
        if proposal.kind not in {"TRANSITION", "RECOVER", "FINISH"}:
            raise ValueError("proposal cannot be executed")
        m = self.memory
        m.epoch += 1
        if proposal.kind == "TRANSITION":
            m.stage, m.stage_since = proposal.target_stage, snap.now
            m.visits.append(m.stage.value)
            if m.stage in (Stage.S1, Stage.S5):
                m.round_id += 1
            if m.stage == Stage.S4:
                m.repositions += 1
            if m.stage in PICK:
                m.round_failure_baseline = dict(m.failures)
            m.exit_after_return = proposal.exit_after_return
        elif proposal.kind == "FINISH":
            m.outcome = proposal.outcome
        elif proposal.kind == "RECOVER":
            m.recovery_skill, m.recovery_since = proposal.skill, snap.now
        # A recovery command does not prove recovery completed or alter physical facts.
        self.issued.clear()


class ChunkFence:
    """Execution-side integration contract, tested offline; NOT a robot controller."""

    def __init__(self):
        self.epoch = 0
        self.pending = False

    def begin_transition(self, expected_epoch):
        if self.pending or expected_epoch != self.epoch:
            raise ValueError("stale or overlapping transition")
        self.pending = True

    def confirm_handoff(self):
        if not self.pending:
            raise ValueError("no pending transition")
        self.epoch += 1
        self.pending = False

    def accepts(self, epoch):
        return not self.pending and epoch == self.epoch

    def permits_prefix(self, previous_epoch):
        return self.accepts(previous_epoch)
