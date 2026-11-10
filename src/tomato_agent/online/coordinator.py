"""Online decisions cause real, acknowledged skill changes through an execution owner."""

from dataclasses import replace

from ..evidence import EvidenceMonitor
from ..models import Config, Evidence, Truth, jsonable
from ..supervisor import Supervisor


class OnlineCoordinator:
    def __init__(self, episode_id, now, config, executor, emit):
        self.config, self.executor, self.emit = config, executor, emit
        # VLM evidence has its own bounded age; the SDK controller retains its fast observation checks.
        rules = Config(
            max_age_s=config.max_result_age_s,
            decision_ttl_s=config.max_result_age_s,
            observation_timeout_s=config.supervision_deadline_s,
            episode_timeout_s=config.max_runtime_s,
        )
        self.supervisor = Supervisor(
            episode_id, now, rules, available_skills=config.validated_skills
        )
        self.monitor = EvidenceMonitor(rules)
        self.review_started_at = now
        self.review_frames = set()
        self.review_signature = None
        self.terminal = False

    def hold(self, source_now, reason):
        status = self.executor.status()
        was_active = status["active"]
        if was_active or status.get("prepared"):
            self.executor.pause()
            self.review_started_at = source_now
            self.review_frames.clear()
            self.review_signature = None
            # Never stabilize a new transition using frames from before the hold.
            self.monitor = EvidenceMonitor(self.supervisor.config)
        self.emit({"event": "hold", "reason": reason, "was_active": was_active})

    def process(self, judgement, current, refresh=None):
        m = self.supervisor.memory
        if self.terminal:
            raise RuntimeError("online episode is closed")
        if judgement.stage_epoch != m.epoch:
            self.emit({"event": "vlm_discard", "reason": "obsolete_epoch"})
            return None
        if (
            current.episode_id != m.episode_id
            or current.clock_id != judgement.snapshot.clock_id
            or current.now < judgement.snapshot.now
            or current.now - min(judgement.snapshot.image_times.values())
            > self.config.max_result_age_s
        ):
            self.hold(current.now, "stale_or_incompatible_vlm_result")
            return None
        invalid = current.invalid_reason(Config())
        if invalid:
            self.hold(current.now, invalid)
            raise RuntimeError(invalid)
        status = self.executor.status()
        if status["fault"]:
            raise RuntimeError(status["fault"])
        if judgement.intent == "STOP":
            self.hold(current.now, judgement.reason)
            self.terminal = True
            return {"kind": "STOP", "reason": judgement.reason}

        snap = replace(judgement.snapshot, now=current.now)
        system_evidence = Evidence(
            "control_fault",
            Truth.FALSE,
            snap.observation_id,
            current.now,
            "sdk_execution_owner",
            (f"sdk://epoch/{status['epoch']}",),
        )
        facts = self.monitor.update(snap, [*judgement.evidence, system_evidence])
        decision = self.supervisor.tick(snap, facts, judgement.events)
        self.emit(
            {
                "event": "supervision",
                "vlm": {
                    "request_id": judgement.request_id,
                    "intent": judgement.intent,
                    "next_skill": judgement.next_skill,
                    "reason": judgement.reason,
                    "evidence": jsonable(judgement.evidence),
                    "events": jsonable(judgement.events),
                },
                "facts": jsonable(facts),
                "decision": jsonable(decision),
                "memory": jsonable(m),
            }
        )
        if decision.kind in {"STOP", "NEED_HELP"}:
            self.hold(current.now, decision.reason)
            self.terminal = True
            return decision
        if judgement.intent == "OBSERVE" or decision.kind == "OBSERVE":
            self.hold(current.now, "additional_observation_needed")
            return decision
        if (
            judgement.next_skill
            and decision.skill
            and judgement.next_skill != decision.skill
        ):
            self.hold(current.now, "vlm_skill_rejected_by_task_conditions")
            return decision
        if decision.kind == "CONTINUE" and status["active"]:
            return decision
        if decision.kind == "CONTINUE":
            decision = self.supervisor.propose_resume(snap, facts)
            if decision is None:
                self.hold(current.now, "held_state_not_in_skill_entry_range")
                return None
        if decision.kind not in {"TRANSITION", "RECOVER", "FINISH", "RESUME"}:
            return decision

        if status["active"]:
            self.hold(current.now, "stage_handoff_reobserve")
            return decision
        # Require two distinct post-hold source observations, not two readings of one image.
        sample_times = tuple(snap.image_times[k] for k in sorted(snap.image_times))
        if min(sample_times) <= self.review_started_at:
            return decision
        signature = (
            decision.kind,
            decision.target_stage,
            decision.skill,
            decision.exit_after_return,
        )
        if signature != self.review_signature:
            self.review_frames.clear()
            self.review_signature = signature
        self.review_frames.add(sample_times)
        if len(self.review_frames) < 2:
            return decision
        if (
            max(
                abs(a - b)
                for a, b in zip(current.joint_pos, snap.joint_pos, strict=True)
            )
            > self.config.max_hold_drift_rad
        ):
            self.hold(current.now, "state_changed_since_vlm_observation")
            self.review_started_at = current.now
            self.review_frames.clear()
            return decision

        receipt = self.executor.prepare(decision)
        try:
            after = refresh() if refresh is not None else current
            if (
                after.clock_id != snap.clock_id
                or after.episode_id != snap.episode_id
                or after.now < current.now
                or after.invalid_reason(Config())
                or after.now - min(snap.image_times.values())
                > self.config.max_result_age_s
                or max(
                    abs(a - b)
                    for a, b in zip(after.joint_pos, snap.joint_pos, strict=True)
                )
                > self.config.max_hold_drift_rad
            ):
                raise ValueError(
                    "handoff state/evidence changed while preparing the skill"
                )
            # Preparation cannot move the body; commit precedes the first command.
            self.supervisor.accept_prepared(
                decision, replace(snap, now=after.now), facts, receipt
            )
            self.executor.activate(receipt)
        except BaseException:
            self.executor.pause()
            raise
        self.review_frames.clear()
        self.emit(
            {
                "event": "skill_activated",
                "receipt": jsonable(receipt),
                "memory": jsonable(m),
            }
        )
        if decision.kind == "FINISH":
            self.terminal = True
        return decision
