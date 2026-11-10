from dataclasses import replace
from pathlib import Path
import sys
import unittest

from tomato_agent.evidence import EvidenceMonitor, Facts
from tomato_agent.models import Config, Event, Evidence, Snapshot, Stage, Truth
from tomato_agent.supervisor import ChunkFence, Supervisor
from tomato_agent.replay import replay

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_demo import demo_rows


def snap(t=100.0, **kw):
    return replace(
        Snapshot(
            f"obs-{t}",
            "episode",
            t,
            t,
            {k: t for k in ("cam0", "cam1", "cam2")},
            (0.0,) * 54,
        ),
        **kw,
    )


def facts(**overrides):
    data = {
        "control_fault": False,
        "contact_hazard": False,
        "damage": False,
        "drop": False,
        "execution_consistent": True,
        "right_stable": True,
        "at_pick_region": True,
        "left_empty": True,
        "left_clear": True,
        "ledger_resolved": True,
        "plate_available": True,
        "return_start_valid": True,
        "fruit_remaining": True,
        "no_remaining": False,
        "fruit_held_detached": False,
        "reposition_benefit": True,
        "target_ready": True,
        "truss_on_plate": True,
    }
    data.update(overrides)
    values = {
        k: Truth.UNKNOWN if v is None else Truth.TRUE if v else Truth.FALSE
        for k, v in data.items()
    }
    return Facts(values, {k: (f"test://{k}",) for k in values})


def picking(**kwargs):
    agent = Supervisor("episode", 100, simulation=True, **kwargs)
    agent.memory.stage, agent.memory.stage_since = Stage.S6, 100
    return agent


class GraphTests(unittest.TestCase):
    def test_s6_repeated_reposition_preserves_history(self):
        agent = picking()
        agent.memory.repositions = 1
        agent.memory.round_id = 2
        agent.memory.failures["fruit"] = 2
        for t, f, target in (
            (116, facts(), Stage.S3),
            (117, facts(right_released=True), Stage.S4),
            (118, facts(reposition_ready=True), Stage.S5),
            (119, facts(), Stage.S6),
        ):
            s = snap(t)
            d = agent.tick(s, f)
            self.assertEqual(d.target_stage, target)
            agent.accept_simulated(d, s, f, handoff_confirmed=True)
        self.assertEqual(agent.memory.repositions, 2)
        self.assertEqual(agent.memory.round_id, 3)
        self.assertEqual(agent.memory.failures["fruit"], 2)
        self.assertEqual(agent.tick(snap(120), facts()).kind, "CONTINUE")
        self.assertTrue(agent.tick(snap(135), facts()).exit_after_return)

    def test_early_finish_full_episode(self):
        log = list(replay(demo_rows(), simulate=True))
        self.assertEqual(log[-1]["memory"]["visits"], ["S1", "S2", "S7"])
        self.assertEqual(log[-1]["memory"]["outcome"], "COMPLETED")
        self.assertEqual(log[-1]["memory"]["deposited"], ["fruit-1"])
        self.assertTrue(all(not row["hardware_executed"] for row in log))

    def test_recovery_full_episode(self):
        log = list(replay(demo_rows(True), simulate=True))
        self.assertEqual(
            log[-1]["memory"]["visits"], ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
        )
        self.assertEqual(log[-1]["memory"]["outcome"], "COMPLETED")
        recoveries = [
            x["decision"]["skill"]
            for x in log
            if x["mock_accepted"] and x["decision"]["kind"] == "RECOVER"
        ]
        self.assertEqual(recoveries, ["disengage_left"])

    def test_replay_without_simulation_never_advances(self):
        log = list(replay(demo_rows()))
        self.assertTrue(all(row["memory"]["stage"] is None for row in log))

    def test_unknown_remaining_does_not_finish(self):
        d = picking().tick(snap(), facts(fruit_remaining=None, no_remaining=None))
        self.assertEqual(d.kind, "OBSERVE")

    def test_attached_hand_requires_disengage(self):
        d = picking().tick(snap(116), facts(left_clear=False, contact_releasable=True))
        self.assertEqual((d.kind, d.skill), ("RECOVER", "disengage_left"))

    def test_unknown_hand_clear_cannot_return(self):
        d = picking().tick(snap(116), facts(left_clear=None))
        self.assertEqual(d.kind, "OBSERVE")

    def test_unknown_return_start_cannot_return(self):
        d = picking().tick(snap(116), facts(return_start_valid=None))
        self.assertEqual(d.kind, "OBSERVE")
        self.assertIn("return_start_valid", d.missing)

    def test_held_fruit_takes_precedence(self):
        d = picking().tick(
            snap(116),
            facts(
                fruit_held_detached=True,
                left_empty=False,
                deposit_path_ready=True,
                reposition_needed=True,
            ),
        )
        self.assertEqual(d.skill, "deposit_held_fruit")

    def test_unstable_support_stops(self):
        self.assertEqual(picking().tick(snap(), facts(right_stable=False)).kind, "STOP")

    def test_unresolved_separation_blocks_return(self):
        agent = picking()
        agent.memory.separated.add("fruit")
        self.assertEqual(agent.tick(snap(116), facts()).kind, "OBSERVE")

    def test_budget_exit_uses_s3_not_s7(self):
        agent = picking()
        agent.memory.repositions = agent.config.max_repositions
        s, f = snap(116), facts()
        d = agent.tick(s, f)
        self.assertEqual(d.target_stage, Stage.S3)
        self.assertTrue(d.exit_after_return)
        agent.accept_simulated(d, s, f, handoff_confirmed=True)
        d = agent.tick(snap(117), facts(right_released=True, stow_verified=True))
        self.assertEqual(d.outcome, "FAILED")

    def test_stage_timeout(self):
        agent = picking()
        agent.memory.stage = Stage.S1
        self.assertEqual(agent.tick(snap(161), facts()).reason, "stage_timeout")

    def test_observation_budget(self):
        agent = picking()
        agent.tick(snap(), facts(right_stable=None))
        d = agent.tick(snap(106), facts(right_stable=None))
        self.assertEqual(d.kind, "NEED_HELP")

    def test_unvalidated_skill(self):
        agent = Supervisor("episode", 100)
        d = agent.tick(snap(), facts())
        self.assertEqual(d.kind, "NEED_HELP")
        self.assertIn("skill_not_validated", d.reason)

    def test_recovery_does_not_repeat_before_feedback(self):
        agent = picking()
        s, f = snap(116), facts(left_clear=False, contact_releasable=True)
        d = agent.tick(s, f)
        agent.accept_simulated(d, s, f, handoff_confirmed=True)
        d = agent.tick(snap(117), f)
        self.assertEqual(d.kind, "CONTINUE")
        self.assertEqual(agent.memory.recovery_skill, "disengage_left")
        self.assertEqual(agent.tick(snap(177), f).reason, "recovery_timeout")


class TransactionTests(unittest.TestCase):
    def test_previous_observation_facts_cannot_be_reused(self):
        agent = picking()
        with self.assertRaisesRegex(ValueError, "different observation"):
            agent.tick(snap(), replace(facts(), observation_id="older-frame"))

    def test_stop_cannot_auto_resume(self):
        agent = picking()
        agent.tick(snap(), facts(control_fault=True))
        self.assertEqual(agent.tick(snap(101), facts()).kind, "STOP")
        receipt = agent.resume_simulated(
            Stage.S6, snap(102), facts(), reviewer="test", reason="mock inspection"
        )
        self.assertFalse(receipt["hardware_executed"])
        self.assertEqual(agent.tick(snap(102), facts()).kind, "CONTINUE")

    def test_resume_cannot_clear_episode_budget(self):
        agent = picking()
        agent.memory.repositions = 2
        agent.memory.failures["fruit"] = 3
        agent.tick(snap(), facts(control_fault=True))
        agent.resume_simulated(
            Stage.S6, snap(101), facts(), reviewer="test", reason="checked"
        )
        self.assertEqual(agent.memory.repositions, 2)
        self.assertEqual(agent.memory.failures["fruit"], 3)

    def test_no_real_execution(self):
        agent = Supervisor("episode", 100, available_skills=("lift_truss",))
        s, f = snap(), facts()
        d = agent.tick(s, f)
        with self.assertRaisesRegex(RuntimeError, "real execution disabled"):
            agent.accept_simulated(d, s, f, handoff_confirmed=True)

    def test_requires_handoff_and_current_conditions(self):
        agent = picking()
        s, f = snap(116), facts()
        d = agent.tick(s, f)
        with self.assertRaises(ValueError):
            agent.accept_simulated(d, s, f)
        with self.assertRaises(ValueError):
            agent.accept_simulated(
                d, snap(116.1), facts(left_clear=False), handoff_confirmed=True
            )
        self.assertEqual(agent.memory.stage, Stage.S6)

    def test_expired_and_superseded_proposals(self):
        agent = picking()
        s, f = snap(116), facts()
        d = agent.tick(s, f)
        with self.assertRaisesRegex(ValueError, "expired"):
            agent.accept_simulated(d, snap(117), f, handoff_confirmed=True)
        agent.tick(snap(116.1), f)
        with self.assertRaisesRegex(ValueError, "superseded"):
            agent.accept_simulated(d, snap(116.1), f, handoff_confirmed=True)

    def test_duplicate_commit(self):
        agent = picking()
        s, f = snap(116), facts()
        d = agent.tick(s, f)
        agent.accept_simulated(d, s, f, handoff_confirmed=True)
        with self.assertRaises(ValueError):
            agent.accept_simulated(d, s, f, handoff_confirmed=True)

    def test_old_result_and_prefix_rejected(self):
        fence = ChunkFence()
        fence.begin_transition(0)
        self.assertFalse(fence.accepts(0))
        fence.confirm_handoff()
        self.assertFalse(fence.accepts(0))
        self.assertFalse(fence.permits_prefix(0))
        self.assertTrue(fence.accepts(1))
        with self.assertRaises(ValueError):
            fence.begin_transition(0)


class EventTests(unittest.TestCase):
    def test_dedup_and_budget_preserved(self):
        agent = picking()
        s, f = snap(), facts()
        event = Event(
            "fail",
            "ATTEMPT_FAILED",
            s.observation_id,
            "fruit",
            "attempt",
            ("test://1",),
        )
        agent.tick(s, f, [event, event])
        self.assertEqual(agent.memory.failures, {"fruit": 1})
        agent.tick(s, f, [replace(event, event_id="fail-report-2")])
        self.assertEqual(agent.memory.failures, {"fruit": 1})

    def test_duplicate_deposit_does_not_reset_progress(self):
        agent = picking()
        s, f = snap(), facts(fruit_separated=True, deposit_verified=True)
        events = [
            Event("sep", "SEPARATED", s.observation_id, "f", references=("test://1",)),
            Event("dep", "DEPOSITED", s.observation_id, "f", references=("test://2",)),
        ]
        agent.tick(s, f, events)
        s2 = snap(101)
        agent.tick(
            s2,
            f,
            [
                replace(
                    events[1], event_id="duplicate", observation_id=s2.observation_id
                )
            ],
        )
        self.assertEqual(agent.memory.last_pick_progress, 100)
        self.assertEqual(agent.memory.deposited, {"f"})

    def test_deposit_requires_separation(self):
        s = snap()
        with self.assertRaises(ValueError):
            picking().tick(
                s,
                facts(deposit_verified=True),
                [Event("e", "DEPOSITED", s.observation_id, "f", references=("r",))],
            )

    def test_conflicting_event_identity(self):
        agent = picking()
        s = snap()
        event = Event("e", "ATTEMPT_FAILED", s.observation_id, "f", "a", ("r",))
        agent.tick(s, facts(), [event])
        with self.assertRaises(ValueError):
            agent.tick(s, facts(), [replace(event, target_id="another")])

    def test_target_lifetime_budget(self):
        agent = picking()
        agent.memory.failures["f"] = 4
        d = agent.tick(snap(target_id="f"), facts())
        self.assertEqual(d.reason, "target_attempt_budget_exhausted")


class EvidenceTests(unittest.TestCase):
    def ev(self, s, value=Truth.TRUE, source="vision", predicate="right_stable"):
        return Evidence(
            predicate, value, s.observation_id, s.now, source, ("test://frame",)
        )

    def test_requires_distinct_frames_and_stability(self):
        monitor = EvidenceMonitor(Config())
        s = snap()
        self.assertFalse(monitor.update(s, [self.ev(s)]).is_true("right_stable"))
        repeated = replace(s, now=100.2)
        self.assertFalse(
            monitor.update(repeated, [self.ev(repeated)]).is_true("right_stable")
        )
        s2 = snap(100.25)
        self.assertTrue(monitor.update(s2, [self.ev(s2)]).is_true("right_stable"))

    def test_conflicting_sources(self):
        monitor = EvidenceMonitor(Config())
        for t in (100, 100.2):
            s = snap(t)
            result = monitor.update(s, [self.ev(s), self.ev(s, Truth.FALSE, "other")])
        self.assertIn("right_stable", result.conflicts)
        self.assertFalse(result.is_true("right_stable"))

    def test_missing_evidence_returns_unknown(self):
        monitor = EvidenceMonitor(Config())
        for t in (100, 100.2):
            s = snap(t)
            monitor.update(s, [self.ev(s)])
        self.assertFalse(monitor.update(snap(100.3), []).is_true("right_stable"))

    def test_stale_or_wrong_observation_evidence(self):
        monitor = EvidenceMonitor(Config())
        s = snap()
        result = monitor.update(s, [replace(self.ev(s), observation_id="old")])
        self.assertFalse(result.is_true("right_stable"))
        result = monitor.update(s, [replace(self.ev(s), timestamp=99)])
        self.assertFalse(result.is_true("right_stable"))

    def test_stale_nan_and_skew_stop(self):
        for s in (
            snap(now=101),
            snap(joint_pos=(float("nan"),) * 54),
            snap(image_times={"cam0": 100, "cam1": 99.8, "cam2": 100}),
        ):
            with self.subTest(s=s):
                self.assertEqual(picking().tick(s, facts()).kind, "STOP")

    def test_clock_jump_latches(self):
        agent = picking()
        agent.tick(snap(102), facts())
        self.assertEqual(agent.tick(snap(101), facts()).kind, "STOP")
        self.assertEqual(agent.tick(snap(103), facts()).kind, "STOP")

    def test_fault_not_debounced(self):
        monitor = EvidenceMonitor(Config())
        s = snap()
        result = monitor.update(s, [self.ev(s, predicate="control_fault")])
        self.assertTrue(result.is_true("control_fault"))

    def test_config_validation(self):
        for kwargs in (
            {"max_repositions": True},
            {"max_age_s": float("nan")},
            {"stable_s": 0},
        ):
            with self.assertRaises(ValueError):
                Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
