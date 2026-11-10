from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

from tomato_agent.adapters.lia_mono import PromptCheckedClient
from tomato_agent.models import Evidence, Event, Snapshot, Stage, Truth
from tomato_agent.online.config import OnlineConfig, ROOT, local_output
from tomato_agent.online.coordinator import OnlineCoordinator
from tomato_agent.online.execution import ExecutionOwner, PreparedReceipt
from tomato_agent.online.vlm import (
    LocalVisionClient,
    VisionFrame,
    VisionRequest,
    VisionJudgement,
    SingleFlightVision,
    parse_judgement,
    request_payload,
)
from tomato_agent.stage_server import StagePolicy, build_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_demo import BASE


def sample(t=100):
    return Snapshot(
        f"obs-{t}",
        "online-test",
        t,
        t,
        {k: t for k in ("cam0", "cam1", "cam2")},
        (0.0,) * 54,
    )


def request():
    snap = sample()
    frames = tuple(
        VisionFrame(
            f"frame-{k}", snap.observation_id, k, 100, "data:image/jpeg;base64,AA=="
        )
        for k in ("cam0", "cam1", "cam2")
    )
    return VisionRequest("request", 3, snap, frames, {}, {})


def reply():
    return {
        "request_id": "request",
        "observation_id": "obs-100",
        "stage_epoch": 3,
        "observed_activity": "S2",
        "target_id": None,
        "facts": {"left_clear": {"value": "TRUE", "frame_ids": ["frame-cam0"]}},
        "events": [],
        "intent": "SWITCH",
        "next_skill": "return_truss",
        "reason": "no progress",
    }


class VLMTests(unittest.TestCase):
    def test_only_one_vlm_request_in_flight_and_shutdown_does_not_wait(self):
        gate = threading.Event()
        worker = SingleFlightVision()
        result = worker.submit(lambda _: gate.wait(1), None)
        with self.assertRaises(RuntimeError):
            worker.submit(lambda _: None, None)
        worker.close()
        gate.set()
        self.assertTrue(result.result(timeout=1))
        with self.assertRaises(RuntimeError):
            worker.submit(lambda _: None, None)

    def test_multiview_request_and_receipt(self):
        payload = request_payload(request(), OnlineConfig())
        content = payload["messages"][1]["content"]
        self.assertEqual(sum(c["type"] == "image_url" for c in content), 3)
        result = parse_judgement(json.dumps(reply()), request())
        self.assertEqual(result.next_skill, "return_truss")
        self.assertEqual(result.evidence[0].timestamp, 100)

    def test_stale_wrong_identity_and_prompt_injection_rejected(self):
        variants = [
            reply() | {"stage_epoch": 2},
            reply() | {"request_id": "old"},
            reply() | {"next_skill": "execute_python"},
            reply() | {"joint_targets": [1]},
            reply() | {"reason": ""},
        ]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_judgement(value, request())

    def test_absent_frame_or_control_fault_claim_rejected(self):
        for facts in (
            {"left_clear": {"value": "TRUE", "frame_ids": ["imaginary"]}},
            {"control_fault": {"value": "FALSE", "frame_ids": ["frame-cam0"]}},
        ):
            with self.assertRaises(ValueError):
                parse_judgement(reply() | {"facts": facts}, request())

    def test_http_client_uses_local_schema_without_model(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def read(self, _):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": json.dumps(reply())},
                            }
                        ]
                    }
                ).encode()

        client = LocalVisionClient(OnlineConfig())
        sent = []
        client.opener = SimpleNamespace(
            open=lambda req, timeout: (sent.append((req, timeout)) or Response())
        )
        self.assertEqual(client.judge(request()).stage_epoch, 3)
        self.assertEqual(json.loads(sent[0][0].data)["model"], "Qwen3-VL-2B-Instruct")

    def test_config_only_local_outputs_and_endpoints(self):
        self.assertEqual(
            local_output(ROOT / "outputs/test.jsonl"), ROOT / "outputs/test.jsonl"
        )
        with self.assertRaises(ValueError):
            local_output("/tmp/wrong.json")
        with self.assertRaises(ValueError):
            OnlineConfig(vlm_url="https://remote.example/v1/chat/completions")


class StageServerTests(unittest.TestCase):
    def test_builder_disables_override_without_changing_profile(self):
        calls = []

        def factory(metadata, stats, prompt, *, override_prompt):
            calls.append(override_prompt)
            return SimpleNamespace(
                infer=lambda obs, **kw: {"actions": [], "received": obs["prompt"]}
            )

        policy = build_policy({}, None, ["lift_truss", "return_truss"], factory)
        from tomato_agent.models import PROMPTS

        for epoch, skill in enumerate(("lift_truss", "return_truss")):
            out = policy.infer({"prompt": PROMPTS[skill], "agent_epoch": epoch})
            self.assertEqual(out["effective_prompt"], PROMPTS[skill])
            self.assertEqual(out["received"], PROMPTS[skill])
            self.assertEqual(out["agent_epoch"], epoch)
        self.assertEqual(calls, [False])

    def test_disallowed_prompt_and_bad_epoch(self):
        policy = StagePolicy(None, ["lift"])
        for obs in ({"prompt": "arbitrary"}, {"prompt": "lift", "agent_epoch": True}):
            with self.assertRaises(ValueError):
                policy.infer(obs)

    def test_rtc_envelope_kept_intact_and_epoch_inside_observation(self):
        original = {"observation": {"prompt": "lift"}, "rtc": {"token": "rtc-1"}}
        received = []

        def infer(req):
            received.append(req)
            return {
                "effective_prompt": req["observation"]["prompt"],
                "agent_epoch": req["observation"]["agent_epoch"],
                "actions": [],
            }

        client = PromptCheckedClient(SimpleNamespace(infer=infer), epoch=7)
        client.infer(original)
        self.assertEqual(set(received[0]), {"observation", "rtc"})
        self.assertEqual(received[0]["observation"]["agent_epoch"], 7)
        self.assertNotIn("agent_epoch", original["observation"])

    def test_old_epoch_response_rejected(self):
        client = PromptCheckedClient(
            SimpleNamespace(
                infer=lambda _: {"effective_prompt": "lift", "agent_epoch": 1}
            ),
            epoch=2,
        )
        with self.assertRaisesRegex(RuntimeError, "another execution epoch"):
            client.infer({"prompt": "lift"})


class FakeBackend:
    def __init__(self):
        self.log = []
        self.sessions = []
        self.fail_stop = False

    def open(self):
        self.log.append("open")

    def stop(self):
        self.log.append("stop")
        if self.fail_stop:
            raise RuntimeError("unconfirmed hold")

    def prepare(self, skill, epoch):
        backend = self

        class Session:
            closed = False
            ticks = 0

            def tick(self):
                if self.closed:
                    raise RuntimeError("retired session used")
                self.ticks += 1

            def close(self):
                self.closed = True
                backend.log.append(f"close-{epoch}")

        self.log.append(f"prepare-{epoch}")
        session = Session()
        self.sessions.append(session)
        return session

    def close(self):
        self.log.append("close-body")


class ExecutionTests(unittest.TestCase):
    def test_prepare_never_moves_activation_and_retirement_are_fenced(self):
        backend = FakeBackend()
        owner = ExecutionOwner(backend, control_hz=200).start()
        try:
            receipt = owner.prepare(
                SimpleNamespace(decision_id="1", expected_epoch=0, skill="lift_truss")
            )
            self.assertEqual(backend.sessions[0].ticks, 0)
            owner.activate(receipt)
            owner.pause()
            old_ticks = backend.sessions[0].ticks
            receipt = owner.prepare(
                SimpleNamespace(decision_id="2", expected_epoch=1, skill="return_truss")
            )
            owner.activate(receipt)
            owner.pause()
            self.assertEqual(backend.sessions[0].ticks, old_ticks)
            self.assertLess(
                backend.log.index("close-1"), backend.log.index("prepare-2")
            )
            self.assertTrue(owner.status()["holding"])
        finally:
            owner.close()

    def test_failed_hold_never_prepares_next_skill(self):
        backend = FakeBackend()
        owner = ExecutionOwner(backend).start()
        backend.fail_stop = True
        with self.assertRaisesRegex(RuntimeError, "unconfirmed hold"):
            owner.pause()
        with self.assertRaises(RuntimeError):
            owner.close()
        self.assertFalse(backend.sessions)

    def test_stop_failure_still_retires_inflight_controller(self):
        backend = FakeBackend()
        owner = ExecutionOwner(backend).start()
        receipt = owner.prepare(
            SimpleNamespace(decision_id="1", expected_epoch=0, skill="lift_truss")
        )
        owner.activate(receipt)
        backend.fail_stop = True
        with self.assertRaises(RuntimeError):
            owner.pause()
        with self.assertRaises(RuntimeError):
            owner.close()
        self.assertTrue(backend.sessions[0].closed)
        self.assertFalse(owner.status()["active"])
        self.assertFalse(owner.status()["holding"])

    def test_failed_activation_receipt_closes_owner(self):
        owner = ExecutionOwner(FakeBackend()).start()
        with self.assertRaises(ValueError):
            owner.activate(PreparedReceipt("unissued", 1, "lift_truss", True))
        with self.assertRaises(RuntimeError):
            owner.close()


class FakeExecutor:
    def __init__(self):
        self.data = {
            "holding": True,
            "active": False,
            "prepared": False,
            "epoch": 0,
            "fault": None,
        }
        self.activated = []
        self.on_prepare = None

    def status(self):
        return dict(self.data)

    def pause(self):
        self.data.update(holding=True, active=False, prepared=False)

    def prepare(self, decision):
        self.data["prepared"] = True
        if self.on_prepare:
            self.on_prepare()
        return PreparedReceipt(
            decision.decision_id, decision.expected_epoch + 1, decision.skill, True
        )

    def activate(self, receipt):
        self.activated.append(receipt)
        self.data.update(
            active=receipt.skill is not None,
            holding=receipt.skill is None,
            epoch=receipt.stage_epoch,
            prepared=False,
        )


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        from tomato_agent.models import PROMPTS

        self.executor = FakeExecutor()
        self.log = []
        self.agent = OnlineCoordinator(
            "online-test",
            100,
            OnlineConfig(validated_skills=tuple(PROMPTS)),
            self.executor,
            self.log.append,
        )
        self.t = 100

    def feed(
        self,
        overrides=None,
        events=(),
        intent="CONTINUE",
        next_skill=None,
        refresh=None,
    ):
        self.t += 0.2
        snap = sample(round(self.t, 5))
        mapping = {**BASE, **(overrides or {})}
        evidence = tuple(
            Evidence(
                k,
                Truth.TRUE if v else Truth.FALSE,
                snap.observation_id,
                snap.now,
                "test",
                ("frame://test",),
            )
            for k, v in mapping.items()
            if k != "control_fault"
        )
        updated = tuple(replace(e, observation_id=snap.observation_id) for e in events)
        judgement = VisionJudgement(
            snap,
            evidence,
            updated,
            intent,
            next_skill,
            "test evidence",
            str(self.t),
            self.agent.supervisor.memory.epoch,
        )
        return self.agent.process(judgement, snap, refresh=refresh)

    def drive(self, stage, values):
        for _ in range(12):
            self.feed(values)
            if (
                self.agent.supervisor.memory.stage == stage
                and self.executor.data["active"]
            ):
                return
        self.fail(f"did not reach {stage}: {self.log[-1]}")

    def test_full_online_normal_and_early_finish_through_executor(self):
        self.drive(Stage.S1, {"truss_on_plate": True})
        self.drive(Stage.S2, {"right_stable": True, "at_pick_region": True})
        events = [
            Event("sep", "SEPARATED", "", "fruit", references=("test",)),
            Event("dep", "DEPOSITED", "", "fruit", references=("test",)),
        ]
        for _ in range(3):
            self.feed(
                {
                    "right_stable": True,
                    "fruit_separated": True,
                    "deposit_verified": True,
                },
                events,
            )
        self.drive(
            Stage.S7,
            {"right_stable": True, "fruit_remaining": False, "no_remaining": True},
        )
        for _ in range(10):
            self.feed(
                {
                    "fruit_remaining": False,
                    "no_remaining": True,
                    "truss_on_plate": True,
                    "right_released": True,
                    "stow_verified": True,
                }
            )
            if self.agent.terminal:
                break
        self.assertEqual(self.agent.supervisor.memory.outcome, "COMPLETED")
        self.assertFalse(self.executor.data["active"])
        self.assertEqual(
            [r.skill for r in self.executor.activated],
            ["lift_truss", "pick_and_deposit", "return_bare_stem", None],
        )

    def test_online_failed_pick_disengages_returns_and_repositions(self):
        self.drive(Stage.S1, {"truss_on_plate": True})
        self.drive(Stage.S2, {"right_stable": True, "at_pick_region": True})
        for _ in range(12):
            self.feed(
                {
                    "right_stable": True,
                    "left_clear": False,
                    "contact_releasable": True,
                    "reposition_needed": True,
                    "reposition_benefit": True,
                }
            )
            if (
                self.executor.activated[-1].skill == "disengage_left"
                and self.executor.data["active"]
            ):
                break
        self.assertEqual(self.executor.activated[-1].skill, "disengage_left")
        self.drive(
            Stage.S3,
            {
                "right_stable": True,
                "left_clear": True,
                "reposition_needed": True,
                "reposition_benefit": True,
            },
        )
        self.drive(
            Stage.S4,
            {
                "truss_on_plate": True,
                "right_released": True,
                "reposition_benefit": True,
            },
        )
        self.drive(Stage.S5, {"truss_on_plate": True, "reposition_ready": True})
        self.drive(Stage.S6, {"right_stable": True, "at_pick_region": True})
        self.assertEqual(self.agent.supervisor.memory.repositions, 1)
        self.assertEqual(
            [r.skill for r in self.executor.activated],
            [
                "lift_truss",
                "pick_and_deposit",
                "disengage_left",
                "return_truss",
                "reposition_truss",
                "lift_truss",
                "pick_and_deposit",
            ],
        )

    def test_stale_result_is_not_relabelled_as_fresh(self):
        self.drive(Stage.S1, {"truss_on_plate": True})
        s = sample(100)
        old = VisionJudgement(
            s,
            (),
            (),
            "SWITCH",
            "pick_and_deposit",
            "old",
            "old",
            self.agent.supervisor.memory.epoch,
        )
        self.assertIsNone(self.agent.process(old, sample(110)))
        self.assertFalse(self.executor.data["active"])

    def test_model_stop_stops_active_execution(self):
        self.drive(Stage.S1, {"truss_on_plate": True})
        result = self.feed(intent="STOP")
        self.assertEqual(result["kind"], "STOP")
        self.assertTrue(self.agent.terminal)
        self.assertFalse(self.executor.data["active"])

    def test_jump_to_bare_stem_rejected(self):
        self.drive(Stage.S1, {"truss_on_plate": True})
        before = len(self.executor.activated)
        for _ in range(5):
            self.feed(
                {"right_stable": True, "at_pick_region": True},
                intent="SWITCH",
                next_skill="return_bare_stem",
            )
        self.assertEqual(len(self.executor.activated), before)
        self.assertEqual(self.agent.supervisor.memory.stage, Stage.S1)

    def test_state_drift_after_prepare_prevents_activation(self):
        def changed():
            return replace(sample(round(self.t + 0.1, 5)), joint_pos=(1.0,) * 54)

        with self.assertRaisesRegex(ValueError, "handoff state"):
            for _ in range(6):
                self.feed({"truss_on_plate": True}, refresh=changed)
        self.assertFalse(self.executor.activated)
        self.assertIsNone(self.agent.supervisor.memory.stage)
        self.assertFalse(self.executor.data["prepared"])

    def test_old_vlm_result_cannot_trigger_action(self):
        s = sample(100)
        result = VisionJudgement(s, (), (), "SWITCH", "lift_truss", "old", "old", 99)
        self.assertIsNone(self.agent.process(result, sample(101)))
        self.assertFalse(self.executor.activated)


if __name__ == "__main__":
    unittest.main()
