from types import SimpleNamespace
import unittest

from tomato_agent.adapters.lia_mono import (
    PromptCheckedClient,
    require_stage_prompt_contract,
    snapshot_from_sdk,
)
from tomato_agent.models import Config


class AdapterTests(unittest.TestCase):
    def observation(self):
        return SimpleNamespace(
            state={"joint_pos": list(range(54))},
            state_time=100,
            image_times={k: 100 for k in ("cam0", "cam1", "cam2")},
            extra={"clock_id": "ros-clock"},
        )

    def test_measured_state_and_stable_source_identity(self):
        obs = self.observation()
        a, b = snapshot_from_sdk(obs, 100, "e"), snapshot_from_sdk(obs, 100.1, "e")
        self.assertEqual(a.observation_id, b.observation_id)
        self.assertEqual(a.joint_pos, tuple(range(54)))
        self.assertIsNone(a.invalid_reason(Config()))
        self.assertEqual(a.clock_id, "ros-clock")

    def test_missing_camera_is_not_fabricated(self):
        obs = self.observation()
        del obs.image_times["cam2"]
        with self.assertRaises(KeyError):
            snapshot_from_sdk(obs, 100, "e")

    def test_fixed_prompt_server_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "contract absent"):
            require_stage_prompt_contract({"prompt": "full task"})
        require_stage_prompt_contract(
            {"stage_prompt_mode": "per_request", "effective_prompt_echo": True}
        )

    def test_effective_prompt_checked_on_each_response(self):
        client = SimpleNamespace(
            infer=lambda request: {"actions": [], "effective_prompt": "old task"}
        )
        checked = PromptCheckedClient(client)
        with self.assertRaisesRegex(RuntimeError, "differs"):
            checked.infer({"prompt": "return truss"})
        client.infer = lambda request: {
            "actions": [],
            "effective_prompt": request["prompt"],
        }
        self.assertEqual(
            checked.infer({"prompt": "return truss"})["effective_prompt"],
            "return truss",
        )


if __name__ == "__main__":
    unittest.main()
