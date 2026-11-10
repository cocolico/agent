from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from ..models import PROMPTS, Snapshot


class ReadOnlySource:
    """Build the actual mono read-only reader, never call body.reset/step/command/stop."""

    def __init__(self, mono_root, *, episode_id, startup_timeout_s=10.0):
        from lia_robots.robots.franka_wuji import config, franka_wuji
        from lia_robots.robots.camera_profiles import (
            load_camera_profile,
            camera_dimensions,
        )

        root = Path(mono_root).resolve()
        profile = load_camera_profile(
            str(root / "data/deployments/pi05/franka_wuji/cameras.yaml")
        )
        cfg = config(
            str(root / "data/robots/franka_wuji.real.yaml"),
            arm_backend="streaming_impedance",
            control_hz=100.0,
            startup_timeout_s=startup_timeout_s,
        )
        # Preserve each camera's ROS topic when overriding dimensions.
        cfg.cameras = {
            name: replace(camera, **camera_dimensions(profile).get(name, {}))
            for name, camera in cfg.cameras.items()
        }
        # Match the existing Pi05 streaming factory: no optional depth/force subscriptions.
        cfg.sensors = {}
        self.definition = franka_wuji(**asdict(cfg))
        self.reader = self.definition.observation_reader(
            instruction="Observe tomato harvesting"
        )
        self.episode_id = episode_id
        self.raw_observation = None

    def open(self):
        self.reader.open()
        return self

    def read(self):
        observation = self.reader.get_observation()
        self.raw_observation = observation
        return snapshot_from_sdk(
            observation, self.reader.clock.now_ns() / 1e9, self.episode_id
        )

    def close(self):
        self.reader.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()


def snapshot_from_sdk(observation, now, episode_id):
    """Only measured joint_pos and source timestamps; no inferred force/velocity."""
    times = {k: float(observation.image_times[k]) for k in ("cam0", "cam1", "cam2")}
    clock_id = str((observation.extra or {}).get("clock_id", ""))
    identity = [episode_id, clock_id, float(observation.state_time), times]
    observation_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:24]
    joints = tuple(float(v) for v in observation.state["joint_pos"])
    return Snapshot(
        observation_id,
        episode_id,
        float(now),
        float(observation.state_time),
        times,
        joints,
        clock_id,
    )


class Pi05PredictionOnly:
    """Optional inference integration; no action is ever submitted to an embodiment.

    Stage prompt support must be advertised by an upgraded server. Existing full
    service does not advertise this and is deliberately rejected. This verifies
    a software contract, not physical skill success.
    """

    def __init__(self, definition, contract_path, uri, *, timeout_s=2.0):
        from lia_policies.pi05 import Pi05Policy, Pi05WebsocketClient
        from lia_policies.pi05.adapters.franka_wuji import (
            FrankaWujiAdapter,
            wuji_config,
        )

        settings = wuji_config(**json.loads(Path(contract_path).read_text()))
        client = Pi05WebsocketClient(uri, timeout_s=timeout_s)
        try:
            metadata = client.get_server_metadata()
            require_stage_prompt_contract(metadata)
            self.policy = Pi05Policy(
                definition.info,
                client=PromptCheckedClient(client),
                config=settings,
                adapter=FrankaWujiAdapter(),
            )
        except BaseException:
            client.close()
            raise
        self.client = client

    def predict(self, observation, skill, *, stage_epoch):
        if skill not in PROMPTS:
            raise ValueError("unknown skill")
        # Observation in the installed SDK is a dataclass. Never mutate the reader's sample.
        prepared = replace(observation, instruction=PROMPTS[skill])
        chunk = self.policy.act(prepared, rtc_context=None)
        return {
            "stage_epoch": stage_epoch,
            "skill": skill,
            "chunk": chunk,
            "hardware_executed": False,
        }

    def close(self):
        self.client.close()


def require_stage_prompt_contract(metadata):
    if (
        metadata.get("stage_prompt_mode") != "per_request"
        or metadata.get("effective_prompt_echo") is not True
    ):
        raise RuntimeError(
            "stage prompt contract absent: existing Full server may override prompt; "
            "upgrade and audit effective prompts before stage inference"
        )


class PromptCheckedClient:
    """Verify the server's effective prompt on every prediction, not metadata alone."""

    def __init__(self, client, *, epoch=None):
        self.client = client
        self.epoch = epoch

    def get_server_metadata(self):
        return self.client.get_server_metadata()

    def reset(self):
        return self.client.reset()  # Intentionally not a physical stop.

    def infer(self, request):
        wrapped = "rtc" in request
        observation = request["observation"] if wrapped else request
        prompt = observation.get("prompt")
        if self.epoch is not None:
            observation = {**observation, "agent_epoch": self.epoch}
            request = (
                {**request, "observation": observation} if wrapped else observation
            )
        response = self.client.infer(request)
        if not isinstance(response, dict) or response.get("effective_prompt") != prompt:
            raise RuntimeError("server effective prompt differs or receipt is absent")
        if self.epoch is not None and (
            type(response.get("agent_epoch")) is not int
            or response["agent_epoch"] != self.epoch
        ):
            raise RuntimeError("response belongs to another execution epoch")
        return response


def audit_local_sdk(mono_root, server_root):
    """Read local source/config only; no ROS nodes, sockets or subprocesses."""
    mono, server = Path(mono_root).resolve(), Path(server_root).resolve()
    paths = {
        "readonly_reader": mono / "lia-robots/src/lia_robots/robots/observation.py",
        "robot_config": mono / "data/robots/franka_wuji.real.yaml",
        "camera_profile": mono / "data/deployments/pi05/franka_wuji/cameras.yaml",
        "pi05_client": mono / "lia-policies/lia_policies/pi05/client.py",
        "pi05_streaming": mono
        / "lia-workbench/lia_workbench/experiments/pi05_franka_wuji/streaming.py",
        "server_profiles": server / "src/pi05_server/profiles.py",
        "server_transforms": server / "src/pi05_server/runtime.py",
    }
    files = {}
    for key, path in paths.items():
        files[key] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            files[key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "scope": "static_source_inspection_only",
        "hardware_connected": False,
        "files": files,
        "baseline": {"joint_position_dim": 54, "rgb_cameras": ["cam0", "cam1", "cam2"]},
        "not_assumed_available": [
            "external_torque",
            "hand_current",
            "tactile",
            "depth",
        ],
        "review_findings": [
            "Inspected ObservationReader decodes joint positions, not effort/current.",
            "Inspected Pi05 camera profile disables depth; streaming factory clears optional sensors.",
            "Inspected Full profile overrides per-request prompt; upgrade contract required.",
            "Inspected websocket reset is a no-op, not an execution stop or buffer fence.",
        ],
        "note": "Findings describe the initial reviewed revision; hashes enable drift review, not live capability proof.",
    }
