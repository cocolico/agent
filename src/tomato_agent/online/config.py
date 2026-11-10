from dataclasses import dataclass, field
import json
from pathlib import Path
from urllib.parse import urlparse

from ..models import PROMPTS, finite

ROOT = Path(__file__).resolve().parents[3]


def local_output(path):
    value = Path(path).expanduser().resolve()
    if not value.is_relative_to(ROOT):
        raise ValueError(f"generated files must stay inside {ROOT}")
    return value


@dataclass(frozen=True)
class OnlineConfig:
    vlm_url: str = "http://127.0.0.1:8009/v1/chat/completions"
    vlm_model: str = "Qwen3-VL-2B-Instruct"
    vlm_timeout_s: float = 3.0
    max_result_age_s: float = 4.0
    supervision_deadline_s: float = 6.0
    poll_hz: float = 20.0
    vlm_interval_s: float = 0.5
    history_frames: int = 3
    image_max_side: int = 512
    max_tokens: int = 1600
    mono_root: str = "/home/user/lpy/li-agent-mono"
    pi05_uri: str = "ws://127.0.0.1:8010"
    model_contract: str = str(ROOT / "runtime/pi05-stage-model.json")
    log_path: str = str(ROOT / "outputs/online.jsonl")
    validated_skills: tuple[str, ...] = ()
    max_runtime_s: float = 300.0
    handoff_timeout_s: float = 10.0
    max_hold_drift_rad: float = 0.03
    execution: dict = field(
        default_factory=lambda: {
            "mode": "rtc",
            "control_hz": 100.0,
            "rounds": 300,
            "sg_window": 0,
            "ruckig_enabled": True,
            "ruckig_max_velocity": 1.0,
            "ruckig_max_acceleration": 3.0,
            "ruckig_max_jerk": 30.0,
        }
    )

    def __post_init__(self):
        for name in ("vlm_url", "pi05_uri"):
            url = urlparse(getattr(self, name))
            schemes = ("http", "https") if name == "vlm_url" else ("ws", "wss")
            if url.scheme not in schemes or url.hostname not in {
                "127.0.0.1",
                "localhost",
                "::1",
            }:
                raise ValueError(f"{name} must address the local model service")
            if url.username or url.password:
                raise ValueError("credentials in endpoint URLs are not supported")
        for name in (
            "vlm_timeout_s",
            "max_result_age_s",
            "supervision_deadline_s",
            "poll_hz",
            "vlm_interval_s",
            "max_runtime_s",
            "handoff_timeout_s",
            "max_hold_drift_rad",
        ):
            if finite(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name, low, high in (
            ("history_frames", 2, 8),
            ("image_max_side", 128, 1024),
            ("max_tokens", 128, 4096),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} outside supported bounds")
        if not 1 <= self.poll_hz <= 100:
            raise ValueError("poll_hz must be in [1, 100]")
        if self.supervision_deadline_s <= self.vlm_timeout_s:
            raise ValueError("supervision deadline must exceed VLM request timeout")
        if not isinstance(self.vlm_model, str) or not self.vlm_model.strip():
            raise ValueError("vlm_model required")
        if not set(self.validated_skills) <= set(PROMPTS):
            raise ValueError("unknown skill in validated_skills")
        object.__setattr__(self, "validated_skills", tuple(self.validated_skills))
        local_output(self.log_path)
        local_output(self.model_contract)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def preflight(self):
        return {
            "vlm_endpoint": self.vlm_url,
            "vlm_model": self.vlm_model,
            "pi05_endpoint": self.pi05_uri,
            "model_contract_exists": Path(self.model_contract).is_file(),
            "validated_skills": list(self.validated_skills),
            "ready_for_startup_checks": bool(self.validated_skills)
            and Path(self.model_contract).is_file(),
            "models_downloaded": False,
            "devices_connected": False,
            "note": "Static config check only; runtime verifies service contracts and SDK readiness.",
        }
