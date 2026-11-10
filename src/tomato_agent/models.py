from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any


class Truth(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class Stage(str, Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"
    S6 = "S6"
    S7 = "S7"


STAGE_SKILL = {
    Stage.S1: "lift_truss",
    Stage.S2: "pick_and_deposit",
    Stage.S3: "return_truss",
    Stage.S4: "reposition_truss",
    Stage.S5: "lift_truss",
    Stage.S6: "pick_and_deposit",
    Stage.S7: "return_bare_stem",
}
PROMPTS = {
    "lift_truss": "Right hand grasps and lifts the truss to the center",
    "pick_and_deposit": "Right hand holds truss while left hand picks tomatoes into left basket",
    "return_truss": "Right hand returns the truss to the right plate",
    "reposition_truss": "Right fingers reposition the truss on the right plate",
    "return_bare_stem": "Right hand returns the bare stem to the right plate",
    # Proposed recovery prompts: require separate demonstration and validation.
    "disengage_left": "Withdraw the left hand from the truss while the right hand supports it",
    "deposit_held_fruit": "Place the detached tomato held by the left hand into the left basket",
}

# Every physical fact is external evidence, never inferred from a command alone.
PREDICATES = frozenset(
    {
        "target_ready",
        "truss_on_plate",
        "left_clear",
        "left_empty",
        "right_stable",
        "at_pick_region",
        "fruit_remaining",
        "no_remaining",
        "plate_available",
        "right_released",
        "reposition_ready",
        "reposition_needed",
        "reposition_benefit",
        "fruit_held_detached",
        "contact_releasable",
        "deposit_path_ready",
        "return_start_valid",
        "stow_verified",
        "ledger_resolved",
        "damage",
        "drop",
        "control_fault",
        "contact_hazard",
        "execution_consistent",
        "fruit_separated",
        "deposit_verified",
    }
)


def finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


@dataclass(frozen=True)
class Config:
    # Draft evaluation values, NOT calibrated hardware operating limits.
    max_age_s: float = 0.5
    max_skew_s: float = 0.08
    future_tolerance_s: float = 0.02
    stable_s: float = 0.15
    decision_ttl_s: float = 0.5
    no_progress_s: float = 15.0
    stage_timeout_s: float = 60.0
    episode_timeout_s: float = 300.0
    max_failures_per_target: int = 2
    max_total_attempts_per_target: int = 4
    max_unassociated_failures: int = 3
    max_repositions: int = 2
    observation_timeout_s: float = 5.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            finite(value, name)
            if name.startswith("max_") and name in {
                "max_failures_per_target",
                "max_total_attempts_per_target",
                "max_unassociated_failures",
                "max_repositions",
            }:
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Snapshot:
    observation_id: str
    episode_id: str
    now: float  # Source clock, not process monotonic time.
    state_time: float
    image_times: dict[str, float]
    joint_pos: tuple[float, ...]
    clock_id: str = "replay"
    observed_activity: str = "UNKNOWN"
    target_id: str | None = None

    def invalid_reason(self, cfg: Config) -> str | None:
        if not self.observation_id or not self.episode_id or not self.clock_id:
            return "missing observation, episode or clock identity"
        try:
            now = finite(self.now, "now")
            stamps = [finite(self.state_time, "state_time")]
            if set(self.image_times) != {"cam0", "cam1", "cam2"}:
                return "three camera timestamps required"
            stamps += [finite(v, "image_time") for v in self.image_times.values()]
            if len(self.joint_pos) != 54:
                return "54 measured joint positions required"
            for value in self.joint_pos:
                finite(value, "joint_pos")
        except ValueError as error:
            return str(error)
        if now - min(stamps) > cfg.max_age_s:
            return "stale observation"
        if max(stamps) - now > cfg.future_tolerance_s:
            return "future observation"
        if max(stamps) - min(stamps) > cfg.max_skew_s:
            return "unsynchronized observation"
        return None


@dataclass(frozen=True)
class Evidence:
    predicate: str
    value: Truth
    observation_id: str
    timestamp: float
    source: str
    references: tuple[str, ...]

    def __post_init__(self):
        if self.predicate not in PREDICATES:
            raise ValueError(f"unknown predicate: {self.predicate}")
        if not isinstance(self.value, Truth):
            raise ValueError("evidence value must be Truth")
        finite(self.timestamp, "evidence timestamp")
        if (
            not self.source
            or not self.observation_id
            or not self.references
            or not all(self.references)
        ):
            raise ValueError("evidence requires source, observation and references")


@dataclass(frozen=True)
class Event:
    event_id: str
    kind: str
    observation_id: str
    target_id: str | None = None
    attempt_id: str | None = None
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    decision_id: str
    episode_id: str
    expected_epoch: int
    observation_id: str
    created_at: float
    expires_at: float
    kind: str
    reason: str
    target_stage: Stage | None = None
    skill: str | None = None
    required: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    outcome: str | None = None
    evidence_refs: tuple[str, ...] = ()
    exit_after_return: bool = False


@dataclass
class Memory:
    episode_id: str
    started_at: float
    stage: Stage | None = None
    epoch: int = 0
    round_id: int = 0
    stage_since: float = 0.0
    last_pick_progress: float = 0.0
    repositions: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    round_failure_baseline: dict[str, int] = field(default_factory=dict)
    unassociated_failures: int = 0
    separated: set[str] = field(default_factory=set)
    deposited: set[str] = field(default_factory=set)
    attempts: set[str] = field(default_factory=set)
    events: dict[str, Event] = field(default_factory=dict)
    visits: list[str] = field(default_factory=list)
    exit_after_return: bool = False
    damage_detected: bool = False
    drop_detected: bool = False
    outcome: str | None = None
    recovery_skill: str | None = None
    recovery_since: float = 0.0


def jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [
            jsonable(v) for v in (sorted(value) if isinstance(value, set) else value)
        ]
    return value
