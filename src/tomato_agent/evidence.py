"""Causal stabilization of external evidence; deliberately no fake visual detector."""

from dataclasses import dataclass

from .models import Config, Evidence, PREDICATES, Snapshot, Truth


@dataclass(frozen=True)
class Facts:
    values: dict[str, Truth]
    references: dict[str, tuple[str, ...]]
    conflicts: tuple[str, ...] = ()
    observation_id: str | None = None

    def is_true(self, key):
        return self.values.get(key, Truth.UNKNOWN) == Truth.TRUE

    def is_false(self, key):
        return self.values.get(key, Truth.UNKNOWN) == Truth.FALSE

    def missing(self, keys):
        return tuple(k for k in keys if not self.is_true(k))


class EvidenceMonitor:
    def __init__(self, config: Config):
        self.config = config
        self.history = {}
        self.last_sample = None
        self.last_now = None
        self.identity = None

    def update(self, snapshot: Snapshot, evidence: list[Evidence]) -> Facts:
        invalid = snapshot.invalid_reason(self.config)
        identity = (snapshot.episode_id, snapshot.clock_id)
        if self.identity not in (None, identity):
            invalid = "episode or clock changed"
        if self.last_now is not None and snapshot.now < self.last_now:
            invalid = "clock moved backwards"
        self.identity, self.last_now = identity, snapshot.now
        if invalid:
            self.history.clear()
            return Facts({}, {}, (invalid,), snapshot.observation_id)
        # Re-reading the same camera frames must not accumulate visual evidence.
        sample = tuple(snapshot.image_times[k] for k in sorted(snapshot.image_times))
        new_sample = sample != self.last_sample
        self.last_sample = sample
        grouped = {}
        for item in evidence:
            age = snapshot.now - item.timestamp
            if (
                item.observation_id != snapshot.observation_id
                or age < -self.config.future_tolerance_s
                or age > self.config.max_age_s
            ):
                continue
            grouped.setdefault(item.predicate, []).append(item)
        values, references, conflicts = {}, {}, []
        for key in PREDICATES:
            rows = grouped.get(key, [])
            candidates = {r.value for r in rows}
            value = next(iter(candidates)) if len(candidates) == 1 else Truth.UNKNOWN
            if len(candidates) > 1:
                conflicts.append(key)
            if value == Truth.UNKNOWN:
                self.history.pop(key, None)
                values[key] = Truth.UNKNOWN
                continue
            old = self.history.get(key)
            if (
                old is None
                or old[0] != value
                or snapshot.now - old[2] > self.config.max_age_s
            ):
                old = (value, snapshot.now, snapshot.now, 1)
            elif new_sample:
                old = (value, old[1], snapshot.now, old[3] + 1)
            self.history[key] = old
            # Immediate fault candidates bypass normal completion debouncing.
            immediate = (
                key in {"control_fault", "contact_hazard", "damage", "drop"}
                and value == Truth.TRUE
            )
            stable = old[3] >= 2 and snapshot.now - old[1] >= self.config.stable_s
            values[key] = value if stable or immediate else Truth.UNKNOWN
            references[key] = tuple(
                sorted({ref for row in rows for ref in row.references})
            )
        # Contradictory physical descriptions invalidate both predicates.
        for a, b in (
            ("fruit_remaining", "no_remaining"),
            ("left_empty", "fruit_held_detached"),
        ):
            if values.get(a) == values.get(b) == Truth.TRUE:
                values[a] = values[b] = Truth.UNKNOWN
                self.history.pop(a, None)
                self.history.pop(b, None)
                conflicts.extend((a, b))
        return Facts(
            values, references, tuple(sorted(set(conflicts))), snapshot.observation_id
        )
