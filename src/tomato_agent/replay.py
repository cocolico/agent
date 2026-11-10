"""Strict JSONL replay: simulated handoff is opt-in on each record."""

import json

from .evidence import EvidenceMonitor
from .models import Config, Event, Evidence, Snapshot, Truth, jsonable
from .supervisor import Supervisor


def decode_record(row):
    if row.get("schema_version") != "tomato-agent-replay-v1":
        raise ValueError("unsupported replay schema")
    fields = dict(row["snapshot"])
    fields["joint_pos"] = tuple(fields["joint_pos"])
    snap = Snapshot(**fields)
    evidence = []
    for item in row.get("evidence", []):
        fields = dict(item)
        fields["value"] = Truth(fields["value"])
        fields["references"] = tuple(fields["references"])
        evidence.append(Evidence(**fields))
    events = [
        Event(**{**item, "references": tuple(item.get("references", []))})
        for item in row.get("events", [])
    ]
    if type(row.get("mock_handoff", False)) is not bool:
        raise ValueError("mock_handoff must be boolean")
    return snap, evidence, events


def replay(records, *, config=None, simulate=False):
    cfg = config or Config()
    monitor, supervisor = EvidenceMonitor(cfg), None
    for row in records:
        snap, evidence, events = decode_record(row)
        if supervisor is None:
            supervisor = Supervisor(snap.episode_id, snap.now, cfg, simulation=simulate)
        facts = monitor.update(snap, evidence)
        proposal = supervisor.tick(snap, facts, events)
        accepted = False
        if (
            simulate
            and row.get("mock_handoff")
            and proposal.kind in {"TRANSITION", "RECOVER", "FINISH"}
        ):
            supervisor.accept_simulated(proposal, snap, facts, handoff_confirmed=True)
            accepted = True
        yield {
            "schema_version": "tomato-agent-log-v1",
            "mode": "simulation" if simulate else "shadow_replay",
            "hardware_executed": False,
            "snapshot": jsonable(snap),
            "facts": jsonable(facts),
            "events": jsonable(events),
            "decision": jsonable(proposal),
            "mock_accepted": accepted,
            "memory": jsonable(supervisor.memory),
        }


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
