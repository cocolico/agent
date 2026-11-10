"""Generate explicit synthetic evidence, never robot demonstrations or training labels."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tomato_agent.models import Snapshot, jsonable  # noqa: E402


BASE = {
    "control_fault": False,
    "contact_hazard": False,
    "damage": False,
    "drop": False,
    "execution_consistent": True,
    "left_empty": True,
    "left_clear": True,
    "ledger_resolved": True,
    "plate_available": True,
    "target_ready": True,
    "fruit_held_detached": False,
    "return_start_valid": True,
    "fruit_remaining": True,
    "no_remaining": False,
}


def demo_rows(recovery=False):
    phases = [
        ({"truss_on_plate": True}, []),
        ({"right_stable": True, "at_pick_region": True}, []),
    ]
    if recovery:
        phases += [
            (
                {
                    "right_stable": True,
                    "at_pick_region": True,
                    "reposition_needed": True,
                    "reposition_benefit": True,
                    "left_clear": False,
                    "contact_releasable": True,
                },
                [],
            ),
            (
                {
                    "right_stable": True,
                    "reposition_needed": True,
                    "reposition_benefit": True,
                },
                [],
            ),
            (
                {
                    "truss_on_plate": True,
                    "right_released": True,
                    "reposition_benefit": True,
                },
                [],
            ),
            ({"truss_on_plate": True, "reposition_ready": True}, []),
            ({"right_stable": True, "at_pick_region": True}, []),
        ]
    phases += [
        ({"right_stable": True, "fruit_separated": True}, [("sep1", "SEPARATED")]),
        ({"right_stable": True, "deposit_verified": True}, [("dep1", "DEPOSITED")]),
        ({"right_stable": True, "fruit_remaining": False, "no_remaining": True}, []),
        (
            {
                "truss_on_plate": True,
                "right_released": True,
                "fruit_remaining": False,
                "no_remaining": True,
                "stow_verified": True,
            },
            [],
        ),
    ]
    n = 0
    for overrides, events in phases:
        for repeat in range(2):
            t = 100.0 + n * 0.2
            ident = f"synthetic-{n}"
            snap = Snapshot(
                ident,
                "demo-recovery" if recovery else "demo-early-finish",
                t,
                t,
                {k: t for k in ("cam0", "cam1", "cam2")},
                (0.0,) * 54,
            )
            facts = BASE | overrides
            yield {
                "schema_version": "tomato-agent-replay-v1",
                "synthetic": True,
                "snapshot": jsonable(snap),
                "evidence": [
                    {
                        "predicate": k,
                        "value": "TRUE" if v else "FALSE",
                        "observation_id": ident,
                        "timestamp": t,
                        "source": "synthetic_fixture",
                        "references": [f"synthetic://{ident}/{k}"],
                    }
                    for k, v in facts.items()
                ],
                "events": [
                    {
                        "event_id": event_id,
                        "kind": kind,
                        "observation_id": ident,
                        "target_id": "fruit-1",
                        "references": [f"synthetic://{ident}/{kind}"],
                    }
                    for event_id, kind in events
                ]
                if repeat
                else [],
                "mock_handoff": repeat == 1,
            }
            n += 1


if __name__ == "__main__":
    target = ROOT / "examples"
    target.mkdir(exist_ok=True)
    for name, recovering in (("early_finish.jsonl", False), ("recovery.jsonl", True)):
        with (target / name).open("w", encoding="utf-8") as stream:
            for row in demo_rows(recovering):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(target / name)
