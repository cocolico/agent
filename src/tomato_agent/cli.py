import argparse
import json
from pathlib import Path
import sys
import time
import uuid

from .adapters.lia_mono import ReadOnlySource, audit_local_sdk
from .evidence import EvidenceMonitor
from .models import Config, jsonable
from .replay import read_jsonl, replay
from .supervisor import Supervisor


def write_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="番茄 Agent 初稿：回放和只读影子运行，不提供真机动作入口"
    )
    parser.add_argument(
        "--config", type=Path, help="监督配置 JSON；默认数值仅供初稿评估"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("replay", help="读取含物理证据的 JSONL")
    run.add_argument("input", type=Path)
    run.add_argument(
        "--simulate", action="store_true", help="允许显式 mock_handoff 推进模拟阶段"
    )
    run.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("inspect-sdk", help="只读源码/配置清单，不连接设备")
    shadow = commands.add_parser(
        "shadow", help="订阅现有 ROS 观测；缺失的物理谓词保持 UNKNOWN"
    )
    for command in (audit, shadow):
        command.add_argument(
            "--mono-root", type=Path, default=Path("/home/user/lpy/li-agent-mono")
        )
    audit.add_argument(
        "--server-root", type=Path, default=Path("/home/user/lpy/pi05-server")
    )
    audit.add_argument("--output", type=Path)
    shadow.add_argument("--seconds", type=float, default=30.0)
    shadow.add_argument("--sample-hz", type=float, default=5.0)
    shadow.add_argument("--startup-timeout", type=float, default=10.0)
    shadow.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = Config(**json.loads(args.config.read_text())) if args.config else Config()
    if args.command == "inspect-sdk":
        report = audit_local_sdk(args.mono_root, args.server_root)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                write_record(stream, report)
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwriting an existing experiment log.
    with args.output.open("x", encoding="utf-8") as stream:
        if args.command == "replay":
            for record in replay(
                read_jsonl(args.input), config=cfg, simulate=args.simulate
            ):
                write_record(stream, record)
        else:
            from .models import finite

            for name in ("seconds", "sample_hz", "startup_timeout"):
                if finite(getattr(args, name), name) <= 0:
                    raise ValueError(f"{name} must be positive")
            monitor, supervisor = EvidenceMonitor(cfg), None
            episode_id = "shadow-" + uuid.uuid4().hex
            try:
                with ReadOnlySource(
                    args.mono_root,
                    episode_id=episode_id,
                    startup_timeout_s=args.startup_timeout,
                ) as source:
                    end = time.monotonic() + args.seconds
                    while time.monotonic() < end:
                        snap = source.read()
                        if supervisor is None:
                            supervisor = Supervisor(episode_id, snap.now, cfg)
                        # No visual model is trained/connected in this initial draft.
                        facts = monitor.update(snap, [])
                        decision = supervisor.tick(snap, facts)
                        write_record(
                            stream,
                            {
                                "schema_version": "tomato-agent-log-v1",
                                "mode": "readonly_shadow",
                                "hardware_executed": False,
                                "snapshot": jsonable(snap),
                                "facts": jsonable(facts),
                                "decision": jsonable(decision),
                            },
                        )
                        time.sleep(1 / args.sample_hz)
            except BaseException as error:
                write_record(
                    stream,
                    {
                        "mode": "readonly_shadow",
                        "event": "observation_source_closed",
                        "hardware_executed": False,
                        "error": str(error),
                    },
                )
                raise
    print(str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"tomato-agent: {error}", file=sys.stderr)
        raise SystemExit(2)
