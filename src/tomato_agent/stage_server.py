"""Agent-owned Pi05 entry point: per-request prompts, no edits to pi05-server."""

import argparse
import json
from pathlib import Path
import sys

from .models import PROMPTS
from .online.config import local_output


class StagePolicy:
    def __init__(self, policy, allowed_prompts):
        self.policy, self.allowed_prompts = policy, frozenset(allowed_prompts)

    def infer(self, observation, **kwargs):
        prompt = observation.get("prompt")
        epoch = observation.get("agent_epoch", 0)
        if prompt not in self.allowed_prompts:
            raise ValueError("prompt is not registered for this stage service")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("invalid agent epoch")
        # Metadata is kept out of model transforms; prompt itself is passed through.
        model_input = {k: v for k, v in observation.items() if k != "agent_epoch"}
        result = dict(self.policy.infer(model_input, **kwargs))
        result.update(effective_prompt=prompt, agent_epoch=epoch)
        return result


def build_policy(metadata, stats, skills, factory=None):
    if factory is None:
        from pi05_server.runtime import create_policy

        factory = create_policy
    prompt = PROMPTS[skills[0]]
    # Explicit argument, not a global monkeypatch of the original Full profile.
    policy = factory(metadata, stats, prompt, override_prompt=False)
    return StagePolicy(policy, [PROMPTS[skill] for skill in skills])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Agent-owned dynamic-stage Pi05 server"
    )
    parser.add_argument(
        "--server-root", type=Path, default=Path("/home/user/lpy/pi05-server")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--skills", nargs="+", choices=sorted(PROMPTS), default=list(PROMPTS)
    )
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--contract-out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    out = local_output(args.contract_out)
    source = args.server_root.resolve() / "src"
    if not (source / "pi05_server/runtime.py").is_file():
        parser.error("pi05-server source is not available")
    sys.path.insert(0, str(source))
    from pi05_server.profiles import resolve_profile

    profile = resolve_profile(args.checkpoint)
    # Preserve physical shape, ordering, stats and checkpoint identity checks.
    # Unsupported future export formats fail explicitly, never bypass normalization checks.
    metadata, stats = profile.adapter.checkpoint_contract(args.checkpoint)
    metadata.update(
        stage_prompt_mode="per_request",
        effective_prompt_echo=True,
        agent_epoch_echo=True,
        stage_skills=list(args.skills),
        prompt=PROMPTS[args.skills[0]],
        warmed_up=False,
        rtc_warmed_up=False,
        agent_service="tomato-stage-v1",
    )
    if args.check_only:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    import jax
    from pi05_server.contracts import CheckedPolicy, fake_observation
    from pi05_server.client_contract import from_server_metadata, write_atomic
    from pi05_server.warmup import warmup_rtc
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("CUDA GPU required for Pi05 inference")
    policy = CheckedPolicy(build_policy(metadata, stats, args.skills))
    warmup_rtc(policy, fake_observation(PROMPTS[args.skills[0]]), metadata)
    contract = from_server_metadata(metadata)

    def started():
        write_atomic(out, contract)
        print(
            f"Stage Pi05 ready at ws://{args.host}:{args.port}; contract={out}",
            flush=True,
        )

    WebsocketPolicyServer(
        policy, host=args.host, port=args.port, metadata=metadata, on_started=started
    ).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
