"""Live entry point. VLM calls are asynchronous and never run in the control thread."""

import argparse
import base64
from collections import deque
import io
import json
from pathlib import Path
from queue import SimpleQueue, Empty
import sys
import time
import uuid

from ..adapters.lia_mono import ReadOnlySource
from ..models import Config, jsonable
from .config import OnlineConfig, local_output
from .coordinator import OnlineCoordinator
from .execution import ExecutionOwner, MonoBackend
from .vlm import LocalVisionClient, VisionFrame, VisionRequest, SingleFlightVision


def encode_frames(raw, snapshot, max_side):
    from PIL import Image

    result = []
    for camera in ("cam0", "cam1", "cam2"):
        pixels = raw.images[camera]
        if str(pixels.dtype) != "uint8" or pixels.ndim != 3 or pixels.shape[-1] != 3:
            raise ValueError("expected SDK RGB uint8 image")
        picture = Image.fromarray(pixels)
        picture.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=80)
        result.append(
            VisionFrame(
                f"{snapshot.observation_id}:{camera}",
                snapshot.observation_id,
                camera,
                snapshot.image_times[camera],
                "data:image/jpeg;base64,"
                + base64.b64encode(buffer.getvalue()).decode(),
            )
        )
    return tuple(result)


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not serializable: {type(value).__name__}")


def run_live(config):
    if not config.validated_skills:
        raise ValueError("configure validated_skills before live execution")
    if not Path(config.model_contract).is_file():
        raise ValueError(
            "start the agent-owned stage server to create model_contract first"
        )
    episode_id = "online-" + uuid.uuid4().hex
    log_path = local_output(config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    messages = SimpleQueue()
    pool = SingleFlightVision()
    source = executor = None
    with log_path.open("x", encoding="utf-8") as log:

        def flush():
            while True:
                try:
                    row = messages.get_nowait()
                except Empty:
                    break
                log.write(
                    json.dumps(
                        jsonable(row),
                        ensure_ascii=False,
                        allow_nan=False,
                        default=_json_default,
                    )
                    + "\n"
                )
            log.flush()

        try:
            source = ReadOnlySource(config.mono_root, episode_id=episode_id)
            # Match controller frequency while preserving configured camera topics and dimensions.
            if source.definition.info.control_hz != config.execution.get(
                "control_hz", 100
            ):
                raise ValueError("SDK control frequency must match execution config")
            backend = MonoBackend(source, config, messages.put)
            # Metadata/layout/RTC checks before opening ROS control or generating actions.
            probe = backend.prepare(config.validated_skills[0], 1)
            probe.close()
            source.open()
            first = source.read()
            if first.invalid_reason(Config()):
                raise ValueError(first.invalid_reason(Config()))
            backend.update_observation(source.raw_observation)

            def refresh():
                snapshot = source.read()
                backend.update_observation(source.raw_observation)
                return snapshot

            executor = ExecutionOwner(
                backend,
                control_hz=config.execution.get("control_hz", 100),
                timeout_s=config.handoff_timeout_s,
            )
            executor.start()
            coordinator = OnlineCoordinator(
                episode_id, first.now, config, executor, messages.put
            )
            client = LocalVisionClient(config)
            history = deque(maxlen=config.history_frames)
            future = pending_request = None
            began = last_good = last_dispatch = time.monotonic()
            last_frame_times = None
            while (
                not coordinator.terminal
                and time.monotonic() - began < config.max_runtime_s
            ):
                loop_start = time.monotonic()
                current = source.read()
                invalid = current.invalid_reason(Config())
                if invalid:
                    raise RuntimeError(invalid)
                backend.update_observation(source.raw_observation)
                status = executor.status()
                if (
                    status["fault"]
                    or loop_start - status["heartbeat"] > config.handoff_timeout_s
                ):
                    raise RuntimeError(status["fault"] or "control heartbeat expired")
                if future is not None and future.done():
                    try:
                        judgement = future.result()
                        if judgement.stage_epoch == coordinator.supervisor.memory.epoch:
                            processed = coordinator.process(
                                judgement, current, refresh=refresh
                            )
                            if processed is not None:
                                last_good = time.monotonic()
                    except (ValueError, KeyError, OSError, TimeoutError) as error:
                        messages.put({"event": "vlm_error", "error": str(error)})
                        coordinator.hold(current.now, "vlm_unavailable_or_invalid")
                    finally:
                        future = pending_request = None
                if loop_start - last_good > config.supervision_deadline_s:
                    raise TimeoutError("no valid local VLM supervision before deadline")
                if coordinator.terminal:
                    break
                if (
                    future is None
                    and loop_start - last_dispatch >= config.vlm_interval_s
                ):
                    frame_times = tuple(
                        current.image_times[k] for k in sorted(current.image_times)
                    )
                    if frame_times != last_frame_times:
                        frames = encode_frames(
                            source.raw_observation, current, config.image_max_side
                        )
                        history.append(frames)
                        last_frame_times = frame_times
                        pending_request = VisionRequest(
                            uuid.uuid4().hex,
                            coordinator.supervisor.memory.epoch,
                            current,
                            tuple(frame for group in history for frame in group),
                            jsonable(coordinator.supervisor.memory),
                            executor.status(),
                        )
                        future = pool.submit(client.judge, pending_request)
                        last_dispatch = loop_start
                        messages.put(
                            {
                                "event": "vlm_request",
                                "request_id": pending_request.request_id,
                                "snapshot": jsonable(current),
                                "stage_epoch": pending_request.stage_epoch,
                                "frame_ids": [
                                    f.frame_id for f in pending_request.frames
                                ],
                            }
                        )
                flush()
                time.sleep(max(0, 1 / config.poll_hz - (time.monotonic() - loop_start)))
            coordinator.hold(source.read().now, "online_run_finished")
            messages.put(
                {
                    "event": "online_finished",
                    "memory": jsonable(coordinator.supervisor.memory),
                }
            )
        except BaseException as error:
            messages.put({"event": "online_failed", "error": str(error)})
            raise
        finally:
            try:
                if executor:
                    executor.close()
            finally:
                if source:
                    source.close()
                pool.close()
                flush()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="在线视觉大模型监督 π0.5；默认仅检查配置"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true", help="获取设备控制权并运行在线采摘"
    )
    args = parser.parse_args(argv)
    config = OnlineConfig.load(args.config)
    if not args.execute:
        print(json.dumps(config.preflight(), ensure_ascii=False, indent=2))
        return 0
    run_live(config)
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
