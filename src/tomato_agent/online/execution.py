"""Single-owner control thread; semantic switches create entirely new controllers."""

from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass, replace
from queue import Queue, Empty
import threading
import time


@dataclass(frozen=True)
class PreparedReceipt:
    decision_id: str
    stage_epoch: int
    skill: str | None
    hold_confirmed: bool
    active: bool = False


class ExecutionOwner:
    def __init__(self, backend, *, control_hz=100, timeout_s=10, clock=time.monotonic):
        self.backend, self.timeout, self.clock = backend, timeout_s, clock
        self.period = 1 / control_hz
        self.commands = Queue()
        self.exit = threading.Event()
        self.ready = Future()
        self.thread = None
        self.session = None
        self.receipt = None
        self._status = {
            "holding": True,
            "active": False,
            "prepared": False,
            "epoch": 0,
            "skill": None,
            "fault": None,
            "heartbeat": clock(),
        }
        self.lock = threading.Lock()

    def status(self):
        with self.lock:
            return dict(self._status)

    def _update(self, **values):
        with self.lock:
            self._status.update(values)

    def start(self):
        self.thread = threading.Thread(
            target=self._run, name="tomato-control-owner", daemon=True
        )
        self.thread.start()
        try:
            self.ready.result(timeout=self.timeout)
        except BaseException:
            self.exit.set()
            raise
        return self

    def call(self, command, **payload):
        if self.thread is None or not self.thread.is_alive() or self.exit.is_set():
            raise RuntimeError(self.status()["fault"] or "control owner is not running")
        future = Future()
        self.commands.put((command, payload, future))
        try:
            return future.result(timeout=self.timeout)
        except TimeoutError:
            self.exit.set()  # A timed-out activation must never happen later.
            future.cancel()
            raise TimeoutError("control handoff acknowledgement timed out") from None

    def pause(self):
        return self.call("pause")

    def prepare(self, decision):
        return self.call(
            "prepare",
            decision_id=decision.decision_id,
            stage_epoch=decision.expected_epoch + 1,
            skill=decision.skill,
        )

    def activate(self, receipt):
        return self.call("activate", receipt=receipt)

    def _retire(self):
        # Detach immediately; even an unconfirmed hold must never leave an old session dispatchable.
        old, self.session = self.session, None
        self.receipt = None
        self._update(active=False, holding=False, prepared=False, skill=None)
        failure = None
        try:
            self.backend.stop()
            self._update(holding=True)
        except BaseException as error:
            failure = error
        try:
            if old is not None:
                old.close()  # Discard old websocket/worker, timeline and RTC prefix, also on stop failure.
        except BaseException as error:
            failure = failure or error
        if failure is not None:
            raise failure

    def _handle(self, command, data):
        if command == "pause":
            self._retire()
            return self.status()
        if command == "prepare":
            if not self.status()["holding"] or self.session is not None:
                raise RuntimeError("pause and retire the previous skill before prepare")
            if data["stage_epoch"] <= self.status()["epoch"]:
                raise ValueError("obsolete skill epoch")
            if data["skill"] is not None:
                self.session = self.backend.prepare(data["skill"], data["stage_epoch"])
            self.receipt = PreparedReceipt(**data, hold_confirmed=True)
            self._update(prepared=True)
            return self.receipt
        if command == "activate":
            receipt = data["receipt"]
            if receipt != self.receipt or receipt is None or self.exit.is_set():
                raise ValueError("unknown or cancelled activation receipt")
            self._update(
                epoch=receipt.stage_epoch,
                skill=receipt.skill,
                active=receipt.skill is not None,
                holding=receipt.skill is None,
                prepared=False,
            )
            self.receipt = None  # Duplicate activate is rejected.
            return self.status()
        raise ValueError("unsupported control command")

    def _run(self):
        failure = None
        try:
            self.backend.open()
            self.backend.stop()
            self.ready.set_result(True)
            while not self.exit.is_set():
                began = self.clock()
                try:
                    command, payload, future = self.commands.get_nowait()
                except Empty:
                    pass
                else:
                    if future.set_running_or_notify_cancel():
                        try:
                            if self.exit.is_set():
                                raise RuntimeError("control owner is shutting down")
                            result = self._handle(command, payload)
                            future.set_result(result)
                        except BaseException as error:
                            future.set_exception(error)
                            raise
                if self.exit.is_set():
                    break
                if hasattr(self.backend, "check_status"):
                    self.backend.check_status()
                if self.status()["active"]:
                    self.session.tick()
                self._update(heartbeat=self.clock())
                self.exit.wait(max(0, self.period - (self.clock() - began)))
        except BaseException as error:
            failure = error
            self._update(fault=f"{type(error).__name__}: {error}", active=False)
            if not self.ready.done():
                self.ready.set_exception(error)
        finally:
            try:
                self._retire()
            except BaseException as error:
                failure = failure or error
                self._update(fault=f"hold/cleanup failed: {error}", holding=False)
            try:
                self.backend.close()
            except BaseException as error:
                failure = failure or error
                self._update(fault=f"backend close failed: {error}")
            self.exit.set()
            while True:
                try:
                    _, _, future = self.commands.get_nowait()
                except Empty:
                    break
                if not future.done():
                    future.set_exception(failure or RuntimeError("owner closed"))

    def close(self):
        self.exit.set()
        if self.thread:
            self.thread.join(timeout=self.timeout)
            if self.thread.is_alive():
                raise TimeoutError("control thread has not confirmed shutdown")
        if self.status()["fault"]:
            raise RuntimeError(self.status()["fault"])


class MonoBackend:
    """Uses the SDK controller and body; never bypasses their device/trajectory checks."""

    def __init__(self, source, config, emit):
        self.body, self.source, self.config, self.emit = (
            source.definition,
            source,
            config,
            emit,
        )
        self.latest = None
        self.lock = threading.Lock()
        self.connected = False

    def update_observation(self, raw):
        with self.lock:
            self.latest = raw

    def open(self):
        from inspect_robots.scene import Scene

        self.body.reset(
            Scene("tomato_agent", "Hold while the online supervisor observes")
        )
        self.connected = True

    def stop(self):
        # SDK stop waits for fresh hold acknowledgement from all joint groups.
        self.body.stop()

    def check_status(self):
        # Check latched SDK faults/fresh joints even while holding between skills.
        self.body.get_joint_observation()

    def prepare(self, skill, epoch):
        return MonoSkillSession(self, skill, epoch)

    def close(self):
        self.body.close()


class MonoSkillSession:
    def __init__(self, backend, skill, epoch):
        import json
        from pathlib import Path
        from lia_policies.pi05 import Pi05Policy, Pi05WebsocketClient
        from lia_policies.pi05.adapters.franka_wuji import (
            FrankaWujiAdapter,
            wuji_config,
            arm_indices,
        )
        from lia_policies.pi05.execution.config import ExecutionConfig
        from lia_policies.pi05.execution.controller import Pi05Controller
        from ..adapters.lia_mono import (
            PromptCheckedClient,
            require_stage_prompt_contract,
        )
        from ..models import PROMPTS

        self.backend, self.skill, self.epoch = backend, skill, epoch
        self.prompt = PROMPTS[skill]
        self.client = Pi05WebsocketClient(backend.config.pi05_uri, timeout_s=2)
        try:
            metadata = self.client.get_server_metadata()
            require_stage_prompt_contract(metadata)
            if metadata.get(
                "agent_epoch_echo"
            ) is not True or skill not in metadata.get("stage_skills", []):
                raise ValueError(
                    "stage service does not support epoch receipts or selected skill"
                )
            if not set(backend.config.validated_skills) <= set(
                metadata.get("stage_skills", [])
            ):
                raise ValueError(
                    "configured skills are absent from the stage service contract"
                )
            settings = wuji_config(
                **json.loads(Path(backend.config.model_contract).read_text())
            )
            execution = ExecutionConfig(**backend.config.execution)
            self.policy = Pi05Policy(
                backend.body.info,
                client=PromptCheckedClient(self.client, epoch=epoch),
                config=settings,
                adapter=FrankaWujiAdapter(),
            )
            self.controller = Pi05Controller(
                backend.body.info,
                settings,
                execution,
                arm_indices(backend.body.info.action_space.semantics.dim_labels),
                source_now=lambda: backend.body.clock.now_ns() / 1e9,
                abort_inference=self.client.close,
                on_event=lambda row: backend.emit(
                    {
                        "event": "controller",
                        "stage_epoch": epoch,
                        "skill": skill,
                        "detail": row,
                    }
                ),
            )
            self.controller._check_rtc(self.policy)
        except BaseException:
            self.client.close()
            raise

    def tick(self):
        with self.backend.lock:
            raw = self.backend.latest
        if raw is None:
            raise RuntimeError("no fresh observation for skill execution")
        measured = self.backend.body.get_joint_observation()
        observation = replace(
            raw,
            state=measured.state,
            state_time=measured.state_time,
            instruction=self.prompt,
        )
        action = self.controller.next_action(self.policy, observation, None, None)
        if action.meta.get("request_stop"):
            raise RuntimeError("skill execution run budget exhausted")
        self.backend.body.command(action)

    def close(self):
        self.controller.close()
