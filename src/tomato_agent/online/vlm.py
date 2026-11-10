"""Local OpenAI-compatible vision endpoint. No dependency on/download of model weights."""

from dataclasses import dataclass, replace
from concurrent.futures import Future
import json
import threading
import urllib.request

from ..models import Evidence, Event, PREDICATES, PROMPTS, Snapshot, Truth, jsonable

VISUAL_PREDICATES = PREDICATES - {"control_fault"}
INTENTS = {"CONTINUE", "SWITCH", "RECOVER", "OBSERVE", "STOP", "FINISH"}
SYSTEM_PROMPT = """你是双FR3+双Wuji番茄采摘任务的在线视觉监督器。
右手支撑番茄串，左手摘取并投放左篮。你看到带时间、视角标签的近期图像与当前执行记录。
只根据可见证据判断实际状态。指令不代表实际动作；看不见、遮挡或矛盾必须UNKNOWN。
手闭合不等于抓住，手张开不等于投放，未检测到果实不等于裸杆。
不能生成关节动作、修改设备限值、控制频率或调用工具。仅选择允许技能。
必须原样返回请求的request_id、observation_id、stage_epoch。
只输出JSON对象，不要Markdown。格式：
{"request_id":"...","observation_id":"...","stage_epoch":0,
 "observed_activity":"S1|S2|S3|S4|S5|S6|S7|UNKNOWN",
 "target_id":null,
 "facts":{"left_clear":{"value":"TRUE|FALSE|UNKNOWN","frame_ids":["实际提供的帧ID"]}},
 "events":[{"event_id":"稳定唯一ID","kind":"SEPARATED|DEPOSITED|ATTEMPT_FAILED|ATTEMPT_UNRESOLVED",
 "target_id":null,"attempt_id":null,"frame_ids":["帧ID"]}],
 "intent":"CONTINUE|SWITCH|RECOVER|OBSERVE|STOP|FINISH",
 "next_skill":null,"reason":"简短可复核依据"}
同一事件重复看到必须使用同一event_id；分离和投放必须关联同一果实。
无法稳定关联目标时不要发明确分离/投放事件，使用UNKNOWN或ATTEMPT_UNRESOLVED。
可省略无法判断的谓词；control_fault由SDK产生，你不能输出。
execution_consistent表示观察到的行为与当前技能是否一致；系统保持时不应误判成执行停滞。
reposition_needed表示当前采摘需要重摆，reposition_benefit须有具体可见改善理由。
"""


@dataclass(frozen=True)
class VisionFrame:
    frame_id: str
    observation_id: str
    camera: str
    source_time: float
    data_url: str


@dataclass(frozen=True)
class VisionRequest:
    request_id: str
    stage_epoch: int
    snapshot: Snapshot
    frames: tuple[VisionFrame, ...]
    memory: dict
    execution: dict


@dataclass(frozen=True)
class VisionJudgement:
    snapshot: Snapshot
    evidence: tuple[Evidence, ...]
    events: tuple[Event, ...]
    intent: str
    next_skill: str | None
    reason: str
    request_id: str
    stage_epoch: int


def request_payload(request, config):
    context = {
        "request_id": request.request_id,
        "observation_id": request.snapshot.observation_id,
        "stage_epoch": request.stage_epoch,
        "snapshot": jsonable(request.snapshot),
        "memory": request.memory,
        "execution": request.execution,
        "allowed_predicates": sorted(VISUAL_PREDICATES),
        "allowed_skills": list(PROMPTS),
    }
    content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}]
    for frame in request.frames:
        content.extend(
            [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "frame_id": frame.frame_id,
                            "camera": frame.camera,
                            "source_time": frame.source_time,
                        }
                    ),
                },
                {"type": "image_url", "image_url": {"url": frame.data_url}},
            ]
        )
    return {
        "model": config.vlm_model,
        "temperature": 0,
        "max_tokens": config.max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    }


def parse_judgement(value, request):
    if isinstance(value, str):
        value = json.loads(value)
    required = {
        "request_id",
        "observation_id",
        "stage_epoch",
        "observed_activity",
        "target_id",
        "facts",
        "events",
        "intent",
        "next_skill",
        "reason",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("VLM response fields do not match the supervision schema")
    if (
        value["request_id"] != request.request_id
        or value["observation_id"] != request.snapshot.observation_id
        or type(value["stage_epoch"]) is not int
        or value["stage_epoch"] != request.stage_epoch
    ):
        raise ValueError("VLM response belongs to an obsolete request/stage")
    if value["observed_activity"] not in {
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "UNKNOWN",
    }:
        raise ValueError("invalid observed activity")
    if value["intent"] not in INTENTS or value["next_skill"] not in {None, *PROMPTS}:
        raise ValueError("VLM requested an unsupported skill/intent")
    if not isinstance(value["reason"], str) or not 1 <= len(value["reason"]) <= 1000:
        raise ValueError("brief decision reason required")
    if value["target_id"] is not None and (
        not isinstance(value["target_id"], str) or not value["target_id"]
    ):
        raise ValueError("invalid target identity")
    frame_map = {f.frame_id: f for f in request.frames}

    def references(ids):
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(i, str) or i not in frame_map for i in ids)
        ):
            raise ValueError("VLM cited absent image evidence")
        return tuple(f"frame://{i}" for i in ids), max(
            frame_map[i].source_time for i in ids
        )

    if (
        not isinstance(value["facts"], dict)
        or not set(value["facts"]) <= VISUAL_PREDICATES
    ):
        raise ValueError("VLM may only emit visual predicates")
    evidence = []
    for name, fact in value["facts"].items():
        if not isinstance(fact, dict) or set(fact) != {"value", "frame_ids"}:
            raise ValueError("invalid predicate record")
        refs, timestamp = references(fact["frame_ids"])
        evidence.append(
            Evidence(
                name,
                Truth(fact["value"]),
                request.snapshot.observation_id,
                timestamp,
                "local_vlm",
                refs,
            )
        )
    if not isinstance(value["events"], list) or len(value["events"]) > 32:
        raise ValueError("invalid event list")
    events = []
    for event in value["events"]:
        if not isinstance(event, dict) or set(event) != {
            "event_id",
            "kind",
            "target_id",
            "attempt_id",
            "frame_ids",
        }:
            raise ValueError("invalid event record")
        if event["kind"] not in {
            "SEPARATED",
            "DEPOSITED",
            "ATTEMPT_FAILED",
            "ATTEMPT_UNRESOLVED",
        }:
            raise ValueError("invalid event kind")
        for key in ("event_id", "target_id", "attempt_id"):
            if event[key] is not None and (
                not isinstance(event[key], str) or not 1 <= len(event[key]) <= 128
            ):
                raise ValueError("invalid event identity")
        if not event["event_id"]:
            raise ValueError("event_id required")
        refs, _ = references(event["frame_ids"])
        events.append(
            Event(
                event["event_id"],
                event["kind"],
                request.snapshot.observation_id,
                event["target_id"],
                event["attempt_id"],
                refs,
            )
        )
    snap = replace(
        request.snapshot,
        observed_activity=value["observed_activity"],
        target_id=value["target_id"],
    )
    return VisionJudgement(
        snap,
        tuple(evidence),
        tuple(events),
        value["intent"],
        value["next_skill"],
        value["reason"],
        request.request_id,
        request.stage_epoch,
    )


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("local VLM endpoint must not redirect")


class LocalVisionClient:
    def __init__(self, config):
        self.config = config
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        )

    def judge(self, request):
        payload = json.dumps(
            request_payload(request, self.config), allow_nan=False
        ).encode()
        http = urllib.request.Request(
            self.config.vlm_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.opener.open(http, timeout=self.config.vlm_timeout_s) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("VLM response exceeds size budget")
        result = json.loads(raw)
        choice = result["choices"][0]
        if choice.get("finish_reason") not in (None, "stop"):
            raise ValueError("VLM response was truncated or refused")
        return parse_judgement(choice["message"]["content"], request)


class SingleFlightVision:
    """At most one HTTP inference; shutdown cannot hang waiting for a stuck model."""

    def __init__(self):
        self.pending = None
        self.closed = False

    def submit(self, function, request):
        if self.closed or (self.pending is not None and not self.pending.done()):
            raise RuntimeError("visual inference is closed or already in flight")
        result = Future()
        self.pending = result

        def work():
            try:
                result.set_result(function(request))
            except BaseException as error:
                result.set_exception(error)

        threading.Thread(target=work, name="tomato-vlm", daemon=True).start()
        return result

    def close(self):
        self.closed = True
