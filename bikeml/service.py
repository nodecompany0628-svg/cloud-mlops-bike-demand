"""요청 처리: 입력 검사 -> 특성 변환 -> 예측 -> 응답.

Lambda 핸들러와 로컬 검사가 같은 함수를 쓴다. numpy 와 표준 라이브러리만 쓴다.
"""

from __future__ import annotations

import json

import numpy as np

from .features import FieldError, to_features

MAX_INSTANCES = 1000


def _error(status: int, code: str, message: str, details=None) -> dict:
    body = {"ok": False, "error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return {"statusCode": status, "body": body}


def parse_event(event) -> dict:
    """직접 호출({"instances": [...]})과 HTTP 형태({"body": "..."}) 둘 다 받는다."""
    if isinstance(event, dict) and "body" in event and "instances" not in event:
        body = event["body"]
        if isinstance(body, str):
            body = json.loads(body)
        return body
    return event


def handle(event, model, model_info: dict) -> dict:
    try:
        payload = parse_event(event)
    except (ValueError, TypeError) as e:
        return _error(400, "INVALID_JSON", f"요청 본문을 JSON 으로 읽지 못했습니다: {e}")

    if not isinstance(payload, dict) or "instances" not in payload:
        return _error(400, "INVALID_INPUT", "요청에 instances 목록이 필요합니다")
    instances = payload["instances"]
    if not isinstance(instances, list) or not instances:
        return _error(400, "INVALID_INPUT", "instances 는 비어 있지 않은 목록이어야 합니다")
    if len(instances) > MAX_INSTANCES:
        return _error(400, "TOO_MANY_INSTANCES", f"한 번에 최대 {MAX_INSTANCES}건까지 보낼 수 있습니다 (받은 건수 {len(instances)})")

    rows, warnings, details = [], [], []
    for i, rec in enumerate(instances):
        try:
            row, w = to_features(rec)
        except FieldError as e:
            details.append({"index": i, "field": e.field, "message": e.message})
            continue
        rows.append(row)
        warnings.extend(f"[{i}] {m}" for m in w)
    if details:
        return _error(400, "INVALID_INPUT", f"{len(details)}건의 입력이 잘못되었습니다", details[:20])

    X = np.asarray(rows, dtype=float)
    raw = np.clip(model.predict(X), 0.0, None)
    preds = [
        {
            "datetime": rec["datetime"],
            "predicted_count": int(round(float(v))),
            "predicted_value": round(float(v), 2),
        }
        for rec, v in zip(instances, raw)
    ]
    body = {
        "ok": True,
        "model_version": model_info.get("version"),
        "model_sha256": model_info.get("sha256"),
        "n": len(preds),
        "unit": "대/시간",
        "predictions": preds,
    }
    if warnings:
        body["warnings"] = warnings[:50]
    return {"statusCode": 200, "body": body}
