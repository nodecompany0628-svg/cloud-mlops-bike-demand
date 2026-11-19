from __future__ import annotations

import numpy as np

from .features import FieldError, to_features

MAX_INSTANCES = 1000


def _error(status: int, code: str, message: str, details=None) -> dict:
    body = {"ok": False, "error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return {"statusCode": status, "body": body}


def handle(payload, model, model_info: dict) -> dict:
    if not isinstance(payload, dict) or "instances" not in payload:
        return _error(400, "INVALID_INPUT", "요청에 instances 목록이 필요합니다")
    instances = payload["instances"]
    if not isinstance(instances, list) or not instances:
        return _error(400, "INVALID_INPUT", "instances 는 비어 있지 않은 목록이어야 합니다")
    if len(instances) > MAX_INSTANCES:
        return _error(400, "TOO_MANY_INSTANCES",
                      f"한 번에 최대 {MAX_INSTANCES}건까지 보낼 수 있습니다 (받은 건수 {len(instances)})")

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
        for rec, v in zip(instances, raw, strict=True)
    ]
    body = {
        "ok": True,
        "model_version": model_info.get("version"),
        "model_sha256": model_info.get("sha256"),
        "n": len(preds),
        "unit": "대/시간",
        "predictions": preds,
    }
    if model_info.get("runtime"):
        body["runtime"] = model_info["runtime"]
    if warnings:
        body["warnings"] = warnings[:50]
    return {"statusCode": 200, "body": body}
