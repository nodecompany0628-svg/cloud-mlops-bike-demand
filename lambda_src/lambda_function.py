"""AWS Lambda 진입점.

환경 변수
  MODEL_URI      s3://버킷/models/<버전>/model.joblib (로컬 검사에서는 파일 경로)
  MODEL_VERSION  모델 버전 이름 (예: hgb-v1)
  MODEL_SHA256   모델 파일 sha256. 받은 파일과 다르면 예측하지 않는다.

모델은 실행 환경마다 한 번 내려받아 메모리에 둔다(콜드 스타트 때만 S3 접근).
불러오기에 실패하면 500 을 돌려주고, 다음 요청에서 다시 시도한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

from bikeml import service

_CACHE: dict = {}


class ModelLoadError(Exception):
    pass


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(uri: str) -> str:
    if not uri.startswith("s3://"):
        return uri
    import boto3  # Lambda 런타임에 들어 있음

    bucket, _, key = uri[len("s3://"):].partition("/")
    local = "/tmp/model.joblib"
    boto3.client("s3").download_file(bucket, key, local)
    return local


def _load_model():
    if "model" in _CACHE:
        return _CACHE["model"], _CACHE["info"]
    uri = os.environ.get("MODEL_URI")
    if not uri:
        raise ModelLoadError("환경 변수 MODEL_URI 가 없습니다")
    t0 = time.perf_counter()
    try:
        path = _download(uri)
    except Exception as e:  # S3 권한, 경로, 연결 오류
        raise ModelLoadError(f"모델 파일을 가져오지 못했습니다 ({uri}): {type(e).__name__}: {e}") from e
    sha = _sha256(path)
    expected = os.environ.get("MODEL_SHA256")
    if expected and sha != expected:
        raise ModelLoadError(f"모델 sha256 불일치: 기대 {expected[:12]}…, 실제 {sha[:12]}…")
    import joblib

    model = joblib.load(path)
    info = {"version": os.environ.get("MODEL_VERSION"), "sha256": sha, "uri": uri}
    _CACHE["model"], _CACHE["info"] = model, info
    _log({"event": "model_loaded", "model_version": info["version"], "sha256": sha,
          "load_ms": round((time.perf_counter() - t0) * 1000, 1)})
    return model, info


def _log(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def lambda_handler(event, context):
    t0 = time.perf_counter()
    request_id = getattr(context, "aws_request_id", "local")
    try:
        model, info = _load_model()
    except ModelLoadError as e:
        _log({"event": "error", "request_id": request_id, "code": "MODEL_LOAD_FAILED", "message": str(e)})
        return {"statusCode": 500, "body": {"ok": False, "error": {"code": "MODEL_LOAD_FAILED", "message": str(e)}}}

    resp = service.handle(event, model, info)
    body = resp["body"]
    _log({
        "event": "request",
        "request_id": request_id,
        "status": resp["statusCode"],
        "n": body.get("n"),
        "error_code": (body.get("error") or {}).get("code"),
        "model_version": info["version"],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    })
    return resp
