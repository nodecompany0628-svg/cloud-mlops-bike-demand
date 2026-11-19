from __future__ import annotations

import hashlib
import json
import os
import platform
import time

from bikeml import service
from bikeml.features import FEATURES

_CACHE: dict = {}


class ModelLoadError(Exception):
    pass


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_s3_uri(uri: str) -> tuple[str, str, str | None]:
    rest = uri[len("s3://"):]
    rest, _, query = rest.partition("?versionId=")
    bucket, _, key = rest.partition("/")
    return bucket, key, query or None


def _download(uri: str) -> str:
    if not uri.startswith("s3://"):
        return uri
    import boto3

    bucket, key, version = parse_s3_uri(uri)
    local = "/tmp/model.joblib"
    extra = {"VersionId": version} if version else None
    boto3.client("s3").download_file(bucket, key, local, ExtraArgs=extra)
    return local


def _runtime() -> dict:
    import boto3
    import numpy
    import sklearn

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "boto3": boto3.__version__,
        "execution_env": os.environ.get("AWS_EXECUTION_ENV", "local"),
    }


def _load_model():
    if "model" in _CACHE:
        return _CACHE["model"], _CACHE["info"]
    uri = os.environ.get("MODEL_URI")
    if not uri:
        raise ModelLoadError("환경 변수 MODEL_URI 가 없습니다")
    t0 = time.perf_counter()
    try:
        path = _download(uri)
        sha = _sha256(path)
    except Exception as e:
        raise ModelLoadError(f"모델 파일을 가져오지 못했습니다 ({uri}): {type(e).__name__}: {e}") from e
    expected = os.environ.get("MODEL_SHA256")
    if expected and sha != expected:
        raise ModelLoadError(f"모델 sha256 불일치: 기대 {expected[:12]}..., 실제 {sha[:12]}...")
    try:
        import joblib

        model = joblib.load(path)
    except Exception as e:
        raise ModelLoadError(f"모델 파일을 읽지 못했습니다: {type(e).__name__}: {e}") from e
    n_features = getattr(model, "n_features_in_", None)
    if n_features != len(FEATURES):
        raise ModelLoadError(f"모델 입력 개수 {n_features} 가 서비스 입력 개수 {len(FEATURES)} 와 다릅니다")
    info = {"version": os.environ.get("MODEL_VERSION"), "sha256": sha, "runtime": _runtime()}
    _CACHE["model"], _CACHE["info"] = model, info
    _log({"event": "model_loaded", "model_version": info["version"], "sha256": sha,
          "runtime": info["runtime"], "load_ms": round((time.perf_counter() - t0) * 1000, 1)})
    return model, info


def _log(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def _fail(request_id: str, code: str, message: str) -> dict:
    _log({"event": "error", "request_id": request_id, "code": code, "message": message})
    return {"statusCode": 500, "body": {"ok": False, "error": {"code": code, "message": message}}}


def lambda_handler(event, context):
    t0 = time.perf_counter()
    request_id = getattr(context, "aws_request_id", "local")
    try:
        model, info = _load_model()
    except ModelLoadError as e:
        return _fail(request_id, "MODEL_LOAD_FAILED", str(e))
    try:
        resp = service.handle(event, model, info)
    except Exception as e:
        return _fail(request_id, "INTERNAL_ERROR", f"{type(e).__name__}: {e}")
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
