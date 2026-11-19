import hashlib
import importlib
import sys

import pytest

from bikeml.common import ROOT, load_config, load_json, model_dir

sys.path.insert(0, str(ROOT / "lambda_src"))
lf = importlib.import_module("lambda_function")

MODEL = model_dir(load_config()) / "model.joblib"
GOOD = {"instances": [{"datetime": "2012-10-01T08:00", "holiday": False, "weather": 1, "temp_c": 17.5,
                       "feels_like_c": 21.0, "humidity": 72, "windspeed_kmh": 9.0}]}


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    lf._CACHE.clear()
    for k in ("MODEL_URI", "MODEL_VERSION", "MODEL_SHA256"):
        monkeypatch.delenv(k, raising=False)
    yield
    lf._CACHE.clear()


def error_code(resp):
    return resp["statusCode"], resp["body"]["error"]["code"]


def test_missing_model_uri(monkeypatch):
    assert error_code(lf.lambda_handler(GOOD, None)) == (500, "MODEL_LOAD_FAILED")


def test_model_path_does_not_exist(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_URI", str(tmp_path / "model.pkl"))
    resp = lf.lambda_handler(GOOD, None)
    assert error_code(resp) == (500, "MODEL_LOAD_FAILED")
    assert "model.pkl" in resp["body"]["error"]["message"]


def test_checksum_mismatch(monkeypatch, tmp_path):
    p = tmp_path / "model.joblib"
    p.write_bytes(b"not the model")
    monkeypatch.setenv("MODEL_URI", str(p))
    monkeypatch.setenv("MODEL_SHA256", "0" * 64)
    resp = lf.lambda_handler(GOOD, None)
    assert error_code(resp) == (500, "MODEL_LOAD_FAILED") and "sha256" in resp["body"]["error"]["message"]


def test_corrupt_model_file(monkeypatch, tmp_path):
    p = tmp_path / "model.joblib"
    p.write_bytes(b"not the model")
    monkeypatch.setenv("MODEL_URI", str(p))
    monkeypatch.setenv("MODEL_SHA256", hashlib.sha256(b"not the model").hexdigest())
    assert error_code(lf.lambda_handler(GOOD, None)) == (500, "MODEL_LOAD_FAILED")


def test_failure_is_not_cached_and_recovers(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_URI", str(tmp_path / "model.pkl"))
    assert lf.lambda_handler(GOOD, None)["statusCode"] == 500
    meta = load_json(MODEL.parent / "metadata.json")
    monkeypatch.setenv("MODEL_URI", str(MODEL))
    monkeypatch.setenv("MODEL_VERSION", meta["version"])
    monkeypatch.setenv("MODEL_SHA256", meta["model_sha256"])
    resp = lf.lambda_handler(GOOD, None)
    assert resp["statusCode"] == 200 and resp["body"]["model_sha256"] == meta["model_sha256"]
    assert resp["body"]["runtime"]["scikit-learn"] == meta["library_versions"]["scikit-learn"]


def test_model_with_wrong_feature_count(monkeypatch, tmp_path):
    import joblib
    import numpy as np
    from sklearn.linear_model import LinearRegression

    p = tmp_path / "model.joblib"
    joblib.dump(LinearRegression().fit(np.eye(3), [1, 2, 3]), p)
    monkeypatch.setenv("MODEL_URI", str(p))
    resp = lf.lambda_handler(GOOD, None)
    assert error_code(resp) == (500, "MODEL_LOAD_FAILED") and "입력 개수" in resp["body"]["error"]["message"]


def test_unexpected_error_during_predict():
    class Broken:
        def predict(self, X):
            raise RuntimeError("boom")
    lf._CACHE["model"], lf._CACHE["info"] = Broken(), {"version": "x", "sha256": "y"}
    resp = lf.lambda_handler(GOOD, None)
    assert error_code(resp) == (500, "INTERNAL_ERROR") and "boom" in resp["body"]["error"]["message"]


def test_parse_s3_uri_with_version():
    assert lf.parse_s3_uri("s3://b/models/v1/model.joblib?versionId=abc") == ("b", "models/v1/model.joblib", "abc")
    assert lf.parse_s3_uri("s3://b/models/v1/model.joblib") == ("b", "models/v1/model.joblib", None)
