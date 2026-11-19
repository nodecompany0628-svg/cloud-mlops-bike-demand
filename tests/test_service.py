import json

import numpy as np

from bikeml.service import MAX_INSTANCES, handle

GOOD = {"datetime": "2012-10-01T08:00", "holiday": False, "weather": 1, "temp_c": 17.5,
        "feels_like_c": 21.0, "humidity": 72, "windspeed_kmh": 9.0}
INFO = {"version": "test", "sha256": "abc"}


class FakeModel:
    """시각(hr)을 그대로 돌려주는 모형. 음수가 나오면 0 으로 잘리는지도 본다."""

    def predict(self, X):
        return np.asarray(X)[:, 2] - 5


def test_ok_response_shape():
    r = handle({"instances": [GOOD, {**GOOD, "datetime": "2012-10-01T02:00"}]}, FakeModel(), INFO)
    assert r["statusCode"] == 200
    b = r["body"]
    assert b["n"] == 2 and b["model_version"] == "test"
    assert [p["predicted_count"] for p in b["predictions"]] == [3, 0]  # 2시 -> -3 -> 0


def test_http_body_string():
    r = handle({"body": json.dumps({"instances": [GOOD]})}, FakeModel(), INFO)
    assert r["statusCode"] == 200


def test_invalid_json_body():
    r = handle({"body": "{not json"}, FakeModel(), INFO)
    assert r["statusCode"] == 400 and r["body"]["error"]["code"] == "INVALID_JSON"


def test_errors_report_index_and_field():
    bad = dict(GOOD)
    del bad["weather"]
    r = handle({"instances": [GOOD, bad]}, FakeModel(), INFO)
    assert r["statusCode"] == 400
    assert r["body"]["error"]["details"] == [{"index": 1, "field": "weather", "message": "필수 입력이 없습니다"}]


def test_limits():
    assert handle({"instances": []}, FakeModel(), INFO)["statusCode"] == 400
    r = handle({"instances": [GOOD] * (MAX_INSTANCES + 1)}, FakeModel(), INFO)
    assert r["body"]["error"]["code"] == "TOO_MANY_INSTANCES"
