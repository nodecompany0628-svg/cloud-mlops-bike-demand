from bikeml.servicecheck import check_response

EXPECTED = {"version": "hgb-v1", "sha256": "s", "sklearn": "1.9.1", "numpy": "2.5.5"}
CASE = {"name": "one", "expect": {"status": 200, "n": 1}}


def response(runtime):
    return {"statusCode": 200, "body": {"n": 1, "predictions": [{"predicted_count": 3, "predicted_value": 3.2}],
                                        "model_version": "hgb-v1", "model_sha256": "s", "runtime": runtime}}


def test_runtime_library_versions_must_match_training():
    assert check_response(CASE, response({"scikit-learn": "1.9.1", "numpy": "2.5.5"}), EXPECTED, {}) == []
    found = check_response(CASE, response({"scikit-learn": "1.9.1", "numpy": "2.4.0"}), EXPECTED, {})
    assert found == ["실행 환경 numpy 2.4.0 (기대 2.5.5)"]
    assert len(check_response(CASE, response({}), EXPECTED, {})) == 2
