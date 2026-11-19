from bikeml.cloud import preflight_problems

SHA = "a" * 64
APP = {"lambda_function.py": "1", "bikeml/service.py": "2"}
GOOD = dict(
    model_sha=SHA,
    model_file_sha=SHA,
    evaluation={"passed": True, "model": {"sha256": SHA}},
    service_check={"all_passed": True, "model": {"sha256": SHA}},
    package_info={"zip_sha256": "z", "app_files": APP, "requirements_sha256": "r"},
    zip_sha="z",
    app_files=APP,
    requirements_sha="r",
)


def problems(**changes):
    return preflight_problems(**{**GOOD, **changes})


def test_all_good():
    assert problems() == []


def test_each_gate():
    assert "metadata.json" in problems(model_file_sha="b" * 64)[0]
    assert "evaluate" in problems(evaluation={"passed": False, "model": {"sha256": SHA}})[0]
    assert "evaluate" in problems(evaluation={"passed": True, "model": {"sha256": "c" * 64}})[0]
    assert "evaluate" in problems(evaluation={})[0]
    assert "check-local" in problems(service_check={"all_passed": False, "model": {"sha256": SHA}})[0]
    assert "zip" in problems(zip_sha=None)[0]
    assert "zip" in problems(zip_sha="other")[0]
    assert "바뀜" in problems(app_files={**APP, "bikeml/service.py": "3"})[0]
    assert "바뀜" in problems(requirements_sha="other")[0]
