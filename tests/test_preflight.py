from bikeml.cloud import preflight_problems

SHA = "a" * 64
APP = {"lambda_function.py": "1", "bikeml/service.py": "2"}
CRIT = {"must_beat_all_baselines": True, "min_mae_reduction_vs_hour_baseline": 0.3}
GOOD = dict(
    model_sha=SHA,
    model_file_sha=SHA,
    evaluation={"passed": True, "model": {"sha256": SHA}, "criteria": CRIT, "eval_data": {"sha256": "t"}},
    service_check={"all_passed": True, "model": {"sha256": SHA}, "app_files": APP},
    package_info={"zip_sha256": "z", "app_files": APP, "requirements_sha256": "r"},
    zip_sha="z",
    app_files=APP,
    requirements_sha="r",
    criteria=CRIT,
    test_sha="t",
)


def problems(**changes):
    return preflight_problems(**{**GOOD, **changes})


def test_all_good():
    assert problems() == []


def test_model_and_evaluation_gates():
    assert "metadata.json" in problems(model_file_sha="b" * 64)[0]
    assert "evaluate" in problems(evaluation={**GOOD["evaluation"], "passed": False})[0]
    assert "evaluate" in problems(evaluation={**GOOD["evaluation"], "model": {"sha256": "c" * 64}})[0]
    assert "evaluate" in problems(evaluation={})[0]
    assert "합격 기준" in problems(criteria={**CRIT, "min_mae_reduction_vs_hour_baseline": 0.1})[0]
    assert "평가 자료" in problems(test_sha="other")[0]


def test_service_check_and_package_gates():
    assert "check-local" in problems(service_check={**GOOD["service_check"], "all_passed": False})[0]
    assert "Lambda 코드" in problems(service_check={**GOOD["service_check"], "app_files": {}})[0]
    assert "zip" in problems(zip_sha=None)[0]
    assert "zip" in problems(zip_sha="other")[0]
    changed = {**APP, "bikeml/service.py": "3"}
    found = problems(app_files=changed, service_check={**GOOD["service_check"], "app_files": changed})
    assert len(found) == 1 and "requirements-lambda" in found[0]
    assert "requirements-lambda" in problems(requirements_sha="other")[0]
