from bikeml.cloud import preflight_problems

SHA = "a" * 64
APP = {"lambda_function.py": "1", "bikeml/service.py": "2"}
CRIT = {"must_beat_all_baselines": True, "min_mae_reduction_vs_hour_baseline": 0.3}
PARAMS = {"loss": "poisson", "max_iter": 500}
MODEL_CFG = {"version": "hgb-v1", "params": PARAMS}
LAMBDA = {"runtime": "python3.12", "architecture": "x86_64"}
META = {"model_sha256": SHA, "version": "hgb-v1", "params": PARAMS, "train_data": {"sha256": "tr"}}
GOOD = dict(
    model_meta=META,
    model_file_sha=SHA,
    evaluation={"passed": True, "model": {"sha256": SHA}, "criteria": CRIT, "eval_data": {"sha256": "t"}},
    service_check={"all_passed": True, "model": {"sha256": SHA}, "app_files": APP},
    package_info={"zip_sha256": "z", "app_files": APP, "requirements_sha256": "r", **LAMBDA},
    zip_sha="z",
    app_files=APP,
    requirements_sha="r",
    criteria=CRIT,
    test_sha="t",
    train_sha="tr",
    model_cfg=MODEL_CFG,
    lambda_cfg=LAMBDA,
)


def problems(**changes):
    return preflight_problems(**{**GOOD, **changes})


def test_all_good():
    assert problems() == []


def test_model_gates():
    assert "metadata.json" in problems(model_file_sha="b" * 64)[0]
    assert "학습 자료와 설정" in problems(train_sha="other")[0]
    assert "학습 자료와 설정" in problems(model_cfg={**MODEL_CFG, "params": {**PARAMS, "max_iter": 5}})[0]
    assert "학습 자료와 설정" in problems(model_cfg={**MODEL_CFG, "version": "hgb-v2"})[0]


def test_evaluation_gates():
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
    assert "아키텍처" in problems(lambda_cfg={**LAMBDA, "architecture": "arm64"})[0]
    assert "아키텍처" in problems(lambda_cfg={**LAMBDA, "runtime": "python3.13"})[0]
