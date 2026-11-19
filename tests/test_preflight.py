from bikeml.cloud import library_mismatches, preflight_problems

SHA = "a" * 64
APP = {"lambda_function.py": "1", "bikeml/service.py": "2"}
CRIT = {"must_beat_all_baselines": True, "min_mae_reduction_vs_hour_baseline": 0.3}
LAMBDA = {"runtime": "python3.12", "architecture": "x86_64"}
GOOD = dict(
    model_meta={"model_sha256": SHA, "library_versions": {"scikit-learn": "1.9.1", "numpy": "2.5.5"}},
    model_file_sha=SHA,
    evaluation={"passed": True, "model": {"sha256": SHA}, "criteria": CRIT, "eval_data": {"sha256": "t"}},
    service_check={"all_passed": True, "model": {"sha256": SHA}, "app_files": APP, "cases_sha256": "c"},
    package_info={"zip_sha256": "z", "app_files": APP, "requirements_sha256": "r", **LAMBDA},
    zip_sha="z",
    app_files=APP,
    requirements_sha="r",
    criteria=CRIT,
    test_sha="t",
    model_mismatch=[],
    split_mismatch=[],
    lambda_cfg=LAMBDA,
    cases_sha="c",
    library_mismatch=[],
)


def problems(**changes):
    return preflight_problems(**{**GOOD, **changes})


def test_all_good():
    assert problems() == []


def test_model_and_data_gates():
    assert "metadata.json" in problems(model_file_sha="b" * 64)[0]
    assert "학습 자료" in problems(model_mismatch=["학습 자료"])[0]
    assert "특성 코드" in problems(model_mismatch=["특성 코드"])[0]
    assert "split" in problems(split_mismatch=["평가 자료 기간"])[0]


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


def test_changed_service_cases_block_deploy():
    assert "검사 사례" in problems(cases_sha="other")[0]


def test_library_versions_must_match_lambda_pins():
    meta = GOOD["model_meta"]
    assert library_mismatches(meta, "numpy==2.5.5\nscikit-learn==1.9.1\nscipy==1.18.1\n") == []
    found = library_mismatches(meta, "numpy==2.5.4\nScikit-Learn == 1.9.1\n")
    assert found == ["numpy 2.5.5, requirements-lambda.txt 2.5.4"]
    assert "라이브러리 버전" in problems(library_mismatch=found)[0]
    assert "라이브러리 버전이 없음" in problems(model_meta={"model_sha256": SHA})[0]
