import argparse
import json
import shutil
import zipfile

import boto3
import pytest
from botocore.exceptions import ClientError

try:
    from moto import mock_aws
except ImportError:
    pytest.skip("moto 가 설치되지 않음 (pip install -r requirements-dev.txt)", allow_module_level=True)

from bikeml import cloud, runlog
from bikeml.cli import STEPS
from bikeml.common import (
    DEFAULT_CONFIG,
    ROOT,
    load_config,
    load_json,
    model_dir,
    sha256_file,
)
from bikeml.servicecheck import LocalInvoker, load_cases

MOTO_ACCOUNT = "123456789012"
NEEDED = [model_dir(load_config()) / "model.joblib", ROOT / "results/evaluation/predictions_test.csv",
          ROOT / "data/processed/test.csv"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDED),
                                reason="로컬 단계 결과가 없음 (python -m bikeml run local)")


def make_args(**kw):
    base = dict(config=str(DEFAULT_CONFIG), profile=None, region=None, lambda_config=None,
                model_uri_override=None, status_name=None, cmd="run", steps=[])
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for k, v in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                 "AWS_SESSION_TOKEN": "testing", "AWS_DEFAULT_REGION": "ap-northeast-2"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("BIKEML_USE_ENV_CREDENTIALS", "1")
    for k in ("MODEL_URI", "MODEL_VERSION", "MODEL_SHA256"):
        monkeypatch.setenv(k, "")

    monkeypatch.setattr(runlog, "RUN_LOG", tmp_path / "runs" / "run_log.jsonl")
    monkeypatch.setattr(runlog, "STAGING", tmp_path / "staging")
    monkeypatch.setattr(runlog, "FAILED", tmp_path / "failed")
    out = tmp_path / "promoted"

    def promote(stage):
        for src in stage.rglob("*"):
            if src.is_file():
                dst = out / src.relative_to(stage)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    monkeypatch.setattr(runlog, "_promote", promote)

    pkg = tmp_path / "lambda_package.zip"
    with zipfile.ZipFile(pkg, "w") as z:
        z.writestr("lambda_function.py", "def lambda_handler(event, context):\n    return {}\n")
    meta = load_json(model_dir(load_config()) / "metadata.json")
    monkeypatch.setattr(cloud, "deploy_preflight",
                        lambda cfg: {"meta": meta, "zip_sha256": sha256_file(pkg), "zip_path": pkg})

    local = LocalInvoker(load_config())

    def fake_invoke(self, payload, qualifier):
        resp, m = local(payload)
        return resp, {**m, "function_version": qualifier}
    monkeypatch.setattr(cloud.Cloud, "invoke", fake_invoke)
    return tmp_path


def records(tmp_path):
    path = tmp_path / "runs" / "run_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read(tmp_path, rel_path):
    return json.loads((tmp_path / "promoted" / rel_path).read_text(encoding="utf-8"))


def run(steps):
    return runlog.run_steps(steps, STEPS, load_config(), make_args(steps=steps), DEFAULT_CONFIG)


def test_cloud_steps_end_to_end(sandbox):
    steps = ["deploy", "deploy", "smoke", "cloud-eval", "status", "cleanup"]
    with mock_aws():
        rc = run(steps)
    recs = records(sandbox)
    assert rc == 0, [r.get("error") for r in recs]
    assert [r["status"] for r in recs] == ["success"] * len(steps)

    deploy = read(sandbox, "results/cloud/deploy.json")
    assert recs[0]["cloud"]["function_version"] == "1"
    assert deploy["function"]["action"] == "updated" and deploy["function"]["version"].isdigit()
    assert "/models/hgb-v1/model.joblib?versionId=" in deploy["env"]["MODEL_URI"]
    assert deploy["model_uri_override"] is False
    assert deploy["cloud"]["bucket"].startswith("bike-demand-mlops-ap-northeast-2-")
    assert deploy["cloud"]["caller_type"] == "user" and "caller" not in deploy["cloud"]
    assert recs[1]["cloud"]["bucket"] == recs[0]["cloud"]["bucket"]
    assert deploy["model_upload"]["uploaded"] is False and deploy["package_upload"]["uploaded"] is False

    smoke = read(sandbox, "results/service_check/cloud.json")
    assert smoke["all_passed"] and smoke["total"] == len(load_cases())
    assert smoke["alias"]["alias"] == "live" and smoke["alias"]["version"] == recs[2]["cloud"]["function_version"]

    ev = read(sandbox, "results/cloud/cloud_eval.json")
    assert ev["identical_to_local"] and ev["rows"] == 2888 and ev["batches"] == 6
    assert ev["metrics_cloud"]["mae"] == ev["metrics_local"]["mae"]

    status = read(sandbox, "results/cloud/status.json")
    assert all(v["exists"] is True for v in status["resources"].values())
    assert status["resources"]["lambda_function"]["alias_live"] == smoke["alias"]["version"]

    clean = read(sandbox, "results/cloud/cleanup.json")
    assert clean["all_gone"] and all(a["result"] == "deleted" for a in clean["actions"])

    texts = [(sandbox / "runs" / "run_log.jsonl").read_text(encoding="utf-8")]
    texts += [p.read_text(encoding="utf-8") for p in (sandbox / "promoted").rglob("*.json")]
    for t in texts:
        assert MOTO_ACCOUNT not in t and "user/moto" not in t
        assert str(ROOT) not in t and str(ROOT).replace("\\", "\\\\") not in t


def test_bucket_without_project_tags_is_left_alone(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    foreign = "bike-demand-mlops-ap-northeast-2-ffffffff"
    with mock_aws():
        s3 = boto3.client("s3", region_name="ap-northeast-2")
        s3.create_bucket(Bucket=foreign, CreateBucketConfiguration={"LocationConstraint": "ap-northeast-2"})
        s3.put_object(Bucket=foreign, Key="keep.txt", Body=b"x")
        rc_deploy = run(["deploy"])
        rc_clean = run(["cleanup"])
        keys = [o["Key"] for o in s3.list_objects_v2(Bucket=foreign).get("Contents", [])]
    recs = records(sandbox)
    assert rc_deploy == 1 and "태그" in recs[0]["error"]
    assert rc_clean == 1 and keys == ["keep.txt"]


def test_status_and_cleanup_refuse_other_region(sandbox, monkeypatch):
    fake_root = sandbox / "root"
    (fake_root / "results" / "cloud").mkdir(parents=True)
    (fake_root / "results" / "cloud" / "deploy.json").write_text(
        json.dumps({"cloud": {"region": "us-west-2"}}), encoding="utf-8")
    monkeypatch.setattr(cloud, "ROOT", fake_root)
    with mock_aws():
        rc = run(["status", "cleanup"])
    recs = records(sandbox)
    assert rc == 1 and [r["status"] for r in recs] == ["failed", "skipped"]
    assert "us-west-2" in recs[0]["error"]


def test_override_deploy_is_recorded_separately(sandbox):
    with mock_aws():
        assert run(["deploy"]) == 0
        args = make_args(steps=["deploy"], model_uri_override="s3://bucket/models/hgb-v1/model.pkl")
        rc = runlog.run_steps(["deploy"], STEPS, load_config(), args, DEFAULT_CONFIG)
    assert rc == 0
    normal = read(sandbox, "results/cloud/deploy.json")
    broken = read(sandbox, "results/cloud/deploy_override.json")
    assert normal["model_uri_override"] is False and "?versionId=" in normal["env"]["MODEL_URI"]
    assert broken["model_uri_override"] is True and broken["env"]["MODEL_URI"].endswith("model.pkl")
    assert records(sandbox)[-1]["summary"]["model_uri_override"] is True


def test_cleanup_when_nothing_exists(sandbox):
    with mock_aws():
        rc = run(["cleanup"])
    assert rc == 0
    clean = read(sandbox, "results/cloud/cleanup.json")
    assert clean["all_gone"] and {a["result"] for a in clean["actions"]} == {"not_found"}


def test_cleanup_fails_when_bucket_state_is_unknown(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    with mock_aws():
        assert run(["deploy"]) == 0
        real_client = boto3.Session.client

        def client(self, name, *a, **kw):
            c = real_client(self, name, *a, **kw)
            if name == "s3":
                def denied(**kwargs):
                    raise ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadBucket")
                c.head_bucket = denied
            return c
        monkeypatch.setattr(boto3.Session, "client", client)
        rc = run(["cleanup"])
    recs = records(sandbox)
    assert rc == 1 and recs[-1]["status"] == "failed"
    failed = sandbox / "failed" / f"{recs[-1]['run_id']}_cleanup" / "results" / "cloud" / "cleanup.json"
    report = json.loads(failed.read_text(encoding="utf-8"))
    assert report["all_gone"] is False and report["after"]["s3_bucket"]["exists"] is None


def test_deploy_stops_before_aws_when_preflight_fails(sandbox, monkeypatch):
    def refuse(cfg):
        raise RuntimeError("배포 전 확인 실패: 이 모델로 evaluate 를 통과한 기록이 없음")
    monkeypatch.setattr(cloud, "deploy_preflight", refuse)
    with mock_aws():
        rc = run(["deploy", "smoke"])
        buckets = boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]
    recs = records(sandbox)
    assert rc == 1 and [r["status"] for r in recs] == ["failed", "skipped"]
    assert buckets == []


def test_cloud_step_requires_explicit_profile(sandbox, monkeypatch):
    monkeypatch.delenv("BIKEML_USE_ENV_CREDENTIALS")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(sandbox / "none-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(sandbox / "none-config"))
    rc = run(["status", "cleanup"])
    recs = records(sandbox)
    assert rc == 1
    assert [r["status"] for r in recs] == ["failed", "skipped"]
    assert "AWS 프로필" in recs[0]["error"]
    assert (sandbox / "failed" / f"{recs[0]['run_id']}_status" / "error.txt").exists()
