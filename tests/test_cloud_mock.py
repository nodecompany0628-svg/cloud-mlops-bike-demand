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
                        lambda cfg, lcfg: {"meta": meta, "zip_sha256": sha256_file(pkg), "zip_path": pkg})
    monkeypatch.setattr(cloud, "REGIONS_STATE", tmp_path / "state" / "regions.json")
    monkeypatch.setattr(cloud, "RESULTS", out / "results")
    shutil.copytree(ROOT / "results" / "evaluation", out / "results" / "evaluation")

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


def run(steps, **kw):
    return runlog.run_steps(steps, STEPS, load_config(), make_args(steps=steps, **kw), DEFAULT_CONFIG)


def regional(resources, region="ap-northeast-2"):
    return resources["regions"][region]


def test_cloud_steps_end_to_end(sandbox):
    steps = ["deploy", "deploy", "smoke", "cloud-eval", "release", "status", "cleanup"]
    with mock_aws():
        rc = run(steps)
    recs = records(sandbox)
    assert rc == 0, [r.get("error") for r in recs]
    assert [r["status"] for r in recs] == ["success"] * len(steps)

    deploy = read(sandbox, "results/cloud/deploy.json")
    assert recs[0]["cloud"]["function_version"] == "1"
    assert deploy["function"]["action"] == "updated" and deploy["function"]["version"].isdigit()
    assert f"/models/hgb-v1/{deploy['env']['MODEL_SHA256'][:12]}/model.joblib?versionId=" in deploy["env"]["MODEL_URI"]
    assert deploy["model_uri_override"] is False
    assert deploy["cloud"]["bucket"].startswith("bike-demand-mlops-ap-northeast-2-")
    assert deploy["cloud"]["caller_type"] == "user" and "caller" not in deploy["cloud"]
    assert recs[1]["cloud"]["bucket"] == recs[0]["cloud"]["bucket"]
    assert deploy["model_upload"]["uploaded"] is False and deploy["package_upload"]["uploaded"] is False

    smoke = read(sandbox, "results/service_check/cloud.json")
    assert smoke["all_passed"] and smoke["total"] == len(load_cases()) and "alias" not in smoke
    assert smoke["function_version"] == recs[1]["cloud"]["function_version"]

    release = read(sandbox, "results/cloud/release.json")
    assert release["alias"]["alias"] == "live" and release["function_version"] == smoke["function_version"]
    assert release["previous_live"] is None and release["identity"] == smoke["identity"]
    assert recs[4]["model"]["sha256"] == deploy["env"]["MODEL_SHA256"]

    ev = read(sandbox, "results/cloud/cloud_eval.json")
    assert ev["identical_to_local"] and ev["rows"] == 2888 and ev["batches"] == 6
    assert ev["metrics_cloud"]["mae"] == ev["metrics_local"]["mae"]

    status = read(sandbox, "results/cloud/status.json")
    assert status["resources"]["iam_role"]["exists"] is True
    assert all(v["exists"] is True for v in regional(status["resources"]).values())
    assert regional(status["resources"])["lambda_function"]["alias_live"] == release["function_version"]
    assert status["other_account_regions"] == []

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


def test_cleanup_also_covers_region_used_by_earlier_deploy(sandbox):
    with mock_aws():
        assert run(["deploy"], region="us-west-2") == 0
        rc = run(["status", "cleanup"])
        west = boto3.client("lambda", region_name="us-west-2").list_functions()["Functions"]
        buckets = boto3.client("s3", region_name="us-west-2").list_buckets()["Buckets"]
    assert rc == 0
    status = read(sandbox, "results/cloud/status.json")
    assert sorted(status["regions_checked"]) == ["ap-northeast-2", "us-west-2"]
    assert regional(status["resources"], "us-west-2")["lambda_function"]["exists"] is True
    clean = read(sandbox, "results/cloud/cleanup.json")
    assert clean["all_gone"] and west == [] and buckets == []


def test_role_without_project_tags_is_left_alone(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    with mock_aws():
        iam = boto3.client("iam", region_name="ap-northeast-2")
        iam.create_role(RoleName="bike-demand-predict-role", AssumeRolePolicyDocument=json.dumps(cloud.LAMBDA_TRUST))
        rc_deploy = run(["deploy"])
        rc_clean = run(["cleanup"])
        roles = [r["RoleName"] for r in iam.list_roles()["Roles"]]
    recs = records(sandbox)
    assert rc_deploy == 1 and "IAM 역할" in recs[0]["error"]
    assert rc_clean == 1 and "bike-demand-predict-role" in roles


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
    assert report["all_gone"] is False and regional(report["after"])["s3_bucket"]["exists"] is None
    role_action = next(x for x in report["actions"] if x["resource"].startswith("iam:"))
    assert role_action["result"].startswith("kept")


def test_deploy_stops_before_aws_when_preflight_fails(sandbox, monkeypatch):
    def refuse(cfg, lcfg):
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


def test_env_credentials_opt_in_requires_env_keys(sandbox, monkeypatch):
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(k)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(sandbox / "none-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(sandbox / "none-config"))
    rc = run(["status"])
    recs = records(sandbox)
    assert rc == 1 and "AWS_ACCESS_KEY_ID" in recs[0]["error"]


def test_release_requires_cloud_eval(sandbox):
    with mock_aws():
        rc = run(["deploy", "smoke", "release"])
        alias = None
        lam = boto3.client("lambda", region_name="ap-northeast-2")
        try:
            alias = lam.get_alias(FunctionName="bike-demand-predict", Name="live")
        except lam.exceptions.ResourceNotFoundException:
            pass
    recs = records(sandbox)
    assert rc == 1 and [r["status"] for r in recs] == ["success", "success", "failed"]
    assert "cloud-eval" in recs[-1]["error"] and alias is None


def test_empty_untagged_bucket_with_our_name_is_adopted(sandbox):
    name = "bike-demand-mlops-ap-northeast-2-0a1b2c3d"
    with mock_aws():
        s3 = boto3.client("s3", region_name="ap-northeast-2")
        s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": "ap-northeast-2"})
        assert run(["deploy"]) == 0
        deploy = read(sandbox, "results/cloud/deploy.json")
        assert run(["cleanup"]) == 0
        left = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    assert deploy["bucket"] == "adopted" and deploy["cloud"]["bucket"] == name and left == []


def test_cleanup_reports_deployments_from_other_account(sandbox):
    cloud.write_json(cloud.REGIONS_STATE, [{"account": cloud.account_key("999999999999"), "region": "eu-west-1"}])
    with mock_aws():
        rc = run(["cleanup"])
    recs = records(sandbox)
    assert rc == 1 and "다른 계정" in recs[-1]["error"]
    assert cloud.deployments() == [{"account": cloud.account_key("999999999999"), "region": "eu-west-1"}]


def test_status_name_is_validated_before_aws(sandbox):
    rc = run(["status"], status_name="../../x")
    recs = records(sandbox)
    assert rc == 1 and "--status-name" in recs[0]["error"]


def test_release_rejects_recreated_function_with_same_version_number(sandbox):
    with mock_aws():
        assert run(["deploy", "smoke", "cloud-eval", "release", "cleanup"]) == 0
        args = make_args(steps=["deploy"], model_uri_override="s3://x/models/hgb-v1/model.pkl")
        assert runlog.run_steps(["deploy"], STEPS, load_config(), args, DEFAULT_CONFIG) == 0
        rc = run(["release"])
        lam = boto3.client("lambda", region_name="ap-northeast-2")
        try:
            alias = lam.get_alias(FunctionName="bike-demand-predict", Name="live")
        except lam.exceptions.ResourceNotFoundException:
            alias = None
    recs = records(sandbox)
    assert rc == 1 and "통과한 기록이 없습니다" in recs[-1]["error"] and alias is None


def test_release_rejects_untested_region(sandbox):
    with mock_aws():
        assert run(["deploy", "smoke", "cloud-eval"]) == 0
        assert run(["deploy"], region="us-west-2") == 0
        rc = run(["release"], region="us-west-2")
    recs = records(sandbox)
    assert rc == 1 and "us-west-2" in recs[-1]["error"]


def test_empty_tags_are_refused(sandbox, monkeypatch):
    cfg = load_config()
    cfg["cloud"]["tags"] = {}
    with mock_aws():
        rc = runlog.run_steps(["status"], STEPS, cfg, make_args(steps=["status"]), DEFAULT_CONFIG)
    assert rc == 1 and "cloud.tags" in records(sandbox)[-1]["error"]


def test_two_buckets_block_deploy_but_cleanup_removes_both(sandbox):
    with mock_aws():
        s3 = boto3.client("s3", region_name="ap-northeast-2")
        for name in ("bike-demand-mlops-ap-northeast-2-00000001", "bike-demand-mlops-ap-northeast-2-00000002"):
            s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": "ap-northeast-2"})
            s3.put_bucket_tagging(Bucket=name, Tagging={"TagSet": [
                {"Key": k, "Value": v} for k, v in load_config()["cloud"]["tags"].items()]})
        rc_deploy = run(["deploy"])
        rc_clean = run(["cleanup"])
        left = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    recs = records(sandbox)
    assert rc_deploy == 1 and "버킷이 여러 개" in recs[0]["error"]
    assert rc_clean == 0 and left == []
