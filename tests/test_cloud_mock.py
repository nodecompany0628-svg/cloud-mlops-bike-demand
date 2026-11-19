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


def failed_result(tmp_path, record, rel_path):
    folder = tmp_path / "failed" / f"{record['run_id']}_{record['step']}"
    return json.loads((folder / rel_path).read_text(encoding="utf-8"))


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
    assert status["resources"]["iam_role"]["exists"] is True and status["regions_unreachable"] == []
    assert status["resources"]["iam_role"]["permissions_boundary"].endswith(":policy/bike-demand-predict-boundary")
    assert status["permissions_boundary"] == {"exists": True, "created_by_this_tool": True, "matches_config": True}
    assert status["run_log_unreadable_lines"] == 0
    assert deploy["permissions_boundary"] in ("created", "exists")
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
    assert rc_clean == 0 and keys == ["keep.txt"]
    assert read(sandbox, "results/cloud/cleanup.json")["not_ours"] == ["ap-northeast-2:s3_bucket"]


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
        created = boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]
        created += iam.list_policies(Scope="Local")["Policies"]
        sleeps = []
        monkeypatch.setattr(cloud.time, "sleep", sleeps.append)
        rc_clean = run(["cleanup"])
        roles = [r["RoleName"] for r in iam.list_roles()["Roles"]]
    recs = records(sandbox)
    assert rc_deploy == 1 and "IAM 역할" in recs[0]["error"] and created == []
    assert rc_clean == 0 and "bike-demand-predict-role" in roles and sleeps == []
    assert read(sandbox, "results/cloud/cleanup.json")["not_ours"] == ["iam_role"]


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
        status_rc = run(["status"])
        rc = run(["cleanup"])
    recs = records(sandbox)
    assert status_rc == 1 and rc == 1 and recs[-1]["status"] == "failed"
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
    rc_overwrite = run(["status"], status_name="deploy")
    recs = records(sandbox)
    assert rc == 1 and rc_overwrite == 1 and "--status-name" in recs[0]["error"] and "--status-name" in recs[1]["error"]


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
    assert rc == 1 and "마지막 실행이 통과가 아닙니다" in recs[-1]["error"] and alias is None


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

    def no_session(*a, **kw):
        raise AssertionError("AWS 세션을 만들면 안 됨")
    monkeypatch.setattr(cloud, "make_session", no_session)
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
        iam = boto3.client("iam", region_name="ap-northeast-2")
        iam_after_refusal = iam.list_roles()["Roles"] + iam.list_policies(Scope="Local")["Policies"]
        rc_clean = run(["cleanup"])
        left = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    recs = records(sandbox)
    assert rc_deploy == 1 and "버킷이 여러 개" in recs[0]["error"] and iam_after_refusal == []
    assert rc_clean == 0 and left == []


def test_release_refuses_when_latest_smoke_failed(sandbox, monkeypatch):
    with mock_aws():
        assert run(["deploy", "smoke", "cloud-eval"]) == 0

        def broken(self, payload, qualifier):
            return {"statusCode": 500, "body": {"ok": False, "error": {"code": "MODEL_LOAD_FAILED"}}}, {}
        monkeypatch.setattr(cloud.Cloud, "invoke", broken)
        assert run(["smoke"]) == 1
        rc = run(["release"])
    recs = records(sandbox)
    assert rc == 1 and "smoke" in recs[-1]["error"]


def test_profile_name_is_not_recorded(sandbox):
    args = make_args(steps=["status"], profile="example-profile")
    with mock_aws():
        runlog.run_steps(["status"], STEPS, load_config(), args, DEFAULT_CONFIG)
    text = (sandbox / "runs" / "run_log.jsonl").read_text(encoding="utf-8")
    assert "example-profile" not in text and "<PROFILE>" in text


def test_cleanup_finds_other_region_from_bucket_names_alone(sandbox):
    with mock_aws():
        assert run(["deploy"], region="us-west-2") == 0
        assert run(["deploy"]) == 0
        cloud.REGIONS_STATE.unlink()
        runlog.RUN_LOG.unlink()
        rc = run(["cleanup"])
        west = boto3.client("lambda", region_name="us-west-2").list_functions()["Functions"]
        buckets = boto3.client("s3", region_name="us-west-2").list_buckets()["Buckets"]
    clean = read(sandbox, "results/cloud/cleanup.json")
    assert rc == 0 and clean["all_gone"] and "us-west-2" in clean["regions_checked"]
    assert west == [] and buckets == []


def test_configured_bucket_in_other_region_is_refused(sandbox):
    cfg = load_config()
    name = "bike-demand-mlops-us-west-2-0f0f0f0f"
    cfg["cloud"]["bucket"] = name
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        rc = runlog.run_steps(["deploy"], STEPS, cfg, make_args(steps=["deploy"]), DEFAULT_CONFIG)
    assert rc == 1 and "리전" in records(sandbox)[-1]["error"]


def test_role_policy_covers_all_project_buckets(sandbox):
    with mock_aws():
        assert run(["deploy"]) == 0
        iam = boto3.client("iam", region_name="ap-northeast-2")
        doc = iam.get_role_policy(RoleName="bike-demand-predict-role", PolicyName="bike-demand-predict-ap-northeast-2")
    stmt = next(s for s in doc["PolicyDocument"]["Statement"] if s["Sid"] == "ReadModels")
    assert stmt["Resource"] == "arn:aws:s3:::bike-demand-mlops-*/models/*"


def test_cleanup_fails_and_keeps_role_when_own_region_unreachable(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    real_init = cloud.Cloud.__init__
    with mock_aws():
        assert run(["deploy"], region="us-west-2") == 0

        def init(self, cfg, args, ctx=None, region=None):
            if region == "us-west-2":
                raise RuntimeError("region not enabled")
            real_init(self, cfg, args, ctx, region)
        monkeypatch.setattr(cloud.Cloud, "__init__", init)
        rc = run(["cleanup"])
        roles = [r["RoleName"] for r in boto3.client("iam", region_name="ap-northeast-2").list_roles()["Roles"]]
    recs = records(sandbox)
    assert rc == 1 and "us-west-2" in recs[-1]["error"] and "bike-demand-predict-role" in roles


def test_blank_run_log_line_is_tolerated_but_broken_line_blocks_release(sandbox):
    with mock_aws():
        assert run(["deploy", "smoke", "cloud-eval"]) == 0
        with open(runlog.RUN_LOG, "a", encoding="utf-8") as f:
            f.write("\n")
        assert run(["status"]) == 0
        with open(runlog.RUN_LOG, "a", encoding="utf-8") as f:
            f.write('{"run_id": "broken"\n')
        rc = run(["release"])
    assert rc == 1 and "읽을 수 없는 줄" in records_tolerant(sandbox)[-1]["error"]


def records_tolerant(tmp_path):
    out = []
    for line in (tmp_path / "runs" / "run_log.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def test_cleanup_finds_custom_bucket_after_it_is_removed_from_config(sandbox):
    cfg = load_config()
    cfg["cloud"]["bucket"] = "bike-demand-mlops-mine"
    with mock_aws():
        assert runlog.run_steps(["deploy"], STEPS, cfg, make_args(steps=["deploy"]), DEFAULT_CONFIG) == 0
        rc = run(["cleanup"])
        left = [b["Name"] for b in boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]]
    assert rc == 0 and left == []


def test_region_not_enabled_is_skipped(sandbox, monkeypatch):
    real_init = cloud.Cloud.__init__
    cloud.write_json(cloud.REGIONS_STATE, [])
    runlog.RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(runlog.RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "x", "step": "status", "status": "success", "cloud": {"region": "me-central-1"}}) + "\n")

    def init(self, cfg, args, ctx=None, region=None):
        if region == "me-central-1":
            raise ClientError({"Error": {"Code": "InvalidClientTokenId", "Message": "region disabled"}}, "GetCallerIdentity")
        real_init(self, cfg, args, ctx, region)
    with mock_aws():
        monkeypatch.setattr(cloud.Cloud, "__init__", init)
        rc = run(["cleanup"])
    clean = read(sandbox, "results/cloud/cleanup.json")
    assert rc == 0 and clean["regions_unreachable"][0]["blocking"] is False


def test_bucket_from_failed_deploy_is_still_cleaned(sandbox, monkeypatch):
    cfg = load_config()
    cfg["cloud"]["bucket"] = "bike-demand-mlops-mine"

    def boom(self, *a, **kw):
        raise RuntimeError("deploy_function failed")
    with mock_aws():
        monkeypatch.setattr(cloud.Cloud, "deploy_function", boom)
        assert runlog.run_steps(["deploy"], STEPS, cfg, make_args(steps=["deploy"]), DEFAULT_CONFIG) == 1
        assert records(sandbox)[-1]["cloud"]["bucket"] == "bike-demand-mlops-mine"
        rc = run(["cleanup"])
        left = [b["Name"] for b in boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]]
    assert rc == 0 and left == []


CURRENT_NAMES = {"function": "bike-demand-predict", "role": "bike-demand-predict-role",
                 "permissions_boundary": "bike-demand-predict-boundary", "bucket_prefix": "bike-demand-mlops"}


def write_deploy_record(cloud_fields):
    runlog.RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(runlog.RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "x", "step": "deploy", "status": "failed", "cloud": {
            "region": "ap-northeast-2", **CURRENT_NAMES, "resources_touched": True, **cloud_fields}}) + "\n")


def test_recorded_bucket_name_that_is_not_valid_is_ignored(sandbox):
    write_deploy_record({"bucket": "bike-demand-mlops-<ACCOUNT_ID>", "creating_bucket": []})
    with mock_aws():
        assert run(["status"]) == 0


def test_recorded_bucket_of_another_account_is_skipped(sandbox, monkeypatch):
    write_deploy_record({"bucket": "bike-demand-mlops-theirs"})

    def region(self, name):
        raise AssertionError("이 계정에 없는 버킷의 위치를 조회하면 안 됨")
    with mock_aws():
        monkeypatch.setattr(cloud.Cloud, "bucket_region", region)
        assert run(["cleanup"]) == 0


def test_bucket_name_that_only_looks_like_a_region_is_ignored(sandbox):
    with mock_aws():
        s3 = boto3.client("s3", region_name="ap-northeast-2")
        s3.create_bucket(Bucket="bike-demand-mlops-my-test-1-0a1b2c3d",
                         CreateBucketConfiguration={"LocationConstraint": "ap-northeast-2"})
        assert run(["status"]) == 0
    status = read(sandbox, "results/cloud/status.json")
    assert "my-test-1" not in status["regions_checked"] and status["regions_unreachable"] == []


def test_boundary_document_matches_docs():
    c = load_config()["cloud"]
    on_disk = json.loads((ROOT / "docs" / "iam" / "lambda-boundary.json").read_text(encoding="utf-8"))
    assert on_disk == cloud.boundary_document(c["bucket_prefix"], c["function_name"])


def test_existing_role_without_boundary_is_refused(sandbox):
    with mock_aws():
        iam = boto3.client("iam", region_name="ap-northeast-2")
        iam.create_role(RoleName="bike-demand-predict-role", AssumeRolePolicyDocument=json.dumps(cloud.LAMBDA_TRUST),
                        Tags=[{"Key": k, "Value": v} for k, v in load_config()["cloud"]["tags"].items()])
        rc = run(["deploy"])
        policies = iam.list_policies(Scope="Local")["Policies"]
        buckets = boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]
    assert rc == 1 and "권한 경계" in records(sandbox)[-1]["error"] and policies == [] and buckets == []


def test_configured_bucket_left_untagged_is_adopted(sandbox):
    cfg = load_config()
    cfg["cloud"]["bucket"] = "bike-demand-mlops-mine"
    write_deploy_record({"bucket": "bike-demand-mlops-mine", "creating_bucket": "bike-demand-mlops-mine"})
    with mock_aws():
        boto3.client("s3", region_name="ap-northeast-2").create_bucket(
            Bucket="bike-demand-mlops-mine", CreateBucketConfiguration={"LocationConstraint": "ap-northeast-2"})
        rc = runlog.run_steps(["deploy"], STEPS, cfg, make_args(steps=["deploy"]), DEFAULT_CONFIG)
    assert rc == 0 and read(sandbox, "results/cloud/deploy.json")["bucket"] == "adopted"


def test_smoke_waits_for_model_permission(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "READY_WAIT_S", 0)
    calls = {"n": 0}
    local_invoke = cloud.Cloud.invoke

    def flaky(self, payload, qualifier):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"statusCode": 500, "body": {"ok": False, "error": {
                "code": "MODEL_LOAD_FAILED", "message": "An error occurred (403) when calling HeadObject: Forbidden"}}}, {}
        return local_invoke(self, payload, qualifier)
    with mock_aws():
        monkeypatch.setattr(cloud.Cloud, "invoke", flaky)
        assert run(["deploy", "smoke"]) == 0
        smoke = read(sandbox, "results/service_check/cloud.json")
        assert smoke["readiness"]["attempts"] == 2 and smoke["readiness"]["waited_for_permissions"] is True
        calls["n"] = 0
        assert run(["smoke"]) == 1
    assert read(sandbox, "results/service_check/cloud.json")["readiness"]["attempts"] == 2


def test_status_fails_when_own_region_unreachable(sandbox, monkeypatch):
    real_init = cloud.Cloud.__init__
    with mock_aws():
        assert run(["deploy"], region="us-west-2") == 0

        def init(self, cfg, args, ctx=None, region=None):
            if region == "us-west-2":
                raise RuntimeError("timeout")
            real_init(self, cfg, args, ctx, region)
        monkeypatch.setattr(cloud.Cloud, "__init__", init)
        rc = run(["status"])
    recs = records(sandbox)
    assert rc == 1 and "확인하지 못했습니다" in recs[-1]["error"]


def test_untagged_configured_bucket_without_record_is_refused(sandbox):
    cfg = load_config()
    cfg["cloud"]["bucket"] = "bike-demand-mlops-users-own"
    with mock_aws():
        s3 = boto3.client("s3", region_name="ap-northeast-2")
        s3.create_bucket(Bucket="bike-demand-mlops-users-own",
                         CreateBucketConfiguration={"LocationConstraint": "ap-northeast-2"})
        rc = runlog.run_steps(["deploy"], STEPS, cfg, make_args(steps=["deploy"]), DEFAULT_CONFIG)
        left = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    assert rc == 1 and "태그" in records(sandbox)[-1]["error"] and left == ["bike-demand-mlops-users-own"]


def test_cloud_eval_records_calls_when_invoke_raises(sandbox, monkeypatch):
    with mock_aws():
        assert run(["deploy"]) == 0

        def boom(self, payload, qualifier):
            raise RuntimeError("read timeout")
        monkeypatch.setattr(cloud.Cloud, "invoke", boom)
        assert run(["cloud-eval"]) == 1
    rid = records(sandbox)[-1]["run_id"]
    calls = json.loads((sandbox / "failed" / f"{rid}_cloud-eval" / "results" / "cloud" / "cloud_eval_calls.json")
                       .read_text(encoding="utf-8"))
    assert calls[0]["status"] is None and "read timeout" in calls[0]["error"]


def test_boundary_kept_when_operator_cannot_delete_policies(sandbox, monkeypatch):
    with mock_aws():
        assert run(["deploy"]) == 0
        real_client = boto3.Session.client

        def client(self, name, *a, **kw):
            c = real_client(self, name, *a, **kw)
            if name == "iam":
                def denied(**kwargs):
                    raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "DeletePolicy")
                c.delete_policy = c.list_policy_versions = c.delete_policy_version = denied
            return c
        monkeypatch.setattr(boto3.Session, "client", client)
        rc = run(["cleanup"])
    clean = read(sandbox, "results/cloud/cleanup.json")
    action = next(a for a in clean["actions"] if a["resource"].startswith("iam-policy:"))
    assert rc == 0 and clean["all_gone"] and action["result"].startswith("kept")


def test_own_region_with_auth_error_still_blocks_cleanup(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    real_init = cloud.Cloud.__init__
    with mock_aws():
        assert run(["deploy"], region="us-west-2") == 0

        def init(self, cfg, args, ctx=None, region=None):
            if region == "us-west-2":
                raise ClientError({"Error": {"Code": "InvalidClientTokenId", "Message": "region disabled"}},
                                  "GetCallerIdentity")
            real_init(self, cfg, args, ctx, region)
        monkeypatch.setattr(cloud.Cloud, "__init__", init)
        status_rc = run(["status"])
        rc = run(["cleanup"])
        roles = [r["RoleName"] for r in boto3.client("iam", region_name="ap-northeast-2").list_roles()["Roles"]]
        functions = boto3.client("lambda", region_name="us-west-2").list_functions()["Functions"]
    clean = failed_result(sandbox, records(sandbox)[-1], "results/cloud/cleanup.json")
    assert status_rc == 1 and rc == 1 and not clean["all_gone"]
    assert clean["regions_unreachable"][0]["blocking"] is True and "us-west-2" in records(sandbox)[-1]["error"]
    assert "bike-demand-predict-role" in roles and len(functions) == 1


def test_broken_run_log_line_fails_status_and_cleanup_but_keeps_role(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    with mock_aws():
        assert run(["deploy"]) == 0
        with open(runlog.RUN_LOG, "a", encoding="utf-8") as f:
            f.write('{"run_id": "broken"\n')
        status_rc = run(["status"])
        rc = run(["cleanup"])
        roles = [r["RoleName"] for r in boto3.client("iam", region_name="ap-northeast-2").list_roles()["Roles"]]
        buckets = boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]
    clean = failed_result(sandbox, records_tolerant(sandbox)[-1], "results/cloud/cleanup.json")
    assert status_rc == 1 and rc == 1 and "읽을 수 없는 줄" in records_tolerant(sandbox)[-1]["error"]
    assert clean["run_log_unreadable_lines"] == 1 and not clean["all_gone"]
    assert "bike-demand-predict-role" in roles and buckets == []


def test_boundary_with_other_content_stops_deploy_before_creating_bucket(sandbox):
    other = cloud.boundary_document("bike-demand-mlops", "other-function")
    with mock_aws():
        boto3.client("iam", region_name="ap-northeast-2").create_policy(
            PolicyName="bike-demand-predict-boundary", PolicyDocument=json.dumps(other))
        rc = run(["deploy"])
        buckets = boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]
        roles = boto3.client("iam", region_name="ap-northeast-2").list_roles()["Roles"]
    assert rc == 1 and "내용이 지금 설정" in records(sandbox)[-1]["error"] and buckets == [] and roles == []
    assert cloud.deployments() == []


def boundary_delete_failing(monkeypatch, codes):
    real_client = boto3.Session.client
    calls = []

    def client(self, name, *a, **kw):
        c = real_client(self, name, *a, **kw)
        if name == "iam":
            real_delete = c.delete_policy

            def delete_policy(**kwargs):
                calls.append(1)
                if len(calls) <= len(codes):
                    raise ClientError({"Error": {"Code": codes[len(calls) - 1], "Message": "x"}}, "DeletePolicy")
                return real_delete(**kwargs)
            c.delete_policy = delete_policy
        return c
    monkeypatch.setattr(boto3.Session, "client", client)
    return calls


def test_boundary_delete_conflict_is_retried(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    with mock_aws():
        assert run(["deploy"]) == 0
        calls = boundary_delete_failing(monkeypatch, ["DeleteConflict"])
        rc = run(["cleanup"])
        policies = boto3.client("iam", region_name="ap-northeast-2").list_policies(Scope="Local")["Policies"]
    assert rc == 0 and len(calls) == 2 and policies == []


def test_cleanup_error_is_not_recorded_as_success(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    with mock_aws():
        assert run(["deploy"]) == 0
        boundary_delete_failing(monkeypatch, ["ServiceFailure"])
        rc = run(["cleanup"])
    rec = records(sandbox)[-1]
    clean = failed_result(sandbox, rec, "results/cloud/cleanup.json")
    assert rc == 1 and "삭제 중 오류" in rec["error"] and clean["errors"][0]["result"] == "error: ServiceFailure"


def test_untagged_log_group_blocks_deploy_before_anything_is_created(sandbox):
    with mock_aws():
        boto3.client("logs", region_name="ap-northeast-2").create_log_group(logGroupName="/aws/lambda/bike-demand-predict")
        rc = run(["deploy"])
        iam = boto3.client("iam", region_name="ap-northeast-2")
        created = (boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]
                   + iam.list_roles()["Roles"] + iam.list_policies(Scope="Local")["Policies"])
    assert rc == 1 and "로그 그룹" in records(sandbox)[-1]["error"] and created == [] and cloud.deployments() == []


def test_boundary_with_extra_versions_is_deleted(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    with mock_aws():
        assert run(["deploy"]) == 0
        iam = boto3.client("iam", region_name="ap-northeast-2")
        arn = next(p["Arn"] for p in iam.list_policies(Scope="Local")["Policies"])
        iam.create_policy_version(PolicyArn=arn, PolicyDocument=json.dumps(
            cloud.boundary_document("bike-demand-mlops", "bike-demand-predict")), SetAsDefault=True)
        real_client = boto3.Session.client

        def client(self, name, *a, **kw):
            c = real_client(self, name, *a, **kw)
            if name == "iam":
                real_delete = c.delete_policy

                def delete_policy(**kwargs):
                    if len(c.list_policy_versions(PolicyArn=kwargs["PolicyArn"])["Versions"]) > 1:
                        raise ClientError({"Error": {"Code": "DeleteConflict", "Message": "versions"}}, "DeletePolicy")
                    return real_delete(**kwargs)
                c.delete_policy = delete_policy
            return c
        monkeypatch.setattr(boto3.Session, "client", client)
        rc = run(["cleanup"])
        policies = iam.list_policies(Scope="Local")["Policies"]
    assert rc == 0 and policies == []


def test_status_fails_when_boundary_state_is_unknown(sandbox, monkeypatch):
    with mock_aws():
        assert run(["deploy"]) == 0
        real_client = boto3.Session.client

        def client(self, name, *a, **kw):
            c = real_client(self, name, *a, **kw)
            if name == "iam":
                def denied(**kwargs):
                    raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "ListPolicyTags")
                c.list_policy_tags = denied
            return c
        monkeypatch.setattr(boto3.Session, "client", client)
        rc = run(["status"])
    status = failed_result(sandbox, records(sandbox)[-1], "results/cloud/status.json")
    assert rc == 1 and status["permissions_boundary"] == {"exists": None, "error": "AccessDenied"}


def test_status_records_unknown_role_state(sandbox, monkeypatch):
    with mock_aws():
        assert run(["deploy"]) == 0
        real_client = boto3.Session.client

        def client(self, name, *a, **kw):
            c = real_client(self, name, *a, **kw)
            if name == "iam":
                def denied(**kwargs):
                    raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetRole")
                c.get_role = denied
            return c
        monkeypatch.setattr(boto3.Session, "client", client)
        rc = run(["status"])
    status = failed_result(sandbox, records(sandbox)[-1], "results/cloud/status.json")
    assert rc == 1 and status["resources"]["iam_role"] == {"exists": None, "error": "AccessDenied"}


def renamed_config():
    cfg = load_config()
    cfg["cloud"]["function_name"] = "bike-demand-predict-2"
    return cfg


def run_with(cfg, step):
    return runlog.run_steps([step], STEPS, cfg, make_args(steps=[step]), DEFAULT_CONFIG)


def test_renamed_config_requires_cleanup_with_old_names(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    renamed = renamed_config()
    with mock_aws():
        assert run(["deploy"]) == 0
        rc_deploy = run_with(renamed, "deploy")
        rc_status = run_with(renamed, "status")
        rc_new = run_with(renamed, "cleanup")
        functions = [f["FunctionName"] for f in boto3.client("lambda", region_name="ap-northeast-2").list_functions()["Functions"]]
        buckets = boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]
        rc_old = run(["cleanup"])
        rc_after = run_with(renamed, "status")
        rc_new_deploy = run_with(renamed, "deploy")
    recs = records(sandbox)
    refused = failed_result(sandbox, recs[3], "results/cloud/cleanup.json")
    assert rc_deploy == 1 and "이전 이름" in recs[1]["error"] and recs[1]["cloud"] is None
    assert rc_status == 1 and rc_new == 1 and "아무것도 지우지 않았습니다" in recs[3]["error"]
    assert refused["refused"] is True and refused["renamed_leftovers"][0]["function"] == "bike-demand-predict"
    assert functions == ["bike-demand-predict"] and len(buckets) == 1
    assert rc_old == 0 and rc_after == 0 and rc_new_deploy == 0


def test_other_account_record_does_not_block_renaming(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    with mock_aws():
        assert run(["deploy"]) == 0
        cloud.write_json(cloud.REGIONS_STATE, cloud.deployments()
                         + [{"account": cloud.account_key("999999999999"), "region": "eu-west-1"}])
        rc_old = run(["cleanup"])
        rc_new = run_with(renamed_config(), "deploy")
    assert rc_old == 1 and "다른 계정" in records(sandbox)[-2]["error"] and rc_new == 0


def test_two_pending_name_sets_can_both_be_cleaned(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    renamed = renamed_config()
    with mock_aws():
        assert run(["deploy"]) == 0
        with open(runlog.RUN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": "y", "step": "deploy", "status": "failed", "cloud": {
                "region": "ap-northeast-2", "function": "bike-demand-predict-2", "role": "bike-demand-predict-role",
                "permissions_boundary": "bike-demand-predict-boundary", "bucket_prefix": "bike-demand-mlops",
                "resources_touched": True}}) + "\n")
        rc_old = run(["cleanup"])
        functions = boto3.client("lambda", region_name="ap-northeast-2").list_functions()["Functions"]
        rc_new = run_with(renamed, "cleanup")
        rc_status = run(["status"])
    assert rc_old == 1 and "이전 이름" in records_tolerant(sandbox)[-3]["error"] and functions == []
    assert rc_new == 0 and rc_status == 0


def test_deploy_refused_at_boundary_check_does_not_block_renaming(sandbox):
    other = cloud.boundary_document("bike-demand-mlops", "other-function")
    with mock_aws():
        boto3.client("iam", region_name="ap-northeast-2").create_policy(
            PolicyName="bike-demand-predict-boundary", PolicyDocument=json.dumps(other))
        assert run(["deploy"]) == 1
        rc = run_with(renamed_config(), "deploy")
    error = records(sandbox)[-1]["error"]
    assert rc == 1 and "내용이 지금 설정" in error and "이전 이름" not in error


def test_deploy_refused_by_checks_does_not_block_renaming(sandbox):
    with mock_aws():
        boto3.client("logs", region_name="ap-northeast-2").create_log_group(logGroupName="/aws/lambda/bike-demand-predict")
        assert run(["deploy"]) == 1
        rc = run_with(renamed_config(), "deploy")
    assert rc == 0, records(sandbox)[-1].get("error")


def test_status_record_does_not_make_user_bucket_adoptable(sandbox):
    cfg = load_config()
    cfg["cloud"]["bucket"] = "bike-demand-mlops-users-own"
    with mock_aws():
        s3 = boto3.client("s3", region_name="ap-northeast-2")
        s3.create_bucket(Bucket="bike-demand-mlops-users-own",
                         CreateBucketConfiguration={"LocationConstraint": "ap-northeast-2"})
        run_with(cfg, "status")
        rc_first = run_with(cfg, "deploy")
        rc_second = run_with(cfg, "deploy")
        run_with(cfg, "cleanup")
        left = [b["Name"] for b in s3.list_buckets()["Buckets"]]
        try:
            tags = s3.get_bucket_tagging(Bucket="bike-demand-mlops-users-own")["TagSet"]
        except ClientError:
            tags = []
    assert rc_first == 1 and rc_second == 1 and left == ["bike-demand-mlops-users-own"] and tags == []


def test_account_id_in_names_is_refused_before_anything_is_created(sandbox):
    cfg = load_config()
    cfg["cloud"]["bucket_prefix"] = f"bike-demand-mlops-{MOTO_ACCOUNT}"
    with mock_aws():
        rc = run_with(cfg, "deploy")
        iam = boto3.client("iam", region_name="ap-northeast-2")
        created = (boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]
                   + iam.list_roles()["Roles"] + iam.list_policies(Scope="Local")["Policies"])
    error = records(sandbox)[-1]["error"]
    assert rc == 1 and "계정 ID" in error and MOTO_ACCOUNT not in error and created == []


def test_deploy_refuses_while_run_log_has_unreadable_lines(sandbox):
    runlog.RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(runlog.RUN_LOG, "a", encoding="utf-8") as f:
        f.write('{"run_id": "cut"\n')
    with mock_aws():
        rc = run(["deploy"])
        buckets = boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]
    assert rc == 1 and "읽을 수 없는 줄" in records_tolerant(sandbox)[-1]["error"] and buckets == []


def test_user_bucket_named_by_refused_deploy_leaves_cleanup_scope(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    cfg = load_config()
    cfg["cloud"]["bucket"] = "bike-demand-mlops-users-own"
    with mock_aws():
        s3 = boto3.client("s3", region_name="ap-northeast-2")
        s3.create_bucket(Bucket="bike-demand-mlops-users-own",
                         CreateBucketConfiguration={"LocationConstraint": "ap-northeast-2"})
        s3.put_object(Bucket="bike-demand-mlops-users-own", Key="mine.txt", Body=b"x")
        run_with(cfg, "status")
        assert run_with(cfg, "deploy") == 1
        assert run(["deploy"]) == 0
        rc = run(["cleanup"])
        left = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    assert rc == 0 and left == ["bike-demand-mlops-users-own"]


def test_own_recorded_bucket_with_unknown_location_blocks_cleanup(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    cfg = load_config()
    cfg["cloud"]["bucket"] = "bike-demand-mlops-mine"
    with mock_aws():
        assert run_with(cfg, "deploy") == 0

        def denied(self, name):
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetBucketLocation")
        monkeypatch.setattr(cloud.Cloud, "bucket_region", denied)
        rc = run(["cleanup"])
        left = [b["Name"] for b in boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]]
        roles = [r["RoleName"] for r in boto3.client("iam", region_name="ap-northeast-2").list_roles()["Roles"]]
    assert rc == 1 and "bike-demand-mlops-mine" in records(sandbox)[-1]["error"]
    assert left == ["bike-demand-mlops-mine"] and roles == ["bike-demand-predict-role"]


def test_release_rejects_same_version_with_other_revision(sandbox, monkeypatch):
    with mock_aws():
        assert run(["deploy", "smoke", "cloud-eval"]) == 0
        real_client = boto3.Session.client

        def client(self, name, *a, **kw):
            c = real_client(self, name, *a, **kw)
            if name == "lambda":
                real_get = c.get_function_configuration
                c.get_function_configuration = lambda **kwargs: {**real_get(**kwargs), "RevisionId": "recreated"}
            return c
        monkeypatch.setattr(boto3.Session, "client", client)
        rc = run(["release"])
    assert rc == 1 and "smoke" in records(sandbox)[-1]["error"]


def test_untagged_bucket_the_tool_created_is_cleaned_after_config_change(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    cfg = load_config()
    cfg["cloud"]["bucket"] = "bike-demand-mlops-mine"

    def interrupted(self):
        raise RuntimeError("tag_bucket 직전에 멈춤")
    with mock_aws():
        monkeypatch.setattr(cloud.Cloud, "tag_bucket", interrupted)
        assert run_with(cfg, "deploy") == 1
        rc = run(["cleanup"])
        left = boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]
    assert rc == 0 and left == []


def test_smoke_does_not_invoke_untagged_function(sandbox, monkeypatch):
    calls = []
    monkeypatch.setattr(cloud.Cloud, "invoke", lambda self, payload, qualifier: calls.append(1))
    with mock_aws():
        arn = boto3.client("iam", region_name="ap-northeast-2").create_role(
            RoleName="someone-else", AssumeRolePolicyDocument=json.dumps(cloud.LAMBDA_TRUST))["Role"]["Arn"]
        boto3.client("lambda", region_name="ap-northeast-2").create_function(
            FunctionName="bike-demand-predict", Runtime="python3.12", Role=arn, Handler="lambda_function.lambda_handler",
            Code={"ZipFile": (sandbox / "lambda_package.zip").read_bytes()}, Publish=True)
        rc_smoke = run(["smoke"])
        rc_eval = run(["cloud-eval"])
    recs = records(sandbox)
    assert rc_smoke == 1 and rc_eval == 1 and "태그" in recs[-1]["error"] and calls == []


def test_foreign_log_group_in_recorded_region_does_not_block_cleanup(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    with mock_aws():
        assert run(["deploy"]) == 0
        boto3.client("logs", region_name="us-west-2").create_log_group(logGroupName="/aws/lambda/bike-demand-predict")
        run(["status"], region="us-west-2")
        rc = run(["cleanup"])
        groups = boto3.client("logs", region_name="us-west-2").describe_log_groups()["logGroups"]
        rc_rename = run_with(renamed_config(), "deploy")
    clean = read(sandbox, "results/cloud/cleanup.json")
    assert rc == 0 and clean["not_ours"] == ["us-west-2:log_group"] and len(groups) == 1 and rc_rename == 0


def test_role_is_kept_while_foreign_function_uses_it(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    with mock_aws():
        assert run(["deploy"]) == 0
        role_arn = boto3.client("iam", region_name="ap-northeast-2").get_role(
            RoleName="bike-demand-predict-role")["Role"]["Arn"]
        boto3.client("lambda", region_name="us-west-2").create_function(
            FunctionName="bike-demand-predict", Runtime="python3.12", Role=role_arn,
            Handler="lambda_function.lambda_handler", Code={"ZipFile": (sandbox / "lambda_package.zip").read_bytes()})
        run(["status"], region="us-west-2")
        rc = run(["cleanup"])
        roles = [r["RoleName"] for r in boto3.client("iam", region_name="ap-northeast-2").list_roles()["Roles"]]
    rec = records(sandbox)[-1]
    clean = failed_result(sandbox, rec, "results/cloud/cleanup.json")
    role_action = next(a for a in clean["actions"] if a["resource"] == "iam:bike-demand-predict-role")
    assert rc == 1 and roles == ["bike-demand-predict-role"] and "태그 없는 같은 이름의 함수" in role_action["result"]


def test_cleanup_elsewhere_does_not_drop_recorded_bucket(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    cfg = load_config()
    cfg["cloud"]["bucket"] = "bike-demand-mlops-mine"
    with mock_aws():
        assert run_with(cfg, "deploy") == 0
        with open(runlog.RUN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": "z", "step": "cleanup", "status": "failed",
                                "cloud": {"region": "ap-northeast-2", **CURRENT_NAMES},
                                "summary": {"all_gone": True, "buckets_gone": []}}) + "\n")
        assert run(["status"]) == 0
        seen = read(sandbox, "results/cloud/status.json")["resources"]["regions"]["ap-northeast-2"]["s3_bucket"]
        rc = run(["cleanup"])
        left = boto3.client("s3", region_name="ap-northeast-2").list_buckets()["Buckets"]
    assert seen["exists"] is True and rc == 0 and left == [] and cloud.recorded_buckets() == set()


def test_unexpected_error_during_cleanup_still_writes_result(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    with mock_aws():
        assert run(["deploy"]) == 0

        def timeout(self):
            raise RuntimeError("read timeout")
        monkeypatch.setattr(cloud.Cloud, "log_group_tags", timeout)
        rc = run(["cleanup"])
    rec = records(sandbox)[-1]
    clean = failed_result(sandbox, rec, "results/cloud/cleanup.json")
    assert rc == 1 and any(e["result"] == "error: RuntimeError" for e in clean["errors"])


def test_bucket_that_only_starts_with_the_prefix_is_not_ours(sandbox, monkeypatch):
    monkeypatch.setattr(cloud, "STATUS_WAIT_S", 0)
    with mock_aws():
        s3 = boto3.client("s3", region_name="ap-northeast-2")
        s3.create_bucket(Bucket="bike-demand-mlops-ap-northeast-2-mydata",
                         CreateBucketConfiguration={"LocationConstraint": "ap-northeast-2"})
        rc_deploy = run(["deploy"])
        rc_clean = run(["cleanup"])
        left = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    assert rc_deploy == 0 and rc_clean == 0 and left == ["bike-demand-mlops-ap-northeast-2-mydata"]
