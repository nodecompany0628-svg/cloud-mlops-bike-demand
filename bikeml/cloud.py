from __future__ import annotations

import base64
import json
import os
import re
import secrets as pysecrets
import time

import pandas as pd

from .common import (
    ROOT,
    load_json,
    model_dir,
    resolve_profile,
    resolve_region,
    sha256_file,
    write_json,
)
from .evaluate import daily_mae, metrics
from .runlog import git_info, now_iso, sanitize
from .servicecheck import expected_model, run_cases, sample_day_comparison

LAMBDA_TRUST = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                   "Action": "sts:AssumeRole"}],
}
NOT_FOUND = {"404", "NoSuchBucket", "NotFound", "ResourceNotFoundException", "NoSuchEntity"}
STATUS_RETRIES = 10
STATUS_WAIT_S = 3


def _code(e) -> str:
    return e.response.get("Error", {}).get("Code", "")


class Cloud:
    def __init__(self, cfg: dict, args, ctx=None):
        import boto3

        self.cfg = cfg
        c = cfg["cloud"]
        self.profile = resolve_profile(cfg, getattr(args, "profile", None))
        self.region = resolve_region(cfg, getattr(args, "region", None))
        if not self.profile and not os.environ.get("AWS_ACCESS_KEY_ID"):
            raise RuntimeError("AWS 프로필을 --profile 또는 환경 변수 AWS_PROFILE 로 지정하세요")
        self.session = boto3.Session(profile_name=self.profile, region_name=self.region)
        ident = self.session.client("sts").get_caller_identity()
        self.account = ident["Account"]
        self.caller_type = ident["Arn"].split(":", 5)[5].split("/")[0]
        if ctx is not None and self.account not in ctx.secrets:
            ctx.secrets.append(self.account)
        print(f"  AWS 계정 …{self.account[-4:]} / {ident['Arn'].split(':', 5)[5]} / 프로필 {self.profile} / 리전 {self.region}")
        self.fn = c["function_name"]
        self.role = c["role_name"]
        self.bucket_prefix = f"{c['bucket_prefix']}-{self.region}-"
        self.log_group = f"/aws/lambda/{self.fn}"
        self.tags = c.get("tags", {})
        self.s3 = self.session.client("s3")
        self.iam = self.session.client("iam")
        self.lam = self.session.client("lambda")
        self.logs = self.session.client("logs")
        self.bucket = c.get("bucket") or self.find_bucket()

    def info(self) -> dict:
        return {"region": self.region, "profile": self.profile, "caller_type": self.caller_type,
                "function": self.fn, "role": self.role, "bucket": self.bucket, "log_group": self.log_group}

    def role_arn(self) -> str:
        return f"arn:aws:iam::{self.account}:role/{self.role}"

    def model_key(self) -> str:
        return f"models/{self.cfg['model']['version']}/model.joblib"

    def find_bucket(self) -> str | None:
        names = []
        for page in self.s3.get_paginator("list_buckets").paginate():
            names += [b["Name"] for b in page.get("Buckets", []) if b["Name"].startswith(self.bucket_prefix)]
        if len(names) > 1:
            raise RuntimeError(f"버킷이 여러 개입니다 {names}. config/pipeline.json 의 cloud.bucket 에 하나를 적으세요")
        return names[0] if names else None

    def ensure_bucket(self) -> str:
        if self.bucket:
            self.s3.head_bucket(Bucket=self.bucket)
            created = False
        else:
            self.bucket = self.bucket_prefix + pysecrets.token_hex(4)
            kwargs = {} if self.region == "us-east-1" else {
                "CreateBucketConfiguration": {"LocationConstraint": self.region}}
            self.s3.create_bucket(Bucket=self.bucket, **kwargs)
            self.s3.get_waiter("bucket_exists").wait(Bucket=self.bucket)
            created = True
        self.s3.put_public_access_block(Bucket=self.bucket, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        self.s3.put_bucket_versioning(Bucket=self.bucket, VersioningConfiguration={"Status": "Enabled"})
        self.s3.put_bucket_tagging(Bucket=self.bucket, Tagging={
            "TagSet": [{"Key": k, "Value": v} for k, v in self.tags.items()]})
        self.s3.put_bucket_lifecycle_configuration(Bucket=self.bucket, LifecycleConfiguration={"Rules": [
            {"ID": "expire-old-packages", "Status": "Enabled", "Filter": {"Prefix": "packages/"},
             "Expiration": {"Days": 30}, "NoncurrentVersionExpiration": {"NoncurrentDays": 1}},
            {"ID": "expire-noncurrent", "Status": "Enabled", "Filter": {"Prefix": ""},
             "NoncurrentVersionExpiration": {"NoncurrentDays": 30}},
        ]})
        return "created" if created else "exists"

    def role_policy(self) -> dict:
        return {
            "Version": "2012-10-17",
            "Statement": [
                {"Sid": "WriteOwnLogs", "Effect": "Allow",
                 "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                 "Resource": f"arn:aws:logs:{self.region}:{self.account}:log-group:{self.log_group}:*"},
                {"Sid": "ReadModels", "Effect": "Allow", "Action": "s3:GetObject",
                 "Resource": f"arn:aws:s3:::{self.bucket}/models/*"},
            ],
        }

    def ensure_role(self) -> str:
        created = False
        try:
            self.iam.get_role(RoleName=self.role)
        except self.iam.exceptions.NoSuchEntityException:
            self.iam.create_role(RoleName=self.role, AssumeRolePolicyDocument=json.dumps(LAMBDA_TRUST),
                                 Description="bike-demand-predict Lambda execution role",
                                 Tags=[{"Key": k, "Value": v} for k, v in self.tags.items()])
            self.iam.get_waiter("role_exists").wait(RoleName=self.role)
            created = True
        self.iam.put_role_policy(RoleName=self.role, PolicyName="bike-demand-predict-inline",
                                 PolicyDocument=json.dumps(self.role_policy()))
        return "created" if created else "exists"

    def ensure_log_group(self, retention_days: int) -> str:
        created = False
        try:
            self.logs.create_log_group(logGroupName=self.log_group, tags=self.tags)
            created = True
        except self.logs.exceptions.ResourceAlreadyExistsException:
            pass
        self.logs.put_retention_policy(logGroupName=self.log_group, retentionInDays=retention_days)
        return "created" if created else "exists"

    def upload(self, local, key: str, metadata: dict | None = None) -> dict:
        extra = {"Metadata": metadata} if metadata else {}
        with open(local, "rb") as f:
            r = self.s3.put_object(Bucket=self.bucket, Key=key, Body=f, **extra)
        return {"uri": f"s3://{self.bucket}/{key}", "version_id": r.get("VersionId"), "etag": r.get("ETag")}

    def function_exists(self) -> bool:
        try:
            self.lam.get_function(FunctionName=self.fn)
            return True
        except self.lam.exceptions.ResourceNotFoundException:
            return False

    def deploy_function(self, lcfg: dict, code_key: str, env: dict, description: str) -> dict:
        conf = dict(Role=self.role_arn(), Handler=lcfg["handler"], Runtime=lcfg["runtime"],
                    MemorySize=lcfg["memory_mb"], Timeout=lcfg["timeout_s"],
                    EphemeralStorage={"Size": lcfg["ephemeral_storage_mb"]},
                    Environment={"Variables": env}, Description=description[:256])
        if not self.function_exists():
            for attempt in range(10):
                try:
                    self.lam.create_function(FunctionName=self.fn, Code={"S3Bucket": self.bucket, "S3Key": code_key},
                                             Architectures=[lcfg["architecture"]], PackageType="Zip",
                                             Tags=self.tags, **conf)
                    break
                except self.lam.exceptions.InvalidParameterValueException as e:
                    if "cannot be assumed" not in str(e) or attempt == 9:
                        raise
                    print(f"  IAM 역할 전파 대기 중… ({attempt + 1}/10)")
                    time.sleep(6)
            self.lam.get_waiter("function_active_v2").wait(FunctionName=self.fn)
            action = "created"
        else:
            self.lam.update_function_code(FunctionName=self.fn, S3Bucket=self.bucket, S3Key=code_key,
                                          Architectures=[lcfg["architecture"]])
            self.lam.get_waiter("function_updated_v2").wait(FunctionName=self.fn)
            self.lam.update_function_configuration(FunctionName=self.fn, **conf)
            self.lam.get_waiter("function_updated_v2").wait(FunctionName=self.fn)
            action = "updated"
        ver = self.lam.publish_version(FunctionName=self.fn, Description=description[:256])
        self.lam.get_waiter("function_active_v2").wait(FunctionName=self.fn, Qualifier=ver["Version"])
        concurrency = self.set_concurrency(lcfg.get("reserved_concurrency"))
        return {"action": action, "version": ver["Version"], "code_sha256": ver["CodeSha256"],
                "last_modified": ver["LastModified"], "memory_mb": ver["MemorySize"],
                "timeout_s": ver["Timeout"], "runtime": ver["Runtime"], "reserved_concurrency": concurrency}

    def set_concurrency(self, n) -> dict:
        from botocore.exceptions import ClientError

        if not n:
            try:
                self.lam.delete_function_concurrency(FunctionName=self.fn)
            except ClientError as e:
                if _code(e) not in NOT_FOUND:
                    raise
            return {"set": False, "reason": "설정하지 않음 (예약 해제)"}
        try:
            self.lam.put_function_concurrency(FunctionName=self.fn, ReservedConcurrentExecutions=int(n))
            return {"set": True, "value": int(n)}
        except ClientError as e:
            if _code(e) != "InvalidParameterValueException":
                raise
            return {"set": False, "reason": f"계정 동시 실행 한도가 낮아 예약하지 못함: {e}"}

    def latest_version(self) -> str:
        versions = []
        for page in self.lam.get_paginator("list_versions_by_function").paginate(FunctionName=self.fn):
            versions += [v["Version"] for v in page["Versions"] if v["Version"] != "$LATEST"]
        if not versions:
            raise RuntimeError("게시된 함수 버전이 없습니다. deploy 를 먼저 실행하세요")
        return str(max(int(v) for v in versions))

    def invoke(self, payload: dict, qualifier: str):
        t0 = time.perf_counter()
        r = self.lam.invoke(FunctionName=self.fn, Qualifier=qualifier, Payload=json.dumps(payload).encode(),
                            LogType="Tail")
        ms = round((time.perf_counter() - t0) * 1000, 1)
        raw = r["Payload"].read().decode()
        tail = base64.b64decode(r.get("LogResult", "")).decode(errors="replace")
        meta = {
            "request_id": r.get("ResponseMetadata", {}).get("RequestId"),
            "function_version": r.get("ExecutedVersion"),
            "function_error": r.get("FunctionError"),
            "client_ms": ms,
            "report": parse_report(tail),
        }
        body = json.loads(raw) if raw else {}
        if not isinstance(body, dict):
            body = {"statusCode": None, "body": {"ok": False, "error": {
                "code": "BAD_RESPONSE", "message": f"응답이 객체가 아닙니다: {raw[:200]}"}}}
        if r.get("FunctionError"):
            meta["log_tail"] = tail[-1500:]
            return {"statusCode": None, "body": {"ok": False, "error": {
                "code": "FUNCTION_ERROR", "message": body.get("errorMessage"),
                "type": body.get("errorType")}}}, meta
        return body, meta

    def bucket_status(self) -> dict:
        from botocore.exceptions import ClientError

        if not self.bucket:
            return {"exists": False}
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            if _code(e) in NOT_FOUND:
                return {"exists": False}
            return {"exists": None, "error": _code(e)}
        n, size = 0, 0
        for page in self.s3.get_paginator("list_object_versions").paginate(Bucket=self.bucket):
            for v in page.get("Versions", []):
                n += 1
                size += v["Size"]
        return {"exists": True, "object_versions": n, "total_mb": round(size / 1e6, 2)}

    def status(self) -> dict:
        out = {}
        try:
            f = self.lam.get_function(FunctionName=self.fn)
            conf = f["Configuration"]
            versions = []
            for page in self.lam.get_paginator("list_versions_by_function").paginate(FunctionName=self.fn):
                versions += [v["Version"] for v in page["Versions"]]
            out["lambda_function"] = {
                "exists": True, "state": conf.get("State"), "runtime": conf["Runtime"],
                "memory_mb": conf["MemorySize"], "timeout_s": conf["Timeout"],
                "last_modified": conf["LastModified"], "versions": versions,
                "reserved_concurrency": (f.get("Concurrency") or {}).get("ReservedConcurrentExecutions"),
                "env_MODEL_URI": conf.get("Environment", {}).get("Variables", {}).get("MODEL_URI")}
        except self.lam.exceptions.ResourceNotFoundException:
            out["lambda_function"] = {"exists": False}
        try:
            self.iam.get_role(RoleName=self.role)
            pols = self.iam.list_role_policies(RoleName=self.role)["PolicyNames"]
            out["iam_role"] = {"exists": True, "inline_policies": pols}
        except self.iam.exceptions.NoSuchEntityException:
            out["iam_role"] = {"exists": False}
        out["s3_bucket"] = self.bucket_status()
        groups = self.logs.describe_log_groups(logGroupNamePrefix=self.log_group)["logGroups"]
        g = next((x for x in groups if x["logGroupName"] == self.log_group), None)
        out["log_group"] = ({"exists": True, "retention_days": g.get("retentionInDays"),
                             "stored_bytes": g.get("storedBytes")} if g else {"exists": False})
        return out

    def cleanup(self) -> list[dict]:
        from botocore.exceptions import ClientError

        done = []

        def step(name, fn):
            try:
                fn()
                done.append({"resource": name, "result": "deleted"})
            except ClientError as e:
                done.append({"resource": name, "result": "not_found" if _code(e) in NOT_FOUND else f"error: {_code(e)}"})

        step(f"lambda:{self.fn}", lambda: self.lam.delete_function(FunctionName=self.fn))
        step(f"logs:{self.log_group}", lambda: self.logs.delete_log_group(logGroupName=self.log_group))

        def del_role():
            for p in self.iam.list_role_policies(RoleName=self.role)["PolicyNames"]:
                self.iam.delete_role_policy(RoleName=self.role, PolicyName=p)
            self.iam.delete_role(RoleName=self.role)
        step(f"iam:{self.role}", del_role)

        def del_bucket():
            if not self.bucket:
                raise ClientError({"Error": {"Code": "NoSuchBucket", "Message": "버킷 없음"}}, "HeadBucket")
            self.s3.head_bucket(Bucket=self.bucket)
            for page in self.s3.get_paginator("list_object_versions").paginate(Bucket=self.bucket):
                objs = [{"Key": v["Key"], "VersionId": v["VersionId"]}
                        for v in page.get("Versions", []) + page.get("DeleteMarkers", [])]
                if objs:
                    self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objs, "Quiet": True})
            self.s3.delete_bucket(Bucket=self.bucket)
        step(f"s3:{self.bucket}", del_bucket)
        return done


REPORT_RE = {
    "duration_ms": r"\tDuration: ([\d.]+) ms",
    "billed_ms": r"Billed Duration: ([\d.]+) ms",
    "memory_mb": r"Memory Size: (\d+) MB",
    "max_memory_used_mb": r"Max Memory Used: (\d+) MB",
    "init_ms": r"Init Duration: ([\d.]+) ms",
}


def parse_report(tail: str) -> dict:
    out = {}
    for k, pat in REPORT_RE.items():
        m = re.search(pat, tail)
        if m:
            out[k] = float(m.group(1))
    if "Task timed out" in tail:
        out["timed_out"] = True
    return out


def all_gone(status: dict) -> bool:
    return all(v.get("exists") is False for v in status.values())


def _lambda_cfg(ctx) -> dict:
    path = ROOT / (ctx.args.lambda_config or ctx.cfg["cloud"]["lambda_config"])
    ctx.add_config(path)
    return load_json(path)


def _load_or_empty(path) -> dict:
    return load_json(path) if path.exists() else {}


def preflight_problems(model_sha: str, model_file_sha: str, evaluation: dict, service_check: dict,
                       package_info: dict, zip_sha: str | None, app_files: dict, requirements_sha: str) -> list[str]:
    problems = []
    if model_file_sha != model_sha:
        problems.append("모델 파일이 metadata.json 의 sha256 과 다름")
    if not evaluation.get("passed") or (evaluation.get("model") or {}).get("sha256") != model_sha:
        problems.append("이 모델로 evaluate 를 통과한 기록이 없음")
    if not service_check.get("all_passed") or (service_check.get("model") or {}).get("sha256") != model_sha:
        problems.append("이 모델로 check-local 을 통과한 기록이 없음")
    if zip_sha is None or zip_sha != package_info.get("zip_sha256"):
        problems.append("build/lambda_package.zip 이 없거나 package.json 과 다름")
    if package_info.get("app_files") != app_files or package_info.get("requirements_sha256") != requirements_sha:
        problems.append("패키지를 만든 뒤 Lambda 코드나 requirements-lambda.txt 가 바뀜")
    return problems


def deploy_preflight(cfg: dict) -> dict:
    from . import package

    meta = load_json(model_dir(cfg) / "metadata.json")
    pkg = _load_or_empty(ROOT / "results" / "cloud" / "package.json")
    zip_path = package.ZIP_PATH
    problems = preflight_problems(
        model_sha=meta["model_sha256"],
        model_file_sha=sha256_file(model_dir(cfg) / "model.joblib"),
        evaluation=_load_or_empty(ROOT / "results" / "evaluation" / "metrics.json"),
        service_check=_load_or_empty(ROOT / "results" / "service_check" / "local.json"),
        package_info=pkg,
        zip_sha=sha256_file(zip_path) if zip_path.exists() else None,
        app_files={arc: sha256_file(src) for arc, src in package.APP_FILES.items()},
        requirements_sha=sha256_file(package.REQUIREMENTS),
    )
    if problems:
        raise RuntimeError("배포 전 확인 실패: " + "; ".join(problems)
                           + ". python -m bikeml run local package 를 먼저 실행하세요")
    return {"meta": meta, "zip_sha256": pkg["zip_sha256"], "zip_path": zip_path}


def step_deploy(ctx) -> None:
    cfg = ctx.cfg
    pre = deploy_preflight(cfg)
    meta, pkg_sha, zip_path = pre["meta"], pre["zip_sha256"], pre["zip_path"]
    mpath = model_dir(cfg) / "model.joblib"
    ctx.add_input(mpath)
    ctx.add_input(zip_path)
    ctx.model = {"name": meta["name"], "version": meta["version"], "sha256": meta["model_sha256"]}
    lcfg = _lambda_cfg(ctx)
    cloud = Cloud(cfg, ctx.args, ctx)

    res = {"cloud": cloud.info()}
    res["bucket"] = cloud.ensure_bucket()
    res["role"] = cloud.ensure_role()
    res["log_group"] = cloud.ensure_log_group(lcfg["log_retention_days"])

    model_key = cloud.model_key()
    res["model_upload"] = cloud.upload(mpath, model_key, {"sha256": meta["model_sha256"], "version": meta["version"]})
    res["metadata_upload"] = cloud.upload(model_dir(cfg) / "metadata.json", f"models/{meta['version']}/metadata.json")
    code_key = f"packages/lambda-{pkg_sha[:12]}.zip"
    res["package_upload"] = cloud.upload(zip_path, code_key, {"sha256": pkg_sha})

    model_uri = getattr(ctx.args, "model_uri_override", None) or f"s3://{cloud.bucket}/{model_key}"
    env = {"MODEL_URI": model_uri, "MODEL_VERSION": meta["version"], "MODEL_SHA256": meta["model_sha256"],
           "JOBLIB_MULTIPROCESSING": "0"}
    git = git_info()
    commit = (git["commit"] or "nogit")[:7] + ("+dirty" if git["code_dirty"] else "")
    desc = f"git {commit} model {meta['version']} pkg {pkg_sha[:12]}"
    res["function"] = cloud.deploy_function(lcfg, code_key, env, desc)
    res["env"] = env
    res["cloud"] = cloud.info()
    write_json(ctx.out("results/cloud/deploy.json"), res)
    ctx.cloud = {**cloud.info(), "function_version": res["function"]["version"],
                 "code_sha256": res["function"]["code_sha256"], "model_uri": model_uri,
                 "package_uri": res["package_upload"]["uri"]}
    ctx.summary = {"function_version": res["function"]["version"], "action": res["function"]["action"],
                   "memory_mb": lcfg["memory_mb"], "timeout_s": lcfg["timeout_s"]}


def _cloud_invoker(ctx):
    cloud = Cloud(ctx.cfg, ctx.args, ctx)
    version = cloud.latest_version()
    meta, expected = expected_model(ctx.cfg)
    ctx.model = {"name": meta["name"], "version": meta["version"], "sha256": meta["model_sha256"]}
    ctx.cloud = {**cloud.info(), "function_version": version}

    def invoke(payload):
        return cloud.invoke(payload, version)
    return cloud, version, expected, invoke


def step_smoke(ctx) -> None:
    cloud, version, expected, invoke = _cloud_invoker(ctx)
    ctx.add_input(ROOT / "samples" / "service_cases.json")
    report = run_cases(invoke, expected, f"lambda:{cloud.fn}:{version}", ctx.cfg, ctx.secrets)
    day = sample_day_comparison(report.pop("responses"), ctx.cfg)
    report["sample_day"] = day
    write_json(ctx.out("results/service_check/cloud.json"), report)
    ctx.summary = {"passed": report["passed"], "total": report["total"], "function_version": version}
    if not report["all_passed"]:
        bad = [f"{c['name']}: {c['problems'][:1]}" for c in report["cases"] if not c["passed"]]
        raise RuntimeError(f"클라우드 서비스 검사 실패 {len(bad)}건: {bad[:3]}")


def step_cloud_eval(ctx) -> None:
    from .data import records

    cfg = ctx.cfg
    cloud, version, expected, invoke = _cloud_invoker(ctx)
    test_path = ROOT / cfg["data"]["processed_dir"] / "test.csv"
    local_path = ROOT / "results" / "evaluation" / "predictions_test.csv"
    ctx.add_input(test_path)
    ctx.add_input(local_path)
    test = pd.read_csv(test_path)
    local = pd.read_csv(local_path)
    if len(local) != len(test) or not (local.datetime.to_numpy() == test.datetime.to_numpy()).all():
        raise RuntimeError("로컬 예측 파일과 평가 자료의 행이 맞지 않습니다. evaluate 단계를 다시 실행하세요")
    recs = records(test)
    size = cfg["evaluation"]["cloud_batch_size"]

    preds, calls = [], []
    for start in range(0, len(recs), size):
        batch = recs[start:start + size]
        resp, meta = sanitize(invoke({"instances": batch}), ctx.secrets)
        calls.append({"rows": f"{start}-{start + len(batch) - 1}", "status": resp.get("statusCode"), **meta})
        body = resp.get("body") or {}
        problem = None
        if resp.get("statusCode") != 200:
            problem = f"Lambda 호출 실패 (행 {start}~): {json.dumps(body, ensure_ascii=False)[:300]}"
        elif body.get("model_sha256") != expected["sha256"]:
            problem = "Lambda 가 다른 모델로 예측했습니다"
        elif len(body.get("predictions") or []) != len(batch):
            problem = f"예측 건수 {len(body.get('predictions') or [])} 가 요청 건수 {len(batch)} 와 다릅니다"
        if problem:
            write_json(ctx.out("results/cloud/cloud_eval_calls.json"), calls)
            raise RuntimeError(problem)
        preds += [p["predicted_value"] for p in body["predictions"]]

    y = test["cnt"].to_numpy()
    out = pd.DataFrame({"datetime": test.datetime, "actual": y, "pred_cloud": preds,
                        "pred_local": local.pred_model})
    out["diff"] = (out.pred_cloud - out.pred_local).round(4)
    out.to_csv(ctx.out("results/cloud/predictions_cloud.csv"), index=False, lineterminator="\n")
    m = metrics(y, out.pred_cloud)
    m["daily_total_mae"] = daily_mae(test.datetime, y, out.pred_cloud)
    local_metrics = load_json(ROOT / "results" / "evaluation" / "metrics.json")["metrics"]["model"]
    report = {
        "function": cloud.fn, "function_version": version, "region": cloud.region,
        "model": expected, "rows": int(len(test)), "batches": len(calls), "batch_size": size,
        "metrics_cloud": m,
        "metrics_local": {k: local_metrics[k] for k in ("n", "mae", "rmse", "bias", "daily_total_mae")},
        "max_abs_diff_vs_local": float(out["diff"].abs().max()),
        "identical_to_local": bool((out["diff"].abs() < 0.011).all()),
        "calls": calls,
    }
    write_json(ctx.out("results/cloud/cloud_eval.json"), report)
    ctx.summary = {"rows": report["rows"], "mae_cloud": m["mae"], "mae_local": local_metrics["mae"],
                   "max_abs_diff_vs_local": report["max_abs_diff_vs_local"], "function_version": version}
    if not report["identical_to_local"]:
        raise RuntimeError(f"클라우드 예측이 로컬과 다릅니다 (최대 차이 {report['max_abs_diff_vs_local']})")


def step_status(ctx) -> None:
    cloud = Cloud(ctx.cfg, ctx.args, ctx)
    st = {"checked_at": now_iso(), **cloud.info(), "resources": cloud.status()}
    name = getattr(ctx.args, "status_name", None) or "status"
    write_json(ctx.out(f"results/cloud/{name}.json"), st)
    ctx.cloud = cloud.info()
    ctx.summary = {k: v.get("exists") for k, v in st["resources"].items()}
    print(json.dumps(ctx.summary, ensure_ascii=False))


def step_cleanup(ctx) -> None:
    cloud = Cloud(ctx.cfg, ctx.args, ctx)
    before = cloud.status()
    deleted = cloud.cleanup()
    after = cloud.status()
    for _ in range(STATUS_RETRIES):
        if all_gone(after):
            break
        time.sleep(STATUS_WAIT_S)
        after = cloud.status()
    report = {"cleaned_at": now_iso(), **cloud.info(), "before": before, "actions": deleted, "after": after,
              "all_gone": all_gone(after)}
    write_json(ctx.out("results/cloud/cleanup.json"), report)
    ctx.cloud = cloud.info()
    ctx.summary = {"all_gone": report["all_gone"], "actions": {d["resource"]: d["result"] for d in deleted}}
    if not report["all_gone"]:
        raise RuntimeError(f"남은 자원이 있거나 확인하지 못했습니다: {after}")
