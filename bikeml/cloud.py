from __future__ import annotations

import base64
import hmac
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
NOT_FOUND = {"404", "NoSuchBucket", "NotFound", "NoSuchKey", "ResourceNotFoundException", "NoSuchEntity"}
STATUS_RETRIES = 10
STATUS_WAIT_S = 3
UPLOAD_RETRIES = 30
UPLOAD_WAIT_S = 30
ALIAS = "live"
FUNCTION_DESCRIPTION = "bike-demand hourly rental prediction"
REGIONS_STATE = ROOT / ".state" / "regions.json"
RESULTS = ROOT / "results"


class NotOurs(Exception):
    pass


def _code(e) -> str:
    return e.response.get("Error", {}).get("Code", "")


def make_session(profile: str | None, region: str):
    import boto3

    if profile:
        return boto3.Session(profile_name=profile, region_name=region)
    if os.environ.get("BIKEML_USE_ENV_CREDENTIALS") != "1":
        raise RuntimeError("AWS 프로필을 --profile 또는 환경 변수 AWS_PROFILE 로 지정하세요. "
                           "환경 변수의 키를 쓰려면 BIKEML_USE_ENV_CREDENTIALS=1 을 함께 지정하세요")
    key, secret = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise RuntimeError("BIKEML_USE_ENV_CREDENTIALS=1 이지만 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY 가 없습니다")
    return boto3.Session(aws_access_key_id=key, aws_secret_access_key=secret,
                         aws_session_token=os.environ.get("AWS_SESSION_TOKEN"), region_name=region)


class Cloud:
    def __init__(self, cfg: dict, args, ctx=None, region: str | None = None):
        self.cfg = cfg
        c = cfg["cloud"]
        self.profile = resolve_profile(cfg, getattr(args, "profile", None))
        self.region = region or resolve_region(cfg, getattr(args, "region", None))
        self.session = make_session(self.profile, self.region)
        ident = self.session.client("sts").get_caller_identity()
        self.account = ident["Account"]
        self.caller_type = ident["Arn"].split(":", 5)[5].split("/")[0]
        if ctx is not None and self.account not in ctx.secrets:
            ctx.secrets.append(self.account)
        print(f"  AWS 계정 ...{self.account[-4:]} / {self.caller_type} / 프로필 {self.profile} / 리전 {self.region}")
        self.fn = c["function_name"]
        self.role = c["role_name"]
        self.bucket_prefix = f"{c['bucket_prefix']}-{self.region}-"
        self.log_group = f"/aws/lambda/{self.fn}"
        self.tags = c.get("tags", {})
        if not self.tags:
            raise RuntimeError("config 의 cloud.tags 가 비어 있으면 이 과제의 자원을 구분할 수 없습니다")
        self.s3 = self.session.client("s3")
        self.iam = self.session.client("iam")
        self.lam = self.session.client("lambda")
        self.logs = self.session.client("logs")
        self.bucket_base = c["bucket_prefix"]
        self.configured = c.get("bucket")
        if self.configured and not self.configured.startswith(f"{self.bucket_base}-"):
            raise RuntimeError(f"cloud.bucket 은 {self.bucket_base}- 로 시작해야 합니다: {self.configured}")
        self.configured_region = self.bucket_region(self.configured) if self.configured else None
        found = self.find_buckets()
        if self.configured and self.configured_region == self.region:
            found = sorted(set(found) | {self.configured})
        self.buckets = found
        if self.configured:
            self.bucket = self.configured if self.configured_region in (None, self.region) else None
        else:
            self.bucket = found[0] if len(found) == 1 else None

    def info(self) -> dict:
        return {"region": self.region, "credentials": "profile" if self.profile else "environment",
                "caller_type": self.caller_type,
                "function": self.fn, "role": self.role, "bucket": self.bucket, "log_group": self.log_group}

    def role_arn(self) -> str:
        return f"arn:aws:iam::{self.account}:role/{self.role}"

    def model_prefix(self, model_sha: str) -> str:
        return f"models/{self.cfg['model']['version']}/{model_sha[:12]}"

    def ours(self, tags: dict) -> bool:
        return all(tags.get(k) == v for k, v in self.tags.items())

    def all_bucket_names(self) -> list[str]:
        names = []
        for page in self.s3.get_paginator("list_buckets").paginate():
            names += [b["Name"] for b in page.get("Buckets", [])]
        return names

    def find_buckets(self) -> list[str]:
        return sorted(n for n in self.all_bucket_names() if n.startswith(self.bucket_prefix))

    def bucket_regions(self) -> set[str]:
        pattern = re.compile(re.escape(self.bucket_base) + r"-([a-z]{2}(?:-[a-z]+)+-\d+)-[0-9a-f]{8}")
        return {m.group(1) for n in self.all_bucket_names() if (m := pattern.fullmatch(n))}

    def bucket_region(self, name: str) -> str | None:
        from botocore.exceptions import ClientError

        try:
            loc = self.s3.get_bucket_location(Bucket=name).get("LocationConstraint")
        except ClientError as e:
            if _code(e) in NOT_FOUND:
                return None
            raise
        return loc or "us-east-1"

    def bucket_tags(self) -> dict:
        from botocore.exceptions import ClientError

        try:
            return {t["Key"]: t["Value"] for t in self.s3.get_bucket_tagging(Bucket=self.bucket)["TagSet"]}
        except ClientError as e:
            if _code(e) in ("NoSuchTagSet", "NoSuchTagSetError"):
                return {}
            raise

    def role_tags(self) -> dict:
        return {t["Key"]: t["Value"] for t in self.iam.list_role_tags(RoleName=self.role)["Tags"]}

    def function_tags(self) -> dict:
        arn = self.lam.get_function(FunctionName=self.fn)["Configuration"]["FunctionArn"]
        return self.lam.list_tags(Resource=arn).get("Tags", {})

    def log_group_tags(self) -> dict:
        return self.logs.list_tags_log_group(logGroupName=self.log_group).get("tags", {})

    def require_ours(self, kind: str, tags: dict) -> None:
        if not self.ours(tags):
            raise NotOurs(f"{kind} 에 이 과제의 태그가 없어 건드리지 않습니다")

    def bucket_is_empty(self) -> bool:
        page = self.s3.list_object_versions(Bucket=self.bucket, MaxKeys=1)
        return not page.get("Versions") and not page.get("DeleteMarkers")

    def leftover_bucket(self, tags: dict) -> bool:
        named = re.fullmatch(re.escape(self.bucket_prefix) + "[0-9a-f]{8}", self.bucket)
        return not tags and bool(named) and self.bucket_is_empty()

    def tag_bucket(self) -> None:
        self.s3.put_bucket_tagging(Bucket=self.bucket, Tagging={
            "TagSet": [{"Key": k, "Value": v} for k, v in self.tags.items()]})

    def ensure_bucket(self) -> str:
        if self.configured and self.configured_region not in (None, self.region):
            raise RuntimeError(f"cloud.bucket {self.configured} 의 리전({self.configured_region})이 "
                               f"지금 리전({self.region})과 다릅니다")
        if not self.configured and len(self.buckets) > 1:
            raise RuntimeError(f"버킷이 여러 개입니다 {self.buckets}. cleanup 으로 정리하거나 "
                               "config/pipeline.json 의 cloud.bucket 에 하나를 적으세요")
        if self.bucket and (not self.configured or self.configured_region == self.region):
            self.s3.head_bucket(Bucket=self.bucket)
            tags = self.bucket_tags()
            created = "exists"
            if not self.ours(tags):
                if not self.leftover_bucket(tags):
                    self.require_ours(f"버킷 {self.bucket}", tags)
                self.tag_bucket()
                created = "adopted"
        else:
            self.bucket = self.configured or self.bucket_prefix + pysecrets.token_hex(4)
            kwargs = {} if self.region == "us-east-1" else {
                "CreateBucketConfiguration": {"LocationConstraint": self.region}}
            self.s3.create_bucket(Bucket=self.bucket, **kwargs)
            self.s3.get_waiter("bucket_exists").wait(Bucket=self.bucket)
            self.tag_bucket()
            self.buckets = [self.bucket]
            created = "created"
        self.s3.put_public_access_block(Bucket=self.bucket, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        self.s3.put_bucket_versioning(Bucket=self.bucket, VersioningConfiguration={"Status": "Enabled"})
        self.s3.put_bucket_lifecycle_configuration(Bucket=self.bucket, LifecycleConfiguration={"Rules": [
            {"ID": "expire-old-packages", "Status": "Enabled", "Filter": {"Prefix": "packages/"},
             "Expiration": {"Days": 30}, "NoncurrentVersionExpiration": {"NoncurrentDays": 1}},
            {"ID": "expire-noncurrent", "Status": "Enabled", "Filter": {"Prefix": ""},
             "NoncurrentVersionExpiration": {"NoncurrentDays": 30}},
        ]})
        return created

    def role_policy(self) -> dict:
        return {
            "Version": "2012-10-17",
            "Statement": [
                {"Sid": "WriteOwnLogs", "Effect": "Allow",
                 "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                 "Resource": f"arn:aws:logs:{self.region}:{self.account}:log-group:{self.log_group}:*"},
                {"Sid": "ReadModels", "Effect": "Allow", "Action": ["s3:GetObject", "s3:GetObjectVersion"],
                 "Resource": f"arn:aws:s3:::{self.bucket}/models/*"},
            ],
        }

    def ensure_role(self) -> str:
        created = False
        try:
            self.iam.get_role(RoleName=self.role)
            self.require_ours(f"IAM 역할 {self.role}", self.role_tags())
        except self.iam.exceptions.NoSuchEntityException:
            self.iam.create_role(RoleName=self.role, AssumeRolePolicyDocument=json.dumps(LAMBDA_TRUST),
                                 Description="bike-demand-predict Lambda execution role",
                                 Tags=[{"Key": k, "Value": v} for k, v in self.tags.items()])
            self.iam.get_waiter("role_exists").wait(RoleName=self.role)
            created = True
        self.iam.put_role_policy(RoleName=self.role, PolicyName=f"bike-demand-predict-{self.region}",
                                 PolicyDocument=json.dumps(self.role_policy()))
        return "created" if created else "exists"

    def ensure_log_group(self, retention_days: int) -> str:
        created = False
        try:
            self.logs.create_log_group(logGroupName=self.log_group, tags=self.tags)
            created = True
        except self.logs.exceptions.ResourceAlreadyExistsException:
            self.require_ours(f"로그 그룹 {self.log_group}", self.log_group_tags())
        self.logs.put_retention_policy(logGroupName=self.log_group, retentionInDays=retention_days)
        return "created" if created else "exists"

    def current_object(self, key: str) -> dict | None:
        from botocore.exceptions import ClientError

        try:
            return self.s3.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if _code(e) in NOT_FOUND:
                return None
            raise

    def upload(self, local, key: str, sha256: str, metadata: dict | None = None,
               need_version: bool = False) -> dict:
        from botocore.exceptions import ClientError

        meta = {"sha256": sha256, **(metadata or {})}
        uri = f"s3://{self.bucket}/{key}"
        head = self.current_object(key)
        if head and head.get("Metadata", {}).get("sha256") == sha256:
            version = head.get("VersionId")
            if not need_version or version not in (None, "null"):
                return {"uri": uri, "version_id": version, "uploaded": False}
        attempts = UPLOAD_RETRIES if need_version else 1
        for attempt in range(attempts):
            with open(local, "rb") as f:
                r = self.s3.put_object(Bucket=self.bucket, Key=key, Body=f, Metadata=meta)
            version = r.get("VersionId")
            if not need_version:
                return {"uri": uri, "version_id": version, "uploaded": True}
            if version and version != "null":
                try:
                    self.s3.head_object(Bucket=self.bucket, Key=key, VersionId=version)
                    return {"uri": uri, "version_id": version, "uploaded": True}
                except ClientError as e:
                    if _code(e) not in NOT_FOUND and _code(e) != "NoSuchVersion":
                        raise
            print(f"  버킷 버전 관리가 반영되기를 기다리는 중... ({attempt + 1}/{attempts})")
            time.sleep(UPLOAD_WAIT_S)
        raise RuntimeError("버킷 버전 관리가 켜지지 않아 객체 버전을 받지 못했습니다")

    def function_exists(self) -> bool:
        try:
            self.lam.get_function(FunctionName=self.fn)
            return True
        except self.lam.exceptions.ResourceNotFoundException:
            return False

    def deploy_function(self, lcfg: dict, code_key: str, env: dict, version_description: str) -> dict:
        conf = dict(Role=self.role_arn(), Handler=lcfg["handler"], Runtime=lcfg["runtime"],
                    MemorySize=lcfg["memory_mb"], Timeout=lcfg["timeout_s"],
                    EphemeralStorage={"Size": lcfg["ephemeral_storage_mb"]},
                    Environment={"Variables": env}, Description=FUNCTION_DESCRIPTION)
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
                    print(f"  IAM 역할 전파 대기 중... ({attempt + 1}/10)")
                    time.sleep(6)
            self.lam.get_waiter("function_active_v2").wait(FunctionName=self.fn)
            action = "created"
        else:
            self.require_ours(f"Lambda 함수 {self.fn}", self.function_tags())
            self.lam.update_function_code(FunctionName=self.fn, S3Bucket=self.bucket, S3Key=code_key,
                                          Architectures=[lcfg["architecture"]])
            self.lam.get_waiter("function_updated_v2").wait(FunctionName=self.fn)
            self.lam.update_function_configuration(FunctionName=self.fn, **conf)
            self.lam.get_waiter("function_updated_v2").wait(FunctionName=self.fn)
            action = "updated"
        ver = self.lam.publish_version(FunctionName=self.fn, Description=version_description[:256])
        self.lam.get_waiter("function_active_v2").wait(FunctionName=self.fn, Qualifier=ver["Version"])
        concurrency = self.set_concurrency(lcfg.get("reserved_concurrency"))
        return {"action": action, "version": ver["Version"], "code_sha256": ver["CodeSha256"],
                "last_modified": ver["LastModified"], "memory_mb": ver["MemorySize"],
                "timeout_s": ver["Timeout"], "runtime": ver["Runtime"],
                "version_description": ver.get("Description"), "reserved_concurrency": concurrency}

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

    def point_alias(self, version: str) -> dict:
        try:
            self.lam.update_alias(FunctionName=self.fn, Name=ALIAS, FunctionVersion=version)
            action = "updated"
        except self.lam.exceptions.ResourceNotFoundException:
            self.lam.create_alias(FunctionName=self.fn, Name=ALIAS, FunctionVersion=version)
            action = "created"
        return {"alias": ALIAS, "version": version, "action": action}

    def version_identity(self, version: str) -> dict:
        conf = self.lam.get_function_configuration(FunctionName=self.fn, Qualifier=version)
        env = conf.get("Environment", {}).get("Variables", {})
        return {"region": self.region, "function": self.fn, "function_version": version,
                "revision_id": conf.get("RevisionId"), "code_sha256": conf.get("CodeSha256"),
                "model_uri": env.get("MODEL_URI"), "model_version": env.get("MODEL_VERSION"),
                "model_sha256": env.get("MODEL_SHA256")}

    def alias_version(self) -> str | None:
        try:
            return self.lam.get_alias(FunctionName=self.fn, Name=ALIAS)["FunctionVersion"]
        except self.lam.exceptions.ResourceNotFoundException:
            return None

    def latest_version(self) -> str:
        versions = []
        for page in self.lam.get_paginator("list_versions_by_function").paginate(FunctionName=self.fn):
            versions += [v["Version"] for v in page["Versions"] if v["Version"] != "$LATEST"]
        if not versions:
            raise RuntimeError("게시된 함수 버전이 없습니다. deploy 를 먼저 실행하세요")
        return str(max(int(v) for v in versions))

    def invoke(self, payload: dict, qualifier: str):
        t0 = time.perf_counter()
        r = self.lam.invoke(FunctionName=self.fn, Qualifier=qualifier,
                            Payload=json.dumps(payload, allow_nan=False).encode(), LogType="Tail")
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

    def one_bucket_status(self, name: str) -> dict:
        from botocore.exceptions import ClientError

        try:
            self.s3.head_bucket(Bucket=name)
        except ClientError as e:
            if _code(e) in NOT_FOUND:
                return {"exists": False, "name": name}
            return {"exists": None, "name": name, "error": _code(e)}
        n, size = 0, 0
        for page in self.s3.get_paginator("list_object_versions").paginate(Bucket=name):
            for v in page.get("Versions", []):
                n += 1
                size += v["Size"]
        return {"exists": True, "name": name, "object_versions": n, "total_mb": round(size / 1e6, 2)}

    def bucket_status(self) -> dict:
        found = [self.one_bucket_status(b) for b in self.buckets]
        if not found:
            return {"exists": False}
        if len(found) == 1:
            return found[0]
        states = [b["exists"] for b in found]
        combined = None if None in states else any(states)
        return {"exists": combined, "buckets": found}

    def regional_status(self) -> dict:
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
                "alias_live": self.alias_version(),
                "env_MODEL_URI": conf.get("Environment", {}).get("Variables", {}).get("MODEL_URI")}
        except self.lam.exceptions.ResourceNotFoundException:
            out["lambda_function"] = {"exists": False}
        out["s3_bucket"] = self.bucket_status()
        groups = self.logs.describe_log_groups(logGroupNamePrefix=self.log_group)["logGroups"]
        g = next((x for x in groups if x["logGroupName"] == self.log_group), None)
        out["log_group"] = ({"exists": True, "retention_days": g.get("retentionInDays"),
                             "stored_bytes": g.get("storedBytes")} if g else {"exists": False})
        return out

    def role_status(self) -> dict:
        try:
            self.iam.get_role(RoleName=self.role)
            pols = self.iam.list_role_policies(RoleName=self.role)["PolicyNames"]
            return {"exists": True, "inline_policies": pols}
        except self.iam.exceptions.NoSuchEntityException:
            return {"exists": False}

    def _delete(self, name: str, fn, done: list) -> None:
        from botocore.exceptions import ClientError

        try:
            fn()
            done.append({"resource": name, "result": "deleted"})
        except NotOurs as e:
            done.append({"resource": name, "result": f"skipped: {e}"})
        except ClientError as e:
            done.append({"resource": name, "result": "not_found" if _code(e) in NOT_FOUND else f"error: {_code(e)}"})

    def cleanup_regional(self) -> list[dict]:
        from botocore.exceptions import ClientError

        done = []

        def del_function():
            self.require_ours(f"Lambda 함수 {self.fn}", self.function_tags())
            self.lam.delete_function(FunctionName=self.fn)

        def del_log_group():
            self.require_ours(f"로그 그룹 {self.log_group}", self.log_group_tags())
            self.logs.delete_log_group(logGroupName=self.log_group)

        def del_bucket(name):
            self.s3.head_bucket(Bucket=name)
            self.bucket = name
            tags = self.bucket_tags()
            if not self.ours(tags) and not self.leftover_bucket(tags):
                self.require_ours(f"버킷 {name}", tags)
            for page in self.s3.get_paginator("list_object_versions").paginate(Bucket=name):
                objs = [{"Key": v["Key"], "VersionId": v["VersionId"]}
                        for v in page.get("Versions", []) + page.get("DeleteMarkers", [])]
                if objs:
                    r = self.s3.delete_objects(Bucket=name, Delete={"Objects": objs, "Quiet": True})
                    if r.get("Errors"):
                        err = r["Errors"][0]
                        raise ClientError({"Error": {"Code": err.get("Code", "DeleteObjectsError"),
                                                     "Message": err.get("Message", "")}}, "DeleteObjects")
            self.s3.delete_bucket(Bucket=name)

        def no_bucket():
            raise ClientError({"Error": {"Code": "NoSuchBucket", "Message": "버킷 없음"}}, "HeadBucket")

        self._delete(f"{self.region}:lambda:{self.fn}", del_function, done)
        self._delete(f"{self.region}:logs:{self.log_group}", del_log_group, done)
        for name in self.buckets:
            self._delete(f"{self.region}:s3:{name}", lambda n=name: del_bucket(n), done)
        if not self.buckets:
            self._delete(f"{self.region}:s3:{self.bucket_prefix}*", no_bucket, done)
        self.bucket = self.buckets[0] if len(self.buckets) == 1 else None
        return done

    def cleanup_role(self) -> list[dict]:
        done = []

        def del_role():
            self.require_ours(f"IAM 역할 {self.role}", self.role_tags())
            for p in self.iam.list_role_policies(RoleName=self.role)["PolicyNames"]:
                self.iam.delete_role_policy(RoleName=self.role, PolicyName=p)
            self.iam.delete_role(RoleName=self.role)

        self._delete(f"iam:{self.role}", del_role, done)
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


def all_gone(resources: dict) -> bool:
    regional = [v for r in resources["regions"].values() for v in r.values()]
    return resources["iam_role"].get("exists") is False and all(v.get("exists") is False for v in regional)


def _local_key() -> bytes:
    path = REGIONS_STATE.parent / "local_key"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pysecrets.token_hex(32), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip().encode()


def account_key(account: str) -> str:
    return hmac.new(_local_key(), account.encode(), "sha256").hexdigest()


def deployments() -> list[dict]:
    data = load_json(REGIONS_STATE) if REGIONS_STATE.exists() else []
    return [d for d in data if isinstance(d, dict) and d.get("account") and d.get("region")]


def remember_region(account: str, region: str) -> None:
    entry = {"account": account_key(account), "region": region}
    deps = deployments()
    if entry not in deps:
        write_json(REGIONS_STATE, deps + [entry])


def forget_account(account: str) -> None:
    if REGIONS_STATE.exists():
        write_json(REGIONS_STATE, [d for d in deployments() if d["account"] != account_key(account)])


def other_account_regions(account: str) -> list[str]:
    return sorted({d["region"] for d in deployments() if d["account"] != account_key(account)})


def run_log_regions() -> set[str]:
    from . import runlog

    if not runlog.RUN_LOG.exists():
        return set()
    regions = set()
    for line in runlog.RUN_LOG.read_text(encoding="utf-8").splitlines():
        region = ((json.loads(line).get("cloud") or {}).get("region"))
        if region:
            regions.add(region)
    return regions


def known_regions(cfg: dict, args, account: str) -> list[str]:
    regions = {resolve_region(cfg, getattr(args, "region", None))} | run_log_regions()
    regions |= {d["region"] for d in deployments() if d["account"] == account_key(account)}
    for name in ("deploy.json", "deploy_override.json"):
        path = RESULTS / "cloud" / name
        if path.exists():
            region = load_json(path).get("cloud", {}).get("region")
            if region:
                regions.add(region)
    return sorted(regions)


def _lambda_cfg(ctx) -> dict:
    path = ROOT / (ctx.args.lambda_config or ctx.cfg["cloud"]["lambda_config"])
    ctx.add_config(path)
    return load_json(path)


def _load_or_empty(path) -> dict:
    return load_json(path) if path.exists() else {}


def preflight_problems(model_meta: dict, model_file_sha: str, evaluation: dict, service_check: dict,
                       package_info: dict, zip_sha: str | None, app_files: dict, requirements_sha: str,
                       criteria: dict, test_sha: str, model_mismatch: list, split_mismatch: list,
                       lambda_cfg: dict) -> list[str]:
    model_sha = model_meta.get("model_sha256")
    problems = []
    if model_file_sha != model_sha:
        problems.append("모델 파일이 metadata.json 의 sha256 과 다름")
    if split_mismatch:
        problems.append(f"정리한 자료가 config 의 split 과 다름({', '.join(split_mismatch)})")
    if model_mismatch:
        problems.append(f"모델이 지금 학습 자료와 설정으로 학습되지 않음({', '.join(model_mismatch)})")
    if not evaluation.get("passed") or (evaluation.get("model") or {}).get("sha256") != model_sha:
        problems.append("이 모델로 evaluate 를 통과한 기록이 없음")
    elif evaluation.get("criteria") != criteria or (evaluation.get("eval_data") or {}).get("sha256") != test_sha:
        problems.append("evaluate 이후 합격 기준이나 평가 자료가 바뀜")
    if not service_check.get("all_passed") or (service_check.get("model") or {}).get("sha256") != model_sha:
        problems.append("이 모델로 check-local 을 통과한 기록이 없음")
    elif service_check.get("app_files") != app_files:
        problems.append("check-local 이후 Lambda 코드가 바뀜")
    if zip_sha is None or zip_sha != package_info.get("zip_sha256"):
        problems.append("build/lambda_package.zip 이 없거나 package.json 과 다름")
    if package_info.get("app_files") != app_files or package_info.get("requirements_sha256") != requirements_sha:
        problems.append("패키지를 만든 뒤 Lambda 코드나 requirements-lambda.txt 가 바뀜")
    if package_info.get("runtime") != lambda_cfg.get("runtime") \
            or package_info.get("architecture") != lambda_cfg.get("architecture"):
        problems.append("패키지의 런타임이나 아키텍처가 Lambda 설정과 다름")
    return problems


def deploy_preflight(cfg: dict, lambda_cfg: dict) -> dict:
    import joblib

    from . import package
    from .data import estimator_mismatch, model_mismatches, split_problems

    meta = load_json(model_dir(cfg) / "metadata.json")
    pdir = ROOT / cfg["data"]["processed_dir"]
    mismatch = model_mismatches(meta, cfg)
    if not mismatch and estimator_mismatch(joblib.load(model_dir(cfg) / "model.joblib"), cfg):
        mismatch.append("모델 파일의 하이퍼파라미터")
    pkg = _load_or_empty(RESULTS / "cloud" / "package.json")
    zip_path = package.ZIP_PATH
    problems = preflight_problems(
        model_meta=meta,
        model_file_sha=sha256_file(model_dir(cfg) / "model.joblib"),
        evaluation=_load_or_empty(RESULTS / "evaluation" / "metrics.json"),
        service_check=_load_or_empty(RESULTS / "service_check" / "local.json"),
        package_info=pkg,
        zip_sha=sha256_file(zip_path) if zip_path.exists() else None,
        app_files={arc: sha256_file(src) for arc, src in package.APP_FILES.items()},
        requirements_sha=sha256_file(package.REQUIREMENTS),
        criteria=cfg["evaluation"]["criteria"],
        test_sha=sha256_file(pdir / "test.csv"),
        model_mismatch=mismatch,
        split_mismatch=split_problems(cfg, pd.read_csv(pdir / "train.csv"), pd.read_csv(pdir / "test.csv")),
        lambda_cfg=lambda_cfg,
    )
    if problems:
        raise RuntimeError("배포 전 확인 실패: " + "; ".join(problems)
                           + ". python -m bikeml run local package 를 먼저 실행하세요")
    return {"meta": meta, "zip_sha256": pkg["zip_sha256"], "zip_path": zip_path}


def step_deploy(ctx) -> None:
    cfg = ctx.cfg
    lcfg = _lambda_cfg(ctx)
    pre = deploy_preflight(cfg, lcfg)
    meta, pkg_sha, zip_path = pre["meta"], pre["zip_sha256"], pre["zip_path"]
    mpath = model_dir(cfg) / "model.joblib"
    ctx.add_input(mpath)
    ctx.add_input(zip_path)
    ctx.model = {"name": meta["name"], "version": meta["version"], "sha256": meta["model_sha256"]}
    cloud = Cloud(cfg, ctx.args, ctx)
    remember_region(cloud.account, cloud.region)

    res = {"cloud": cloud.info()}
    res["bucket"] = cloud.ensure_bucket()
    res["role"] = cloud.ensure_role()
    res["log_group"] = cloud.ensure_log_group(lcfg["log_retention_days"])

    prefix = cloud.model_prefix(meta["model_sha256"])
    model_key = f"{prefix}/model.joblib"
    res["model_upload"] = cloud.upload(mpath, model_key, meta["model_sha256"], {"version": meta["version"]},
                                       need_version=True)
    meta_path = model_dir(cfg) / "metadata.json"
    res["metadata_upload"] = cloud.upload(meta_path, f"{prefix}/metadata.json", sha256_file(meta_path))
    code_key = f"packages/lambda-{pkg_sha[:12]}.zip"
    res["package_upload"] = cloud.upload(zip_path, code_key, pkg_sha)

    override = getattr(ctx.args, "model_uri_override", None)
    model_uri = override or f"s3://{cloud.bucket}/{model_key}?versionId={res['model_upload']['version_id']}"
    env = {"MODEL_URI": model_uri, "MODEL_VERSION": meta["version"], "MODEL_SHA256": meta["model_sha256"],
           "JOBLIB_MULTIPROCESSING": "0"}
    git = git_info()
    commit = (git["commit"] or "nogit")[:7] + ("+dirty" if git["code_dirty"] else "")
    desc = f"git {commit} model {meta['version']} pkg {pkg_sha[:12]}"
    res["function"] = cloud.deploy_function(lcfg, code_key, env, desc)
    res["env"] = env
    res["cloud"] = cloud.info()
    res["model_uri_override"] = bool(override)
    write_json(ctx.out("results/cloud/deploy_override.json" if override else "results/cloud/deploy.json"), res)
    ctx.state["deployed_version"] = res["function"]["version"]
    ctx.cloud = {**cloud.info(), "function_version": res["function"]["version"],
                 "code_sha256": res["function"]["code_sha256"], "model_uri": model_uri,
                 "package_uri": res["package_upload"]["uri"]}
    ctx.summary = {"function_version": res["function"]["version"], "action": res["function"]["action"],
                   "memory_mb": lcfg["memory_mb"], "timeout_s": lcfg["timeout_s"],
                   "model_uri_override": bool(override)}


def _cloud_invoker(ctx):
    cloud = Cloud(ctx.cfg, ctx.args, ctx)
    version = ctx.state.get("deployed_version") or cloud.latest_version()
    meta, expected = expected_model(ctx.cfg)
    ctx.model = {"name": meta["name"], "version": meta["version"], "sha256": meta["model_sha256"]}
    ctx.cloud = {**cloud.info(), "function_version": version}

    def invoke(payload):
        return cloud.invoke(payload, version)
    return cloud, version, expected, invoke


def step_smoke(ctx) -> None:
    cloud, version, expected, invoke = _cloud_invoker(ctx)
    ctx.add_input(ROOT / "samples" / "service_cases.json")
    identity = cloud.version_identity(version)
    ctx.cloud = {**ctx.cloud, "identity": identity}
    report = run_cases(invoke, expected, f"lambda:{cloud.fn}:{version}", ctx.cfg, ctx.secrets)
    day = sample_day_comparison(report.pop("responses"), ctx.cfg)
    report["sample_day"] = day
    report["function_version"] = version
    report["identity"] = identity
    if day is None:
        report["all_passed"] = False
        report.setdefault("problems", []).append("예시 날짜 비교를 만들지 못함 (prepare 결과 확인)")
    write_json(ctx.out("results/service_check/cloud.json"), report)
    ctx.summary = {"passed": report["passed"], "total": report["total"], "function_version": version}
    if not report["all_passed"]:
        bad = [f"{c['name']}: {c['problems'][:1]}" for c in report["cases"] if not c["passed"]]
        bad += report.get("problems", [])
        raise RuntimeError(f"클라우드 서비스 검사 실패 {len(bad)}건: {bad[:3]}")


def step_cloud_eval(ctx) -> None:
    from .data import evaluation_rows, records

    cfg = ctx.cfg
    cloud, version, expected, invoke = _cloud_invoker(ctx)
    identity = cloud.version_identity(version)
    ctx.cloud = {**ctx.cloud, "identity": identity}
    test_path = ROOT / cfg["data"]["processed_dir"] / "test.csv"
    local_path = RESULTS / "evaluation" / "predictions_test.csv"
    ctx.add_input(test_path)
    ctx.add_input(local_path)
    test, _ = evaluation_rows(pd.read_csv(test_path))
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
    local_metrics = load_json(RESULTS / "evaluation" / "metrics.json")["metrics"]["model"]
    report = {
        "function": cloud.fn, "function_version": version, "region": cloud.region,
        "identity": identity, "model": expected, "rows": int(len(test)), "batches": len(calls), "batch_size": size,
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


def last_status(step: str, identity: dict) -> str | None:
    from . import runlog

    if not runlog.RUN_LOG.exists():
        return None
    status = None
    for line in runlog.RUN_LOG.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("step") == step and (rec.get("cloud") or {}).get("identity") == identity:
            status = rec.get("status")
    return status


def step_release(ctx) -> None:
    cloud = Cloud(ctx.cfg, ctx.args, ctx)
    cloud.require_ours(f"Lambda 함수 {cloud.fn}", cloud.function_tags())
    version = ctx.state.get("deployed_version") or cloud.latest_version()
    current = cloud.version_identity(version)
    recorded = sanitize(current, ctx.secrets)
    missing = [step for step in ("smoke", "cloud-eval") if last_status(step, recorded) != "success"]
    if missing:
        raise RuntimeError(f"{cloud.region} 의 버전 {version} 은 {', '.join(missing)} 의 마지막 실행이 통과가 아닙니다 "
                           "(같은 리전, 리비전, 코드, 모델 경로로 실행한 가장 최근 기록이 성공이어야 합니다)")
    previous = cloud.alias_version()
    alias = cloud.point_alias(version)
    report = {"released_at": now_iso(), **cloud.info(), "function_version": version, "identity": current,
              "previous_live": previous, "alias": alias}
    write_json(ctx.out("results/cloud/release.json"), report)
    ctx.model = {"name": ctx.cfg["model"]["name"], "version": current["model_version"],
                 "sha256": current["model_sha256"]}
    ctx.cloud = {**cloud.info(), "function_version": version, "alias": alias, "model_uri": current["model_uri"]}
    ctx.summary = {"live": version, "previous_live": previous}


def _clouds(ctx) -> list[Cloud]:
    first = Cloud(ctx.cfg, ctx.args, ctx)
    regions = set(known_regions(ctx.cfg, ctx.args, first.account)) | first.bucket_regions()
    if first.configured_region:
        regions.add(first.configured_region)
    others = sorted(r for r in regions if r != first.region)
    return [first] + [Cloud(ctx.cfg, ctx.args, ctx, region=r) for r in others]


def _status_all(clouds: list[Cloud]) -> dict:
    return {"iam_role": clouds[0].role_status(),
            "regions": {c.region: c.regional_status() for c in clouds}}


def step_status(ctx) -> None:
    name = getattr(ctx.args, "status_name", None) or "status"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise RuntimeError(f"--status-name 에는 영문, 숫자, _, - 만 쓸 수 있습니다: {name}")
    clouds = _clouds(ctx)
    resources = _status_all(clouds)
    st = {"checked_at": now_iso(), "regions_checked": [c.region for c in clouds], **clouds[0].info(),
          "other_account_regions": other_account_regions(clouds[0].account), "resources": resources}
    write_json(ctx.out(f"results/cloud/{name}.json"), st)
    ctx.cloud = {**clouds[0].info(), "regions_checked": st["regions_checked"]}
    ctx.summary = {"iam_role": resources["iam_role"].get("exists"),
                   **{f"{r}:{k}": v.get("exists") for r, res in resources["regions"].items() for k, v in res.items()}}
    print(json.dumps(ctx.summary, ensure_ascii=False))


def step_cleanup(ctx) -> None:
    clouds = _clouds(ctx)
    before = _status_all(clouds)
    deleted = [d for c in clouds for d in c.cleanup_regional()]
    if all(d["result"] in ("deleted", "not_found") for d in deleted):
        deleted += clouds[0].cleanup_role()
    else:
        deleted.append({"resource": f"iam:{clouds[0].role}", "result": "kept: 남은 지역 자원이 이 역할을 쓸 수 있어 지우지 않음"})
    after = _status_all(clouds)
    for _ in range(STATUS_RETRIES):
        if all_gone(after):
            break
        time.sleep(STATUS_WAIT_S)
        after = _status_all(clouds)
    others = other_account_regions(clouds[0].account)
    report = {"cleaned_at": now_iso(), "regions_checked": [c.region for c in clouds], **clouds[0].info(),
              "before": before, "actions": deleted, "after": after, "all_gone": all_gone(after),
              "other_account_regions": others}
    write_json(ctx.out("results/cloud/cleanup.json"), report)
    ctx.cloud = {**clouds[0].info(), "regions_checked": report["regions_checked"]}
    ctx.summary = {"all_gone": report["all_gone"], "actions": {d["resource"]: d["result"] for d in deleted},
                   "other_account_regions": others}
    if not report["all_gone"]:
        raise RuntimeError(f"남은 자원이 있거나 확인하지 못했습니다: {after}")
    forget_account(clouds[0].account)
    if others:
        raise RuntimeError(f"이 계정의 자원은 모두 지웠지만, 다른 계정에서 배포한 기록이 남아 있습니다(리전 {others}). "
                           "그 계정의 프로필로 cleanup 을 실행하세요")
