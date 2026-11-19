from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .common import ROOT, rel, resolve_profile, sha256_file

RUN_LOG = ROOT / "runs" / "run_log.jsonl"
STAGING = ROOT / "results" / "_staging"
FAILED = ROOT / "results" / "failed"

CODE_PATHS = ["bikeml", "lambda_src", "config", "requirements.txt", "requirements-dev.txt", "requirements-lambda.txt", "tests", "samples/service_cases.json"]


def now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _git(*args) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info() -> dict:
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", *CODE_PATHS)
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "code_dirty": bool(dirty) if dirty is not None else None,
    }


def env_info() -> dict:
    versions = {}
    for mod in ("numpy", "scipy", "sklearn", "pandas", "joblib", "boto3"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "libs": versions}


def _path_variants(path: Path) -> list[str]:
    if path == Path(path.anchor):
        return []
    raw = str(path)
    return [raw.replace("\\", "\\" * 4), raw.replace("\\", "\\" * 2), raw, path.as_posix()]


PATH_END = r"(?=[\\/\"'\s:;,)\]>]|$)"


def _redactions(secrets: list) -> list[tuple[re.Pattern, str]]:
    pats = [(re.compile(r"(?<!\d)" + re.escape(s) + r"(?!\d)"), "<ACCOUNT_ID>")
            for s in secrets if isinstance(s, str) and s]
    for path, repl in ((ROOT, "<REPO>"), (Path.home(), "~")):
        pats += [(re.compile(re.escape(v) + PATH_END, re.IGNORECASE), repl) for v in _path_variants(path)]
    return pats


PROFILE_IN_ERROR = re.compile(r"(profile \()[^)]*(\))")
PROFILE_QUOTED = re.compile(r"(profile \")[^\"]*(\")")
PROFILE_LIST = re.compile(r"(profiles: \[)[^\]]*(\])")
ARN_ACCOUNT = re.compile(r"(arn:aws[\w-]*:[\w-]*:[\w-]*:)\d{12}")
PRINCIPAL = re.compile(r"((?:user|assumed-role|federated-user)/)[^\s\\\"',;)]+")


def _user_pattern() -> re.Pattern | None:
    name = Path.home().name
    return re.compile(r"(Users[\\/]+)" + re.escape(name) + PATH_END, re.IGNORECASE) if name else None


def sanitize(obj, secrets: list[str]):
    if isinstance(obj, str):
        for pat, repl in _redactions(secrets):
            obj = pat.sub(lambda _m, r=repl: r, obj)
        obj = ARN_ACCOUNT.sub(r"\1<ACCOUNT_ID>", obj)
        obj = PRINCIPAL.sub(r"\1<NAME>", obj)
        obj = PROFILE_IN_ERROR.sub(r"\1<PROFILE>\2", obj)
        obj = PROFILE_QUOTED.sub(r"\1<PROFILE>\2", obj)
        obj = PROFILE_LIST.sub(r"\1<PROFILE>\2", obj)
        for s in secrets:
            if isinstance(s, tuple) and s[0] == "profile" and s[1]:
                obj = re.sub(r"(profile\s+)" + re.escape(s[1]) + r"(?![\w-])", r"\1<PROFILE>", obj)
        pat = _user_pattern()
        return pat.sub(r"\1<USER>", obj) if pat else obj
    if isinstance(obj, dict):
        return {k: sanitize(v, secrets) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(sanitize(v, secrets) for v in obj)
    return obj


@dataclass
class StepContext:
    run_id: str
    step: str
    cfg: dict
    args: object
    stage: Path
    inputs: list = field(default_factory=list)
    configs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    model: dict | None = None
    cloud: dict | None = None
    summary: dict = field(default_factory=dict)
    secrets: list = field(default_factory=list)
    state: dict = field(default_factory=dict)

    def add_input(self, path) -> None:
        self.inputs.append({"path": rel(path), "sha256": sha256_file(path)})

    def add_config(self, path) -> None:
        self.configs.append({"path": rel(path), "sha256": sha256_file(path)})

    def out(self, relpath: str) -> Path:
        p = self.stage / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        if relpath not in self.outputs:
            self.outputs.append(relpath)
        return p


TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".csv", ".md"}


def _scrub(stage: Path, secrets: list[str]) -> list[str]:
    unreadable = []
    for f in sorted(stage.rglob("*")):
        if f.relative_to(stage).parts[:1] == ("data",):
            continue
        if f.is_file() and f.suffix in TEXT_SUFFIXES:
            try:
                text = f.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                unreadable.append(f.relative_to(stage).as_posix())
                continue
            clean = sanitize(text, secrets)
            if clean != text:
                f.write_bytes(clean.encode("utf-8"))
    return unreadable


def _promote(stage: Path) -> None:
    moves = [(src, ROOT / src.relative_to(stage)) for src in sorted(stage.rglob("*")) if src.is_file()]
    done = []
    try:
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            backup = dst.with_name(dst.name + ".promote-bak") if dst.exists() else None
            if backup:
                os.replace(dst, backup)
            done.append((dst, backup))
            shutil.copy2(src, dst)
    except BaseException:
        for dst, backup in reversed(done):
            if dst.exists():
                dst.unlink()
            if backup:
                os.replace(backup, dst)
        raise
    for _, backup in done:
        if backup and backup.exists():
            backup.unlink()


def append_record(record: dict) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_steps(step_names: list[str], registry: dict, cfg: dict, args, config_path: Path) -> int:
    run_id = new_run_id()
    git = git_info()
    env = env_info()
    shared_state: dict = {}
    print(f"run_id={run_id} commit={(git['commit'] or 'none')[:7]} code_dirty={git['code_dirty']}")
    if git["code_dirty"]:
        print("  주의: 커밋하지 않은 코드 변경이 있습니다. 기록의 커밋과 실제 코드가 다를 수 있습니다.")

    secrets: list = []
    profile = resolve_profile(cfg, getattr(args, "profile", None))
    if profile:
        secrets.append(("profile", profile))
    for i, name in enumerate(step_names):
        stage = STAGING / run_id / name
        stage.mkdir(parents=True, exist_ok=True)
        ctx = StepContext(run_id=run_id, step=name, cfg=cfg, args=args, stage=stage,
                          secrets=secrets, state=shared_state)
        ctx.add_config(config_path)
        started = now_iso()
        t0 = time.perf_counter()
        error = None
        interrupted = False
        try:
            registry[name](ctx)
            unreadable = _scrub(stage, secrets)
            if unreadable:
                raise RuntimeError(f"UTF-8 이 아닌 결과 파일이 있어 기록을 가리지 못했습니다: {unreadable}")
            _promote(stage)
            status = "success"
            result_location = ctx.outputs
        except (Exception, KeyboardInterrupt) as e:
            interrupted = isinstance(e, KeyboardInterrupt)
            status = "failed"
            error = "KeyboardInterrupt: 사용자가 중단함" if interrupted else f"{type(e).__name__}: {e}"
            (stage / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            for bad in _scrub(stage, secrets):
                (stage / bad).unlink()
            dest = FAILED / f"{run_id}_{name}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stage), str(dest))
            result_location = [rel(dest)]
        shutil.rmtree(stage, ignore_errors=True)

        record = {
            "run_id": run_id,
            "step": name,
            "status": status,
            "started_at": started,
            "finished_at": now_iso(),
            "duration_s": round(time.perf_counter() - t0, 2),
            "git": git,
            "model": ctx.model,
            "inputs": ctx.inputs,
            "configs": ctx.configs,
            "args": {k: ("<PROFILE>" if k == "profile" and v else v)
                     for k, v in vars(args).items() if k not in ("func",)},
            "cloud": ctx.cloud,
            "result_location": result_location,
            "summary": ctx.summary,
            "error": error,
            "env": env,
        }
        append_record(sanitize(record, secrets))
        mark = "OK " if status == "success" else "FAIL"
        print(f"[{mark}] {name} ({record['duration_s']}s) -> {', '.join(result_location) or '-'}")
        if error:
            print(f"       {sanitize(error, secrets)}")
            for skipped in step_names[i + 1:]:
                append_record(sanitize({
                    "run_id": run_id, "step": skipped, "status": "skipped",
                    "started_at": None, "finished_at": now_iso(), "git": git,
                    "error": f"이전 단계 실패로 실행하지 않음: {name}",
                }, secrets))
                print(f"[SKIP] {skipped}")
            _cleanup_staging(run_id)
            return 130 if interrupted else 1
    _cleanup_staging(run_id)
    return 0


def _cleanup_staging(run_id: str) -> None:
    shutil.rmtree(STAGING / run_id, ignore_errors=True)
    try:
        STAGING.rmdir()
    except OSError:
        pass


def read_run_log() -> tuple[list[dict], int]:
    if not RUN_LOG.exists():
        return [], 0
    records, broken = [], 0
    for line in RUN_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            broken += 1
    return records, broken


def print_runs(limit: int = 30) -> None:
    if not RUN_LOG.exists():
        print("실행 기록이 없습니다")
        return
    rows, _ = read_run_log()
    print(f"{'run_id':24} {'step':12} {'status':8} {'commit':8} {'model':8} {'finished_at':26} error")
    for r in rows[-limit:]:
        commit = ((r.get("git") or {}).get("commit") or "")[:7]
        model = (r.get("model") or {}).get("version") or ""
        err = (r.get("error") or "")[:60]
        print(f"{r['run_id']:24} {r['step']:12} {r['status']:8} {commit:8} {model:8} {r.get('finished_at') or '':26} {err}")
