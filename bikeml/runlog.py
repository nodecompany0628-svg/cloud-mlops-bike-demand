"""실행 기록과 단계 실행기.

단계마다 runs/run_log.jsonl 에 한 줄씩 남긴다:
시각, 성공 여부, Git 커밋, 모델 버전, 입력과 설정의 해시, 결과 위치, 오류.

각 단계는 결과를 results/_staging/<run_id>/<step>/ 에 먼저 쓴다.
성공하면 저장소의 제자리(results/, models/ ...)로 옮기고,
실패하면 results/failed/<run_id>_<step>/ 로 옮겨 정상 결과와 섞이지 않게 한다.
실패한 단계 뒤의 단계는 실행하지 않고 skipped 로 기록한다.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, rel, sha256_file

RUN_LOG = ROOT / "runs" / "run_log.jsonl"
STAGING = ROOT / "results" / "_staging"
FAILED = ROOT / "results" / "failed"

CODE_PATHS = ["bikeml", "lambda_src", "config", "requirements.txt", "requirements-lambda.txt", "tests", "samples/service_cases.json"]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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
        # 코드·설정에 커밋하지 않은 변경이 있으면 true (결과 파일 변경은 보지 않음)
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


def _redactions(secrets: list[str]) -> list[tuple[str, str]]:
    pairs = [(s, "<ACCOUNT_ID>") for s in secrets if s]
    # 공개 기록에 PC 사용자 폴더 이름이 남지 않게 경로를 줄인다(저장소 경로를 먼저).
    pairs += [(str(ROOT), "<REPO>"), (ROOT.as_posix(), "<REPO>"),
              (str(Path.home()), "~"), (Path.home().as_posix(), "~")]
    return pairs


def sanitize(obj, secrets: list[str]):
    """기록에 계정 ID, 로컬 경로 같은 값이 들어가지 않게 바꾼다."""
    if isinstance(obj, str):
        for s, repl in _redactions(secrets):
            obj = obj.replace(s, repl)
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v, secrets) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v, secrets) for v in obj]
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
    secrets: list = field(default_factory=list)  # 기록에서 가릴 값(계정 ID)
    state: dict = field(default_factory=dict)    # 같은 실행의 단계끼리 넘기는 값

    def add_input(self, path) -> None:
        self.inputs.append({"path": rel(path), "sha256": sha256_file(path)})

    def add_config(self, path) -> None:
        self.configs.append({"path": rel(path), "sha256": sha256_file(path)})

    def out(self, relpath: str) -> Path:
        """결과 파일 경로(스테이징 안). 성공하면 저장소의 relpath 로 옮겨진다."""
        p = self.stage / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        if relpath not in self.outputs:
            self.outputs.append(relpath)
        return p


TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".csv", ".md"}


def _scrub(stage: Path, secrets: list[str]) -> None:
    """결과 파일을 제자리로 옮기기 전에 계정 ID, 로컬 경로를 가린다."""
    for f in stage.rglob("*"):
        if f.is_file() and f.suffix in TEXT_SUFFIXES:
            text = f.read_text(encoding="utf-8")
            clean = sanitize(text, secrets)
            if clean != text:
                f.write_text(clean, encoding="utf-8", newline="\n")


def _promote(stage: Path) -> None:
    for src in sorted(stage.rglob("*")):
        if src.is_file():
            dst = ROOT / src.relative_to(stage)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


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

    secrets: list[str] = []
    for i, name in enumerate(step_names):
        stage = STAGING / run_id / name
        stage.mkdir(parents=True, exist_ok=True)
        ctx = StepContext(run_id=run_id, step=name, cfg=cfg, args=args, stage=stage,
                          secrets=secrets, state=shared_state)
        ctx.add_config(config_path)
        started = now_iso()
        t0 = time.perf_counter()
        error = None
        try:
            registry[name](ctx)
            _scrub(stage, secrets)
            _promote(stage)
            status = "success"
            result_location = ctx.outputs
        except Exception as e:  # 단계 실패: 기록하고 다음 단계는 멈춘다
            status = "failed"
            error = f"{type(e).__name__}: {e}"
            (stage / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            _scrub(stage, secrets)
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
            "args": {k: v for k, v in vars(args).items() if k not in ("func",)},
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
            return 1
    _cleanup_staging(run_id)
    return 0


def _cleanup_staging(run_id: str) -> None:
    shutil.rmtree(STAGING / run_id, ignore_errors=True)
    try:
        STAGING.rmdir()
    except OSError:
        pass


def print_runs(limit: int = 30) -> None:
    if not RUN_LOG.exists():
        print("실행 기록이 없습니다")
        return
    rows = [json.loads(l) for l in RUN_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{'run_id':24} {'step':12} {'status':8} {'commit':8} {'model':8} {'finished_at':26} error")
    for r in rows[-limit:]:
        commit = ((r.get("git") or {}).get("commit") or "")[:7]
        model = (r.get("model") or {}).get("version") or ""
        err = (r.get("error") or "")[:60]
        print(f"{r['run_id']:24} {r['step']:12} {r['status']:8} {commit:8} {model:8} {r.get('finished_at') or '':26} {err}")

