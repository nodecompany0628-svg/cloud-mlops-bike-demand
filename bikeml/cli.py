from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import cloud, data, evaluate, package, servicecheck, train
from .common import DEFAULT_CONFIG, load_config
from .runlog import print_runs, run_steps

STEPS = {
    "fetch": data.step_fetch,
    "prepare": data.step_prepare,
    "train": train.step_train,
    "evaluate": evaluate.step_evaluate,
    "check-local": servicecheck.step_check_local,
    "package": package.step_package,
    "deploy": cloud.step_deploy,
    "smoke": cloud.step_smoke,
    "cloud-eval": cloud.step_cloud_eval,
    "release": cloud.step_release,
    "status": cloud.step_status,
    "cleanup": cloud.step_cleanup,
}
GROUPS = {
    "local": ["fetch", "prepare", "train", "evaluate", "check-local"],
    "cloud": ["package", "deploy", "smoke", "cloud-eval", "release"],
}
GROUPS["all"] = GROUPS["local"] + GROUPS["cloud"]


def expand(names: list[str]) -> list[str]:
    out = []
    for n in names:
        steps = GROUPS.get(n, [n])
        for s in steps:
            if s not in STEPS:
                raise SystemExit(f"알 수 없는 단계: {s} (가능: {', '.join(list(STEPS) + list(GROUPS))})")
            out.append(s)
    return out


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(prog="python -m bikeml", description="자전거 대여량 예측 MLOps 파이프라인")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="파이프라인 설정 파일")
    p.add_argument("--profile", default=None, help="AWS 프로필 (기본: AWS_PROFILE 환경 변수 또는 기본 자격 증명)")
    p.add_argument("--region", default=None, help="AWS 리전 (기본: 설정 파일의 ap-northeast-2)")
    p.add_argument("--lambda-config", default=None, help="Lambda 설정 파일 (기본: config/lambda.json)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="단계를 순서대로 실행 (실패하면 멈춤)")
    r.add_argument("steps", nargs="+", help="단계 또는 묶음: local, cloud, all, " + ", ".join(STEPS))
    r.add_argument("--model-uri-override", default=None,
                   help="deploy 에서 Lambda 의 MODEL_URI 를 이 값으로 바꾼다 (오류 재현용)")
    r.add_argument("--status-name", default=None, help="status 결과 파일 이름 (results/cloud/<이름>.json)")

    sub.add_parser("status", help="AWS 자원 상태 조회 (= run status)")
    sub.add_parser("cleanup", help="AWS 자원 삭제 후 상태 재조회 (= run cleanup)")
    rl = sub.add_parser("runs", help="실행 기록 보기")
    rl.add_argument("-n", type=int, default=30)

    args = p.parse_args(argv)
    if args.lambda_config:
        args.lambda_config = str(Path(args.lambda_config).resolve())
    if args.cmd == "runs":
        print_runs(args.n)
        return 0
    if args.cmd in ("status", "cleanup"):
        steps = [args.cmd]
        args.model_uri_override = None
        args.status_name = None
    else:
        steps = expand(args.steps)
    cfg = load_config(args.config)
    return run_steps(steps, STEPS, cfg, args, Path(args.config).resolve())
