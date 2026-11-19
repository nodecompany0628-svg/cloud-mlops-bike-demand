from __future__ import annotations

import copy
import importlib
import json
import os
import sys
import time

from .common import ROOT, load_json, model_dir, write_json
from .runlog import sanitize

CASES_FILE = ROOT / "samples" / "service_cases.json"


def load_cases() -> list[dict]:
    return load_json(CASES_FILE)["cases"]


def expected_model(cfg: dict) -> tuple[dict, dict]:
    meta = load_json(model_dir(cfg) / "metadata.json")
    return meta, {"version": meta["version"], "sha256": meta["model_sha256"],
                  "sklearn": meta["library_versions"]["scikit-learn"]}


def build_payload(case: dict, by_name: dict, cfg: dict) -> dict:
    if "payload" in case:
        return copy.deepcopy(case["payload"])
    if "payload_file" in case:
        return load_json(ROOT / case["payload_file"].format(sample_day=cfg["evaluation"]["sample_day"]))
    gen = case["generate"]
    one = by_name[gen["repeat_instance_of"]]["payload"]["instances"][0]
    return {"instances": [dict(one) for _ in range(gen["count"])]}


def check_response(case: dict, resp: dict, expected_model: dict, responses: dict) -> list[str]:
    exp = case["expect"]
    problems = []
    status = resp.get("statusCode")
    body = resp.get("body") or {}
    if status != exp["status"]:
        problems.append(f"상태 코드 {status} (기대 {exp['status']}): {json.dumps(body, ensure_ascii=False)[:300]}")
        return problems
    if status == 200:
        preds = body.get("predictions") or []
        if "n" in exp and (body.get("n") != exp["n"] or len(preds) != exp["n"]):
            problems.append(f"건수 {body.get('n')} (기대 {exp['n']})")
        for p in preds:
            if not isinstance(p.get("predicted_count"), int) or p["predicted_count"] < 0 or p["predicted_value"] < 0:
                problems.append(f"예측값이 0 이상 정수가 아님: {p}")
                break
        if body.get("model_version") != expected_model["version"]:
            problems.append(f"모델 버전 {body.get('model_version')} (기대 {expected_model['version']})")
        if body.get("model_sha256") != expected_model["sha256"]:
            problems.append("모델 sha256 이 다름")
        runtime_sklearn = (body.get("runtime") or {}).get("scikit-learn")
        if "sklearn" in expected_model and runtime_sklearn != expected_model["sklearn"]:
            problems.append(f"실행 환경 scikit-learn {runtime_sklearn} (기대 {expected_model['sklearn']})")
        if "warning_contains" in exp and not any(exp["warning_contains"] in w for w in body.get("warnings", [])):
            problems.append(f"경고에 '{exp['warning_contains']}' 없음")
        ref = exp.get("same_prediction_as")
        if ref:
            ref_body = (responses.get(ref) or {}).get("body") or {}
            a = [p["predicted_value"] for p in preds]
            b = [p["predicted_value"] for p in ref_body.get("predictions", [])]
            if a != b:
                problems.append(f"{ref} 와 예측이 다름: {a} != {b}")
    else:
        err = body.get("error") or {}
        if "error_code" in exp and err.get("code") != exp["error_code"]:
            problems.append(f"오류 코드 {err.get('code')} (기대 {exp['error_code']})")
        if "error_field" in exp:
            fields = [d.get("field") for d in err.get("details", [])]
            if exp["error_field"] not in fields:
                problems.append(f"오류 필드 {fields} (기대 {exp['error_field']})")
    return problems


def run_cases(invoke, expected_model: dict, target: str, cfg: dict, secrets=()) -> dict:
    cases = load_cases()
    by_name = {c["name"]: c for c in cases}
    responses, results = {}, []
    for case in cases:
        payload = build_payload(case, by_name, cfg)
        runs = []
        for _ in range(case.get("repeat", 1)):
            resp, meta = sanitize(invoke(payload), list(secrets))
            runs.append((resp, meta))
        resp, meta = runs[0]
        responses[case["name"]] = resp
        problems = check_response(case, resp, expected_model, responses)
        if len(runs) > 1:
            first = json.dumps(runs[0][0].get("body"), sort_keys=True)
            if any(json.dumps(r.get("body"), sort_keys=True) != first for r, _ in runs[1:]):
                problems.append("같은 입력인데 응답이 다름")
        body = resp.get("body") or {}
        results.append({
            "name": case["name"],
            "about": case.get("about"),
            "expect": case["expect"],
            "status": resp.get("statusCode"),
            "passed": not problems,
            "problems": problems,
            "error": body.get("error"),
            "warnings": body.get("warnings"),
            "predictions_head": (body.get("predictions") or [])[:3],
            "meta": meta,
        })
    passed = sum(r["passed"] for r in results)
    runtime = next(((r.get("body") or {}).get("runtime") for r in responses.values()
                    if (r.get("body") or {}).get("runtime")), None)
    return {"target": target, "model": expected_model, "runtime": runtime, "passed": passed,
            "total": len(results), "all_passed": passed == len(results), "cases": results,
            "responses": responses}


def sample_day_comparison(responses: dict, cfg: dict) -> dict | None:
    day = cfg["evaluation"]["sample_day"]
    actual_path = ROOT / "samples" / "generated" / f"actual_{day}.json"
    body = (responses.get("normal_day_24h") or {}).get("body") or {}
    if not actual_path.exists() or not body.get("predictions"):
        return None
    actual = load_json(actual_path)
    if [p["datetime"] for p in body["predictions"]] != actual["datetime"]:
        return None
    rows = []
    for t, a, p in zip(actual["datetime"], actual["cnt"], body["predictions"], strict=True):
        rows.append({"datetime": t, "actual": a, "predicted": p["predicted_count"]})
    mae = sum(abs(r["actual"] - r["predicted"]) for r in rows) / len(rows)
    return {"day": day, "mae": round(mae, 2), "actual_total": sum(r["actual"] for r in rows),
            "predicted_total": sum(r["predicted"] for r in rows), "hours": rows}


class LocalInvoker:
    def __init__(self, cfg: dict):
        mdir = model_dir(cfg)
        meta, self.expected_model = expected_model(cfg)
        os.environ["MODEL_URI"] = str(mdir / "model.joblib")
        os.environ["MODEL_VERSION"] = meta["version"]
        os.environ["MODEL_SHA256"] = meta["model_sha256"]
        src = str(ROOT / "lambda_src")
        if src not in sys.path:
            sys.path.insert(0, src)
        self.fn = importlib.reload(importlib.import_module("lambda_function"))
        self.fn._CACHE.clear()
        self.name = meta["name"]

    def __call__(self, payload: dict):
        event = json.loads(json.dumps(payload))
        t0 = time.perf_counter()
        resp = self.fn.lambda_handler(event, None)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return json.loads(json.dumps(resp)), {"client_ms": ms}


def step_check_local(ctx) -> None:
    inv = LocalInvoker(ctx.cfg)
    ctx.model = {"name": inv.name, "version": inv.expected_model["version"], "sha256": inv.expected_model["sha256"]}
    ctx.add_input(ROOT / "samples" / "service_cases.json")
    report = run_cases(inv, inv.expected_model, "local", ctx.cfg)
    day = sample_day_comparison(report.pop("responses"), ctx.cfg)
    report["sample_day"] = day
    write_json(ctx.out("results/service_check/local.json"), report)
    ctx.summary = {"passed": report["passed"], "total": report["total"],
                   "sample_day_mae": day["mae"] if day else None}
    if not report["all_passed"]:
        bad = [c["name"] for c in report["cases"] if not c["passed"]]
        raise RuntimeError(f"서비스 검사 실패: {bad}")
