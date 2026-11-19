import argparse
import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from bikeml import evaluate, runlog
from bikeml.common import DEFAULT_CONFIG, ROOT, load_config, model_dir, sha256_file
from bikeml.data import TARGET, feature_matrix
from bikeml.train import build_model

CFG = load_config()
GOOD_MODEL = model_dir(CFG)
TRAIN = ROOT / CFG["data"]["processed_dir"] / "train.csv"
pytestmark = pytest.mark.skipif(not (GOOD_MODEL / "model.joblib").exists() or not TRAIN.exists(),
                                reason="로컬 단계 결과가 없음 (python -m bikeml run local)")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "ROOT", tmp_path)
    monkeypatch.setattr(runlog, "RUN_LOG", tmp_path / "runs" / "run_log.jsonl")
    monkeypatch.setattr(runlog, "STAGING", tmp_path / "results" / "_staging")
    monkeypatch.setattr(runlog, "FAILED", tmp_path / "results" / "failed")
    published = tmp_path / "models" / CFG["model"]["version"]
    candidate = tmp_path / "models" / "_candidate" / CFG["model"]["version"]
    published.mkdir(parents=True)
    candidate.mkdir(parents=True)
    for name in ("model.joblib", "metadata.json"):
        shutil.copy2(GOOD_MODEL / name, published / name)
    monkeypatch.setattr(evaluate, "model_dir", lambda cfg: published)
    monkeypatch.setattr(evaluate, "candidate_dir", lambda cfg: candidate)
    return tmp_path, published, candidate


def write_candidate(candidate, model):
    joblib.dump(model, candidate / "model.joblib", compress=3)
    meta = json.loads((GOOD_MODEL / "metadata.json").read_text(encoding="utf-8"))
    meta["model_sha256"] = sha256_file(candidate / "model.joblib")
    (candidate / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta["model_sha256"]


def run_evaluate():
    args = argparse.Namespace(config=str(DEFAULT_CONFIG), cmd="run", steps=["evaluate"])
    return runlog.run_steps(["evaluate"], {"evaluate": evaluate.step_evaluate}, CFG, args, DEFAULT_CONFIG)


def test_failed_candidate_does_not_replace_published_model(isolated):
    root, published, candidate = isolated
    train = pd.read_csv(TRAIN)
    shuffled = np.random.default_rng(0).permutation(train[TARGET].to_numpy(dtype=float))
    weak = build_model(CFG["model"]["params"])
    weak.fit(np.asarray(feature_matrix(train)), shuffled)
    write_candidate(candidate, weak)
    before = sha256_file(published / "model.joblib")

    assert run_evaluate() == 1
    rec = [json.loads(x) for x in (root / "runs" / "run_log.jsonl").read_text(encoding="utf-8").splitlines()][-1]
    assert "합격 기준 미달" in rec["error"]
    assert sha256_file(published / "model.joblib") == before
    assert not (root / "results" / "evaluation" / "metrics.json").exists()


def test_candidate_whose_file_disagrees_with_metadata_is_not_used(isolated):
    root, published, candidate = isolated
    train = pd.read_csv(TRAIN)
    other = build_model({**CFG["model"]["params"], "max_iter": 1})
    other.fit(np.asarray(feature_matrix(train)), train[TARGET].to_numpy(dtype=float))
    write_candidate(candidate, other)

    assert run_evaluate() == 0
    metrics = json.loads((root / "results" / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_source"] == "published"


def test_passing_candidate_is_published(isolated):
    root, published, candidate = isolated
    good = joblib.load(GOOD_MODEL / "model.joblib")
    sha = write_candidate(candidate, good)
    (published / "model.joblib").unlink()

    assert run_evaluate() == 0
    assert sha256_file(published / "model.joblib") == sha
    metrics = json.loads((root / "results" / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["passed"] and metrics["model_source"] == "candidate" and metrics["model"]["sha256"] == sha


def make_stale(folder, field="train_data"):
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if field == "train_data":
        meta["train_data"]["sha256"] = "0" * 64
    else:
        meta["feature_code_sha256"] = "0" * 64
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def test_stale_candidate_falls_back_to_matching_published_model(isolated):
    root, published, candidate = isolated
    write_candidate(candidate, joblib.load(GOOD_MODEL / "model.joblib"))
    make_stale(candidate)
    before = sha256_file(published / "model.joblib")

    assert run_evaluate() == 0
    metrics = json.loads((root / "results" / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_source"] == "published"
    assert sha256_file(published / "model.joblib") == before


def test_no_matching_model_stops_evaluation(isolated):
    root, published, candidate = isolated
    write_candidate(candidate, joblib.load(GOOD_MODEL / "model.joblib"))
    make_stale(candidate)
    make_stale(published, "feature_code")

    assert run_evaluate() == 1
    rec = [json.loads(x) for x in (root / "runs" / "run_log.jsonl").read_text(encoding="utf-8").splitlines()][-1]
    assert "train" in rec["error"] and "학습 자료" in rec["error"] and "특성 코드" in rec["error"]
    assert not (root / "results" / "evaluation" / "metrics.json").exists()


def test_candidate_with_wrong_sha_falls_back_to_published_model(isolated):
    root, published, candidate = isolated
    shutil.copy2(GOOD_MODEL / "model.joblib", candidate / "model.joblib")
    meta = json.loads((GOOD_MODEL / "metadata.json").read_text(encoding="utf-8"))
    (candidate / "metadata.json").write_text(json.dumps({**meta, "model_sha256": "0" * 64}), encoding="utf-8")

    assert run_evaluate() == 0
    metrics = json.loads((root / "results" / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_source"] == "published"
