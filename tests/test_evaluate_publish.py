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
    weak = build_model({**CFG["model"]["params"], "max_iter": 1})
    weak.fit(np.asarray(feature_matrix(train)), train[TARGET].to_numpy(dtype=float))
    write_candidate(candidate, weak)
    before = sha256_file(published / "model.joblib")

    assert run_evaluate() == 1
    assert sha256_file(published / "model.joblib") == before
    assert not (root / "results" / "evaluation" / "metrics.json").exists()


def test_passing_candidate_is_published(isolated):
    root, published, candidate = isolated
    good = joblib.load(GOOD_MODEL / "model.joblib")
    sha = write_candidate(candidate, good)
    (published / "model.joblib").unlink()

    assert run_evaluate() == 0
    assert sha256_file(published / "model.joblib") == sha
    metrics = json.loads((root / "results" / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["passed"] and metrics["model_source"] == "candidate" and metrics["model"]["sha256"] == sha
