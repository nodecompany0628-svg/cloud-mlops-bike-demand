from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from .common import ROOT, rel, sha256_file, write_json
from .data import FEATURE_CODE, TARGET, feature_matrix, split_problems
from .features import FEATURES
from .runlog import now_iso


def build_model(params: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**params)


def step_train(ctx) -> None:
    cfg = ctx.cfg
    mcfg = cfg["model"]
    train_path = ROOT / cfg["data"]["processed_dir"] / "train.csv"
    ctx.add_input(train_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(ROOT / cfg["data"]["processed_dir"] / "test.csv")
    bad = split_problems(cfg, train, test)
    if bad:
        raise RuntimeError(f"{', '.join(bad)}이 config 의 split 과 다릅니다. prepare 를 먼저 실행하세요")

    X = np.asarray(feature_matrix(train), dtype=float)
    y = train[TARGET].to_numpy(dtype=float)

    model = build_model(mcfg["params"])
    t0 = time.perf_counter()
    model.fit(X, y)
    fit_s = time.perf_counter() - t0
    train_mae = float(np.mean(np.abs(model.predict(X) - y)))

    mdir = f"models/_candidate/{mcfg['version']}"
    model_path = ctx.out(f"{mdir}/model.joblib")
    joblib.dump(model, model_path, compress=3)
    sha = sha256_file(model_path)

    meta = {
        "name": mcfg["name"],
        "version": mcfg["version"],
        "created_at": now_iso(),
        "algorithm": "sklearn.ensemble.HistGradientBoostingRegressor",
        "params": mcfg["params"],
        "features": FEATURES,
        "feature_code_sha256": sha256_file(FEATURE_CODE),
        "target": "cnt (시간당 전체 대여 건수, 대/시간)",
        "train_data": {"path": rel(train_path), "sha256": sha256_file(train_path),
                       "rows": int(len(train)), "first": train.datetime.min(), "last": train.datetime.max()},
        "library_versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__,
                             "joblib": joblib.__version__},
        "n_iter": int(model.n_iter_),
        "fit_seconds": round(fit_s, 2),
        "train_mae": round(train_mae, 2),
        "model_file": "model.joblib",
        "model_sha256": sha,
    }
    write_json(ctx.out(f"{mdir}/metadata.json"), meta)
    ctx.model = {"name": mcfg["name"], "version": mcfg["version"], "sha256": sha}
    ctx.summary = {"train_rows": int(len(train)), "train_mae": meta["train_mae"], "fit_seconds": meta["fit_seconds"]}
