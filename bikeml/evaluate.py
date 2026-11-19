from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .common import ROOT, load_json, model_dir, sha256_file, write_json
from .data import TARGET, feature_matrix
from .features import FEATURES

SANDY_DAYS = ("2012-10-29", "2012-10-30")


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": round(float(np.mean(np.abs(err))), 2),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "bias": round(float(np.mean(err)), 2),
    }


def daily_mae(dt: pd.Series, y_true, y_pred) -> float:
    d = pd.DataFrame({"day": dt.str.slice(0, 10), "y": y_true, "p": y_pred}).groupby("day").sum()
    return round(float(np.mean(np.abs(d.p - d.y))), 1)


def load_model(cfg):
    mdir = model_dir(cfg)
    meta = load_json(mdir / "metadata.json")
    path = mdir / "model.joblib"
    sha = sha256_file(path)
    if sha != meta["model_sha256"]:
        raise ValueError(f"모델 파일 sha256 이 metadata 와 다릅니다 ({sha[:12]}... != {meta['model_sha256'][:12]}...)")
    return joblib.load(path), meta, path


def baselines(train: pd.DataFrame, X_train: np.ndarray, X_test: np.ndarray) -> dict:
    y = train[TARGET].to_numpy(dtype=float)
    hr_i, wd_i = FEATURES.index("hr"), FEATURES.index("workingday")
    table = pd.DataFrame({"hr": X_train[:, hr_i], "wd": X_train[:, wd_i], "y": y}).groupby(["hr", "wd"]).y.mean()
    keys = list(zip(X_test[:, hr_i], X_test[:, wd_i], strict=True))
    return {
        "mean": np.full(len(X_test), y.mean()),
        "hour_mean": np.array([table.get(k, y.mean()) for k in keys]),
    }


def step_evaluate(ctx) -> None:
    cfg = ctx.cfg
    pdir = ROOT / cfg["data"]["processed_dir"]
    model, meta, mpath = load_model(cfg)
    ctx.model = {"name": meta["name"], "version": meta["version"], "sha256": meta["model_sha256"]}
    for p in (pdir / "train.csv", pdir / "test.csv", mpath):
        ctx.add_input(p)

    train = pd.read_csv(pdir / "train.csv")
    test = pd.read_csv(pdir / "test.csv")
    X_train = np.asarray(feature_matrix(train), dtype=float)
    X_test = np.asarray(feature_matrix(test), dtype=float)
    y = test[TARGET].to_numpy(dtype=float)

    pred = np.round(np.clip(model.predict(X_test), 0, None), 2)
    base = baselines(train, X_train, X_test)

    methods = {"model": pred, **{f"baseline_{k}": v for k, v in base.items()}}
    result = {}
    not_sandy = ~test.datetime.str.slice(0, 10).isin(SANDY_DAYS).to_numpy()
    for name, p in methods.items():
        m = metrics(y, p)
        m["daily_total_mae"] = daily_mae(test.datetime, y, p)
        m["excluding_sandy"] = metrics(y[not_sandy], p[not_sandy])
        result[name] = m

    crit = cfg["evaluation"]["criteria"]
    mae = result["model"]["mae"]
    reduction = 1 - mae / result["baseline_hour_mean"]["mae"]
    checks = {
        "beats_all_baselines": all(mae < result[k]["mae"] for k in result if k.startswith("baseline_")),
        "mae_reduction_vs_hour_baseline": round(float(reduction), 3),
        "meets_reduction": bool(reduction >= crit["min_mae_reduction_vs_hour_baseline"]),
    }
    passed = (checks["beats_all_baselines"] or not crit["must_beat_all_baselines"]) and checks["meets_reduction"]

    out = test[["datetime", "holiday", "weather", "temp_c", "humidity"]].copy()
    out["actual"] = y.astype(int)
    out["pred_model"] = pred
    out["pred_baseline_mean"] = np.round(base["mean"], 2)
    out["pred_baseline_hour_mean"] = np.round(base["hour_mean"], 2)
    out["abs_error_model"] = np.round(np.abs(pred - y), 2)
    out.to_csv(ctx.out("results/evaluation/predictions_test.csv"), index=False, lineterminator="\n")
    out.sort_values("abs_error_model", ascending=False).head(15).to_csv(
        ctx.out("results/evaluation/worst_errors.csv"), index=False, lineterminator="\n")

    worst = out.loc[out.abs_error_model.idxmax()]
    wday = pd.Timestamp(worst.datetime[:10])
    near = out[(out.datetime.str.slice(11, 16) == worst.datetime[11:16])
               & (pd.to_datetime(out.datetime.str.slice(0, 10)) - wday).abs().le(pd.Timedelta(days=7))].copy()
    near["weekday"] = pd.to_datetime(near.datetime.str.slice(0, 10)).dt.day_name().str[:3]
    near[["datetime", "weekday", "holiday", "weather", "temp_c", "actual", "pred_model",
          "pred_baseline_hour_mean", "abs_error_model"]].to_csv(
        ctx.out("results/evaluation/worst_case_context.csv"), index=False, lineterminator="\n")

    day = out.assign(day=out.datetime.str.slice(0, 10)).groupby("day")[
        ["actual", "pred_model", "pred_baseline_hour_mean"]].sum().round(1)
    day["error_model"] = (day.pred_model - day.actual).round(1)
    day.reset_index().to_csv(ctx.out("results/evaluation/daily_totals.csv"), index=False, lineterminator="\n")

    by_month = out.assign(m=out.datetime.str.slice(0, 7)).groupby("m").apply(
        lambda g: pd.Series({"n": len(g), "actual_mean": round(g.actual.mean(), 1),
                             "mae_model": round(g.abs_error_model.mean(), 2),
                             "mae_hour_baseline": round((g.pred_baseline_hour_mean - g.actual).abs().mean(), 2)}),
        include_groups=False)

    report = {
        "model": ctx.model,
        "eval_data": {"path": "data/processed/test.csv", "sha256": sha256_file(pdir / "test.csv"),
                      "rows": int(len(test)),
                      "first": test.datetime.min(), "last": test.datetime.max(),
                      "actual_mean": round(float(y.mean()), 2)},
        "unit": "대/시간 (daily_total_mae 는 대/일)",
        "metrics": result,
        "by_month": by_month.reset_index().astype({"n": int}).to_dict(orient="records"),
        "criteria": crit,
        "checks": checks,
        "passed": bool(passed),
    }
    write_json(ctx.out("results/evaluation/metrics.json"), report)
    ctx.summary = {"model_mae": mae, "baseline_hour_mean_mae": result["baseline_hour_mean"]["mae"],
                   "baseline_mean_mae": result["baseline_mean"]["mae"], "passed": bool(passed)}
    if not passed:
        raise RuntimeError(f"합격 기준 미달: {checks}")
