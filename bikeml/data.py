from __future__ import annotations

import hashlib
import io
import math
import urllib.request
import zipfile

import pandas as pd

from .common import ROOT, sha256_file, write_json
from .features import FEATURES, to_features

TEMP_MIN, TEMP_MAX = -8.0, 39.0
ATEMP_MIN, ATEMP_MAX = -16.0, 50.0
HUM_SCALE = 100.0
WIND_SCALE = 67.0

REQUEST_COLUMNS = ["datetime", "holiday", "weather", "temp_c", "feels_like_c", "humidity", "windspeed_kmh"]
FEATURE_CODE = ROOT / "bikeml" / "features.py"
TARGET = "cnt"
EXPECTED_ROWS = 17379


def step_fetch(ctx) -> None:
    dcfg = ctx.cfg["data"]
    raw_dir = ROOT / dcfg["raw_dir"]
    expected = dcfg["files"]

    def all_ok() -> bool:
        return all((raw_dir / n).exists() and sha256_file(raw_dir / n) == h for n, h in expected.items())

    downloaded = False
    if not all_ok():
        print(f"  원자료 내려받기: {dcfg['url']}")
        with urllib.request.urlopen(dcfg["url"], timeout=60) as r:
            blob = r.read()
        got = hashlib.sha256(blob).hexdigest()
        if got != dcfg["zip_sha256"]:
            raise RuntimeError(f"zip sha256 불일치: 기대 {dcfg['zip_sha256'][:12]}..., 실제 {got[:12]}...")
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in expected:
                ctx.out(f"{dcfg['raw_dir']}/{name}").write_bytes(z.read(name))
        for name, h in expected.items():
            got = sha256_file(ctx.stage / dcfg["raw_dir"] / name)
            if got != h:
                raise RuntimeError(f"{name} sha256 불일치")
        downloaded = True
    ctx.summary = {"downloaded": downloaded, "verified_files": list(expected), "source": dcfg["url"]}
    for name in expected:
        p = (ctx.stage if downloaded else ROOT) / dcfg["raw_dir"] / name
        ctx.inputs.append({"path": f"{dcfg['raw_dir']}/{name}", "sha256": sha256_file(p)})


def load_raw(cfg) -> pd.DataFrame:
    return pd.read_csv(ROOT / cfg["data"]["raw_dir"] / "hour.csv")


def quality_checks(df: pd.DataFrame) -> dict:
    problems = []
    if len(df) != EXPECTED_ROWS:
        problems.append(f"행 수 {len(df)} != {EXPECTED_ROWS}")
    if df.isna().any().any():
        problems.append("빈 값이 있음")
    if df.duplicated(["dteday", "hr"]).any():
        problems.append("같은 날짜·시각 중복")
    if not (df.cnt == df.casual + df.registered).all():
        problems.append("cnt != casual + registered")
    for c in ("temp", "atemp", "hum", "windspeed"):
        if not df[c].between(0, 1).all():
            problems.append(f"{c} 정규화 범위(0~1) 밖")
    if not df.hr.between(0, 23).all() or not df.weathersit.between(1, 4).all():
        problems.append("hr 또는 weathersit 범위 밖")
    if problems:
        raise ValueError("원자료 점검 실패: " + "; ".join(problems))

    per_day = df.groupby("dteday").size()
    return {
        "rows": int(len(df)),
        "days": int(per_day.size),
        "missing_hours": int(per_day.size * 24 - len(df)),
        "days_with_missing_hours": int((per_day < 24).sum()),
        "hum_zero_rows": int((df.hum == 0).sum()),
        "hum_zero_dates": sorted(df.loc[df.hum == 0, "dteday"].unique().tolist()),
        "windspeed_zero_rows": int((df.windspeed == 0).sum()),
        "weathersit_counts": {str(k): int(v) for k, v in df.weathersit.value_counts().sort_index().items()},
        "cnt_min": int(df.cnt.min()),
        "cnt_max": int(df.cnt.max()),
    }


def to_requests(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "datetime": df.dteday + "T" + df.hr.astype(int).map("{:02d}:00".format),
        "holiday": df.holiday.astype(bool),
        "weather": df.weathersit.astype(int),
        "temp_c": (TEMP_MIN + df.temp * (TEMP_MAX - TEMP_MIN)).round(2),
        "feels_like_c": (ATEMP_MIN + df.atemp * (ATEMP_MAX - ATEMP_MIN)).round(2),
        "humidity": (df.hum * HUM_SCALE).round(1),
        "windspeed_kmh": (df.windspeed * WIND_SCALE).round(2),
    })
    out.loc[out.humidity == 0, "humidity"] = math.nan
    out[TARGET] = df.cnt.astype(int)
    return out


def records(df: pd.DataFrame) -> list[dict]:
    recs = df[REQUEST_COLUMNS].to_dict(orient="records")
    for r in recs:
        r["holiday"] = bool(r["holiday"])
        r["weather"] = int(r["weather"])
    return recs


def feature_matrix(df: pd.DataFrame, allow_missing_humidity: bool = True) -> list[list[float]]:
    return [to_features(r, allow_missing_humidity=allow_missing_humidity)[0] for r in records(df)]


def check_derived(raw: pd.DataFrame, X: list[list[float]]) -> dict:
    fx = pd.DataFrame(X, columns=FEATURES)
    pairs = {"yr": "yr", "mnth": "mnth", "hr": "hr", "weekday": "weekday",
             "workingday": "workingday", "holiday": "holiday", "weathersit": "weathersit"}
    mismatches = {f: int((fx[f].values != raw[c].values).sum()) for f, c in pairs.items()}
    if any(mismatches.values()):
        raise ValueError(f"날짜에서 계산한 값이 원자료와 다릅니다: {mismatches}")
    return mismatches


def split(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = cfg["split"]
    day = df.datetime.str.slice(0, 10)
    train = df[(day >= s["train_start"]) & (day <= s["train_end"])]
    test = df[(day >= s["test_start"]) & (day <= s["test_end"])]
    if len(train) + len(test) != len(df):
        raise ValueError("분할 기간이 전체 자료를 덮지 않습니다")
    if train.datetime.max() >= test.datetime.min():
        raise ValueError("학습 자료가 평가 자료보다 늦은 시각을 포함합니다")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_problems(cfg: dict, train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    s = cfg["split"]
    out = []
    tr, te = train.datetime.str.slice(0, 10), test.datetime.str.slice(0, 10)
    if tr.empty or tr.min() < s["train_start"] or tr.max() > s["train_end"]:
        out.append("학습 자료 기간")
    if te.empty or te.min() < s["test_start"] or te.max() > s["test_end"]:
        out.append("평가 자료 기간")
    return out


def estimator_mismatch(model, cfg: dict) -> bool:
    params = model.get_params()
    return any(params.get(k) != v for k, v in cfg["model"]["params"].items())


def evaluation_rows(test: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    keep = test.humidity.notna()
    return test[keep].reset_index(drop=True), int((~keep).sum())


def model_mismatches(meta: dict, cfg: dict) -> list[str]:
    train_path = ROOT / cfg["data"]["processed_dir"] / "train.csv"
    out = []
    if (meta.get("train_data") or {}).get("sha256") != sha256_file(train_path):
        out.append("학습 자료")
    if meta.get("params") != cfg["model"]["params"]:
        out.append("하이퍼파라미터")
    if meta.get("version") != cfg["model"]["version"]:
        out.append("모델 버전")
    if meta.get("features") != FEATURES or meta.get("feature_code_sha256") != sha256_file(FEATURE_CODE):
        out.append("특성 코드")
    return out


def _part_summary(part: pd.DataFrame) -> dict:
    return {
        "rows": int(len(part)),
        "first": part.datetime.min(),
        "last": part.datetime.max(),
        "cnt_mean": round(float(part[TARGET].mean()), 2),
        "cnt_median": float(part[TARGET].median()),
        "humidity_missing": int(part.humidity.isna().sum()),
    }


def step_prepare(ctx) -> None:
    cfg = ctx.cfg
    raw_path = ROOT / cfg["data"]["raw_dir"] / "hour.csv"
    ctx.add_input(raw_path)
    raw = load_raw(cfg)
    quality = quality_checks(raw)

    df = to_requests(raw)
    X = feature_matrix(df)
    mismatches = check_derived(raw, X)

    train, test = split(df, cfg)
    pdir = cfg["data"]["processed_dir"]
    train.to_csv(ctx.out(f"{pdir}/train.csv"), index=False, lineterminator="\n")
    test.to_csv(ctx.out(f"{pdir}/test.csv"), index=False, lineterminator="\n")

    day = cfg["evaluation"]["sample_day"]
    day_df = test[test.datetime.str.startswith(day)]
    write_json(ctx.out(f"samples/generated/request_{day}.json"), {"instances": records(day_df)})
    write_json(ctx.out(f"samples/generated/actual_{day}.json"),
               {"datetime": day_df.datetime.tolist(), "cnt": day_df[TARGET].astype(int).tolist()})

    report = {
        "source_file": "data/raw/hour.csv",
        "quality": quality,
        "unit_conversion": {
            "temp_c": f"{TEMP_MIN} + temp * ({TEMP_MAX} - {TEMP_MIN})",
            "feels_like_c": f"{ATEMP_MIN} + atemp * ({ATEMP_MAX} - {ATEMP_MIN})",
            "humidity": f"hum * {HUM_SCALE}",
            "windspeed_kmh": f"windspeed * {WIND_SCALE}",
        },
        "cleaning": {
            "humidity_zero_to_missing": quality["hum_zero_rows"],
            "missing_hours_not_imputed": quality["missing_hours"],
            "windspeed_zero_kept": quality["windspeed_zero_rows"],
        },
        "excluded_columns": {
            "casual, registered": "둘을 더하면 정답(cnt)이라 정답이 새어 들어감",
            "instant": "시간 순서를 나타내는 행 번호이고 서비스 입력에 없음",
            "season": "월(mnth)로 대신함, 원자료 정의(1=봄)가 실제 달력과 어긋남",
            "dteday, yr, mnth, weekday, workingday": "요청의 datetime, holiday 로 다시 계산(원자료 값과 일치 확인)",
        },
        "derived_feature_mismatches": mismatches,
        "split": {"rule": cfg["split"], "train": _part_summary(train), "test": _part_summary(test)},
    }
    write_json(ctx.out("results/data/prepare_report.json"), report)
    ctx.summary = {"train_rows": len(train), "test_rows": len(test),
                   "humidity_zero_to_missing": quality["hum_zero_rows"]}
