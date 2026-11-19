import pandas as pd
import pytest

from bikeml.common import ROOT, load_config
from bikeml.data import (
    check_derived,
    feature_matrix,
    load_raw,
    quality_checks,
    split,
    to_requests,
)

CFG = load_config()
RAW = ROOT / CFG["data"]["raw_dir"] / "hour.csv"
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="data/raw/hour.csv 없음 (python -m bikeml run fetch)")


def test_quality_and_derived_features():
    raw = load_raw(CFG)
    q = quality_checks(raw)
    assert q["rows"] == 17379 and q["hum_zero_rows"] == 22
    df = to_requests(raw)
    assert df.humidity.isna().sum() == 22
    assert set(check_derived(raw, feature_matrix(df)).values()) == {0}


def test_split_is_chronological_and_complete():
    df = to_requests(load_raw(CFG))
    train, test = split(df, CFG)
    assert len(train) + len(test) == len(df)
    assert train.datetime.max() < test.datetime.min()
    assert "casual" not in train.columns and "registered" not in train.columns


def test_unit_conversion_ranges():
    df = to_requests(load_raw(CFG))
    assert df.temp_c.between(-8, 39).all()
    assert df.feels_like_c.between(-16, 50).all()
    assert df.windspeed_kmh.between(0, 67).all()
    assert pd.Timestamp(df.datetime.iloc[0]) == pd.Timestamp("2011-01-01 00:00")


def test_split_problems_detects_changed_periods():
    from bikeml.data import split_problems
    df = to_requests(load_raw(CFG))
    train, test = split(df, CFG)
    assert split_problems(CFG, train, test) == []
    moved = {**CFG, "split": {**CFG["split"], "train_end": "2012-06-30", "test_start": "2012-07-01"}}
    assert split_problems(moved, train, test) == ["학습 자료 기간"]
    shorter = {**CFG, "split": {**CFG["split"], "test_end": "2012-11-30"}}
    assert split_problems(shorter, train, test) == ["평가 자료 기간"]


def test_model_mismatches_detects_each_change():
    import json

    from bikeml.common import model_dir
    from bikeml.data import model_mismatches
    meta = json.loads((model_dir(CFG) / "metadata.json").read_text(encoding="utf-8"))
    assert model_mismatches(meta, CFG) == []
    assert model_mismatches({**meta, "params": {**meta["params"], "max_iter": 5}}, CFG) == ["하이퍼파라미터"]
    assert model_mismatches({**meta, "version": "other"}, CFG) == ["모델 버전"]
    assert model_mismatches({**meta, "feature_code_sha256": "0"}, CFG) == ["특성 코드"]
    assert model_mismatches({**meta, "train_data": {"sha256": "0"}}, CFG) == ["학습 자료"]


def test_evaluation_rows_drop_missing_humidity():
    from bikeml.data import evaluation_rows
    df = to_requests(load_raw(CFG))
    kept, dropped = evaluation_rows(df)
    assert dropped == 22 and kept.humidity.notna().all() and len(kept) == len(df) - 22
