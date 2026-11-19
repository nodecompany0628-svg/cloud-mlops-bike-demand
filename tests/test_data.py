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
