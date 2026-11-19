import math

import pytest

from bikeml.features import FEATURES, FieldError, to_features

GOOD = {"datetime": "2012-10-01T08:00", "holiday": False, "weather": 1, "temp_c": 17.5,
        "feels_like_c": 21.0, "humidity": 72, "windspeed_kmh": 9.0}


def feat(rec):
    row, warnings = to_features(rec)
    return dict(zip(FEATURES, row)), warnings


def test_calendar_values_follow_dataset_convention():
    f, w = feat(GOOD)  # 2012-10-01 은 월요일
    assert (f["yr"], f["mnth"], f["hr"], f["weekday"], f["workingday"], f["holiday"]) == (1, 10, 8, 1, 1, 0)
    assert w == []


def test_saturday_and_holiday():
    f, _ = feat({**GOOD, "datetime": "2011-01-01T00:00"})  # 토요일 = 6 (원자료와 같음)
    assert f["weekday"] == 6 and f["workingday"] == 0
    f, _ = feat({**GOOD, "holiday": True})
    assert f["holiday"] == 1 and f["workingday"] == 0


def test_year_outside_training_is_clipped_with_warning():
    f, w = feat({**GOOD, "datetime": "2026-05-05T18:00"})
    assert f["yr"] == 1 and any("학습 기간" in m for m in w)


@pytest.mark.parametrize("field,value", [
    ("datetime", "2012-10-01T25:00"),
    ("datetime", 20121001),
    ("weather", 5),
    ("weather", True),
    ("temp_c", "17.5"),
    ("humidity", 0),
    ("humidity", 130),
    ("windspeed_kmh", float("nan")),
    ("holiday", "no"),
])
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(FieldError) as e:
        to_features({**GOOD, field: value})
    assert e.value.field == field


def test_missing_field():
    rec = dict(GOOD)
    del rec["feels_like_c"]
    with pytest.raises(FieldError) as e:
        to_features(rec)
    assert e.value.field == "feels_like_c"


def test_unknown_field_is_ignored_with_warning():
    a, _ = to_features(GOOD)
    b, w = to_features({**GOOD, "cnt": 999})
    assert a == b and any("cnt" in m for m in w)


def test_missing_humidity_allowed_only_for_training():
    row, _ = to_features({**GOOD, "humidity": float("nan")}, allow_missing_humidity=True)
    assert math.isnan(row[FEATURES.index("humidity")])
