import math

import pytest

from bikeml.features import FEATURES, FieldError, to_features

GOOD = {"datetime": "2012-10-01T08:00", "holiday": False, "weather": 1, "temp_c": 17.5,
        "feels_like_c": 21.0, "humidity": 72, "windspeed_kmh": 9.0}


def feat(rec):
    row, warnings = to_features(rec)
    return dict(zip(FEATURES, row, strict=True)), warnings


def test_calendar_values_follow_dataset_convention():
    f, w = feat(GOOD)
    assert (f["yr"], f["mnth"], f["hr"], f["weekday"], f["workingday"], f["holiday"]) == (1, 10, 8, 1, 1, 0)
    assert w == []


def test_saturday_and_holiday():
    f, _ = feat({**GOOD, "datetime": "2011-01-01T00:00"})
    assert f["weekday"] == 6 and f["workingday"] == 0
    f, _ = feat({**GOOD, "holiday": True})
    assert f["holiday"] == 1 and f["workingday"] == 0


def test_year_outside_training_is_clipped_with_warning():
    f, w = feat({**GOOD, "datetime": "2026-05-05T18:00"})
    assert f["yr"] == 1 and any("학습 기간" in m for m in w)


@pytest.mark.parametrize("field,value", [
    ("datetime", "2012-10-01T25:00"),
    ("datetime", 20121001),
    ("datetime", "2012-10-01"),
    ("datetime", "20121001"),
    ("datetime", "2012-W40-1"),
    ("temp_c", 10 ** 400),
    ("temp_c", None),
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


def test_space_separator_and_timezone_warning():
    f, w = feat({**GOOD, "datetime": "2012-10-01 08:00"})
    assert f["hr"] == 8 and w == []
    f, w = feat({**GOOD, "datetime": "2012-10-01T08:00+09:00"})
    assert f["hr"] == 8 and any("시간대" in m for m in w)


def test_minutes_are_dropped_with_warning():
    f, w = feat({**GOOD, "datetime": "2012-10-01T08:45"})
    assert f["hr"] == 8 and any("분·초" in m for m in w)


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
