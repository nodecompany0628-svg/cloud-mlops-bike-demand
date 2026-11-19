from __future__ import annotations

import math
from datetime import datetime

FEATURES = [
    "yr",
    "mnth",
    "hr",
    "weekday",
    "workingday",
    "holiday",
    "weathersit",
    "temp_c",
    "feels_like_c",
    "humidity",
    "windspeed_kmh",
]

REQUIRED_FIELDS = [
    "datetime",
    "holiday",
    "weather",
    "temp_c",
    "feels_like_c",
    "humidity",
    "windspeed_kmh",
]

NUMERIC_RANGES = {
    "temp_c": (-30.0, 50.0),
    "feels_like_c": (-40.0, 60.0),
    "humidity": (1.0, 100.0),
    "windspeed_kmh": (0.0, 150.0),
}

TRAIN_YEARS = (2011, 2012)
DATE_ONLY_MAX_LEN = 10


class FieldError(Exception):
    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_datetime(value) -> datetime:
    if not isinstance(value, str):
        raise FieldError("datetime", "문자열이어야 합니다 (예: 2012-10-01T08:00)")
    s = value.strip()
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise FieldError("datetime", f"날짜·시각 형식이 아닙니다: {value!r} (예: 2012-10-01T08:00)") from None
    if len(s) <= DATE_ONLY_MAX_LEN:
        raise FieldError("datetime", f"시각이 없습니다: {value!r} (예: 2012-10-01T08:00)")
    return dt


def parse_holiday(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise FieldError("holiday", "true/false 또는 0/1 이어야 합니다")


def parse_weather(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FieldError("weather", "1~4 정수여야 합니다")
    if not 1 <= value <= 4:
        raise FieldError("weather", f"1~4 범위를 벗어났습니다: {value}")
    return value


def parse_number(field: str, value, allow_missing: bool = False) -> float:
    if value is None and allow_missing:
        return math.nan
    if not _is_number(value):
        raise FieldError(field, "숫자여야 합니다")
    try:
        v = float(value)
    except OverflowError:
        raise FieldError(field, "숫자가 너무 큽니다") from None
    if math.isnan(v) and allow_missing:
        return math.nan
    if not math.isfinite(v):
        raise FieldError(field, "유한한 숫자여야 합니다")
    lo, hi = NUMERIC_RANGES[field]
    if not lo <= v <= hi:
        raise FieldError(field, f"{lo}~{hi} 범위를 벗어났습니다: {v}")
    return v


def to_features(record: dict, allow_missing_humidity: bool = False) -> tuple[list[float], list[str]]:
    if not isinstance(record, dict):
        raise FieldError("(instance)", "객체(JSON object)여야 합니다")
    for f in REQUIRED_FIELDS:
        if f not in record:
            raise FieldError(f, "필수 입력이 없습니다")

    warnings: list[str] = []
    dt = parse_datetime(record["datetime"])
    holiday = parse_holiday(record["holiday"])
    weather = parse_weather(record["weather"])
    temp = parse_number("temp_c", record["temp_c"])
    feels = parse_number("feels_like_c", record["feels_like_c"])
    hum = parse_number("humidity", record["humidity"], allow_missing=allow_missing_humidity)
    wind = parse_number("windspeed_kmh", record["windspeed_kmh"])

    if dt.tzinfo is not None:
        warnings.append("시간대 표시는 무시하고 입력한 시각 그대로 계산했습니다(모델은 현지 시각 기준)")
    if dt.minute or dt.second or dt.microsecond:
        warnings.append(f"분·초는 무시하고 {dt.hour}시로 계산했습니다")

    lo, hi = TRAIN_YEARS
    year = min(max(dt.year, lo), hi)
    if year != dt.year:
        warnings.append(f"{dt.year}년은 학습 기간({lo}~{hi}) 밖이라 {year}년 수준으로 계산했습니다")

    sunday_first = (dt.weekday() + 1) % 7
    workingday = int(dt.weekday() < 5 and holiday == 0)

    unknown = sorted(set(record) - set(REQUIRED_FIELDS))
    if unknown:
        warnings.append("알 수 없는 필드는 무시했습니다: " + ", ".join(unknown))

    row = [
        float(year - lo),
        float(dt.month),
        float(dt.hour),
        float(sunday_first),
        float(workingday),
        float(holiday),
        float(weather),
        temp,
        feels,
        hum,
        wind,
    ]
    return row, warnings
