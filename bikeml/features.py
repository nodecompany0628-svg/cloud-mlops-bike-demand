"""요청 한 건(한 시간)을 모델 입력 벡터로 바꾼다.

학습 자료를 만들 때와 Lambda 에서 예측할 때 모두 이 모듈을 쓴다.
같은 코드로 변환해야 학습과 서비스의 입력이 어긋나지 않는다.
표준 라이브러리만 쓴다(Lambda 패키지를 가볍게 유지).
"""

from __future__ import annotations

import math
from datetime import datetime

# 모델에 들어가는 순서. 바꾸면 모델을 다시 학습해야 한다.
FEATURES = [
    "yr",            # 2011 -> 0, 2012 -> 1 (학습 기간 밖은 가까운 쪽으로 맞춤)
    "mnth",          # 1~12
    "hr",            # 0~23
    "weekday",       # 0=일요일 ... 6=토요일 (원자료와 같은 방식)
    "workingday",    # 주말도 공휴일도 아니면 1
    "holiday",       # 공휴일이면 1
    "weathersit",    # 1 맑음 ~ 4 폭우·폭설
    "temp_c",        # 기온 °C
    "feels_like_c",  # 체감 기온 °C
    "humidity",      # 습도 %
    "windspeed_kmh", # 풍속 km/h
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

# 허용 범위. 원자료 범위(기온 -7~39°C, 체감 -16~50°C, 풍속 0~57)보다 조금 넓게 잡고,
# 물리적으로 말이 안 되는 값만 막는다.
NUMERIC_RANGES = {
    "temp_c": (-30.0, 50.0),
    "feels_like_c": (-40.0, 60.0),
    "humidity": (1.0, 100.0),
    "windspeed_kmh": (0.0, 150.0),
}

TRAIN_YEARS = (2011, 2012)


class FieldError(Exception):
    """입력 한 건의 잘못된 필드."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_datetime(value) -> datetime:
    if not isinstance(value, str):
        raise FieldError("datetime", "문자열이어야 합니다 (예: 2012-10-01T08:00)")
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        raise FieldError("datetime", f"날짜·시각 형식이 아닙니다: {value!r} (예: 2012-10-01T08:00)") from None


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
    v = float(value)
    if math.isnan(v) and allow_missing:
        return math.nan
    if not math.isfinite(v):
        raise FieldError(field, "유한한 숫자여야 합니다")
    lo, hi = NUMERIC_RANGES[field]
    if not lo <= v <= hi:
        raise FieldError(field, f"{lo}~{hi} 범위를 벗어났습니다: {v}")
    return v


def to_features(record: dict, allow_missing_humidity: bool = False) -> tuple[list[float], list[str]]:
    """요청 한 건 -> (FEATURES 순서의 값, 경고 목록).

    잘못된 필드가 있으면 FieldError 를 낸다. 여러 필드가 틀렸으면 첫 번째 것만 알린다.
    allow_missing_humidity 는 학습 자료(센서 오류로 습도를 비운 행)에만 쓴다.
    """
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

    lo, hi = TRAIN_YEARS
    year = min(max(dt.year, lo), hi)
    if year != dt.year:
        warnings.append(
            f"{dt.year}년은 학습 기간({lo}~{hi}) 밖이라 {year}년 수준으로 계산했습니다"
        )

    py_weekday = dt.weekday()           # 0=월 ... 6=일
    weekday = (py_weekday + 1) % 7      # 0=일 ... 6=토 (원자료 방식)
    workingday = int(py_weekday < 5 and holiday == 0)

    unknown = sorted(set(record) - set(REQUIRED_FIELDS))
    if unknown:
        warnings.append("알 수 없는 필드는 무시했습니다: " + ", ".join(unknown))

    row = [
        float(year - lo),
        float(dt.month),
        float(dt.hour),
        float(weekday),
        float(workingday),
        float(holiday),
        float(weather),
        temp,
        feels,
        hum,
        wind,
    ]
    return row, warnings
