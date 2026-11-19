# 데이터와 모델

## 1. 데이터 출처와 사용 조건

| 항목 | 내용 |
|---|---|
| 이름 | Bike Sharing Dataset (UCI Machine Learning Repository, id 275) |
| 주소 | https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset |
| 내려받은 파일 | `bike+sharing+dataset.zip` (sha256 `b70182d0…845401`) 안의 `hour.csv`, `Readme.txt` |
| 원 출처 | Capital Bikeshare(미국 워싱턴 D.C.) 2011~2012년 대여 기록, 날씨는 freemeteo.com |
| 사용 조건 | CC BY 4.0 (UCI 데이터셋 페이지 표기). 출처를 밝히면 공유·수정·재배포 가능 |
| 인용 | Fanaee-T, H., Gama, J. "Event labeling combining ensemble detectors and background knowledge", *Progress in Artificial Intelligence* (2013), doi:10.1007/s13748-013-0040-3 |

재배포가 허용되므로 `data/raw/hour.csv`와 `Readme.txt`를 저장소에 함께 두었습니다.
`python -m bikeml run fetch`는 이 파일들의 sha256을 [config/pipeline.json](../config/pipeline.json)의 값과 비교합니다. 파일이 없거나 값이 다르면 UCI에서 zip을 다시 받아 zip의 해시를 확인한 뒤, 필요한 두 파일만 꺼냅니다.

## 2. 데이터 한 건의 의미

`hour.csv`의 한 행은 **한 시간 동안 시스템 전체에서 시작된 대여 건수**와 그 시간의 달력·날씨 정보입니다. 17,379행이며 2011-01-01 00시부터 2012-12-31 23시까지입니다.

| 원자료 열 | 서비스 입력 이름 | 단위·값 | 사용 |
|---|---|---|---|
| dteday + hr | `datetime` | `YYYY-MM-DDTHH:MM` | 연·월·시·요일·근무일을 서비스가 직접 계산 |
| holiday | `holiday` | true/false (D.C. 공휴일) | 입력 |
| weathersit | `weather` | 1 맑음·구름 조금, 2 안개·흐림, 3 약한 비·눈, 4 폭우·폭설 | 입력 |
| temp | `temp_c` | °C | 입력 (아래 변환) |
| atemp | `feels_like_c` | °C | 입력 (아래 변환) |
| hum | `humidity` | % | 입력 (×100) |
| windspeed | `windspeed_kmh` | km/h로 가정 | 입력 (×67) |
| **cnt** | — | **대/시간** | **정답** |
| casual, registered | — | 대/시간 | **제외**: 둘을 더하면 cnt (정답 누출) |
| instant | — | 행 번호 | 제외: 시간 순서 번호라 서비스 입력에 없음 |
| season | — | 1~4 | 제외: 월로 대신함 |
| yr, mnth, weekday, workingday | — | — | 원자료 값은 쓰지 않고 `datetime`, `holiday`에서 다시 계산 |

마지막 줄의 달력 값은 서비스가 받은 날짜로 계산합니다. 학습에도 같은 함수([bikeml/features.py](../bikeml/features.py))를 써서, 학습과 서비스가 같은 방식으로 입력을 만듭니다. `prepare` 단계는 계산한 값이 원자료의 yr, mnth, hr, weekday, workingday, holiday, weathersit와 17,379행 모두 같은지 확인하고, 다르면 멈춥니다(불일치 0건).

### 단위 변환과 근거

원자료의 기온·습도·풍속은 0~1로 정규화되어 있습니다. 실제 사용자는 °C, %로 입력하므로 원래 단위로 되돌렸습니다.

- 습도 = hum × 100, 풍속 = windspeed × 67. Readme에 "최댓값 67로 나눔"이라고 적혀 있지만 단위는 없습니다. 최댓값 57이 km/h로 자연스러워 km/h로 가정했습니다.
- 기온은 두 설명이 서로 다릅니다. 같은 zip의 Readme.txt는 "41로 나눔"(체감 50)이라고 하고, UCI 웹페이지는 `(t − (−8)) / (39 − (−8))`(체감 −16~50)이라고 합니다. 2년 중 가장 추운 시각인 2011-01-22 04시(정규화 0.02)를 넣어 보면 Readme 식은 +0.8°C, 웹페이지 식은 −7.1°C입니다. Readme 식을 따르면 두 해 겨울 내내 워싱턴 기온이 한 번도 영하로 내려가지 않았다는 뜻이 되어 맞지 않습니다. 그래서 **웹페이지 식**을 썼습니다(`temp_c = −8 + 47·temp`, `feels_like_c = −16 + 66·atemp`).
- 트리 모델은 값의 순서만 보므로, 어느 식을 쓰든 평가 점수는 같습니다. 식이 영향을 주는 것은 "사용자가 넣은 섭씨온도를 모델이 어떻게 해석하는가"뿐이며, 이 부분은 한계에 적었습니다.

## 3. 잘못된 값과 빠진 값의 처리

[results/data/prepare_report.json](../results/data/prepare_report.json)에 자동으로 기록됩니다.

| 점검 | 결과 | 처리 |
|---|---|---|
| 빈 값, (날짜, 시각) 중복 | 0건 | 있으면 멈춤 |
| cnt = casual + registered | 전 행 성립 | 아니면 멈춤 |
| 정규화 값 0~1, 시각 0~23, 날씨 1~4 범위 | 전 행 성립 | 아니면 멈춤 |
| 습도 0% | 22행, 모두 2011-03-10 | 비 오는 날의 0%는 불가능한 값이므로 센서 오류로 보고 **결측(NaN)** 처리. 모델이 결측을 직접 다룸 |
| 풍속 0 | 2,180행 | 실제 무풍인지 결측인지 구분할 수 없어 **그대로 둠** (한계) |
| 빠진 시간 | 165시간 (학습 125, 평가 40) | 채우지 않음. 대여가 0건인 시간은 행이 없는 것으로 보임(cnt 최솟값이 1). 2012-10-29~30은 허리케인 샌디로 운영이 멈춰 하루 반 동안 행이 거의 없음 |

서비스 입력은 [bikeml/features.py](../bikeml/features.py)에서 검사합니다. 필수 필드가 없거나, 타입이 틀리거나(숫자 자리에 문자열 등), 범위를 벗어나면(시각 25시, 날씨 5, 습도 130%·0% 등) 400 오류로 돌려보내며 몇 번째 입력의 어느 필드가 틀렸는지 알려 줍니다. 학습 기간(2011~2012) 밖의 연도는 가까운 연도 수준으로 계산하고, 응답에 경고를 붙입니다.

## 4. 학습용·평가용 분할

**시간 순서로 나눴습니다.** 실제 서비스는 과거 자료로 학습해 미래를 예측하므로, 무작위로 나누면 같은 날의 앞뒤 시간이 양쪽에 섞여 점수가 실제보다 좋게 나옵니다.

| 구분 | 기간 | 행 수 | 평균 대여량 |
|---|---|---|---|
| 학습 | 2011-01-01 ~ 2012-08-31 | 14,491 | 179.3 대/시간 |
| 평가 | 2012-09-01 ~ 2012-12-31 | 2,888 | 240.2 대/시간 |

- 하이퍼파라미터는 [config/pipeline.json](../config/pipeline.json)에 미리 고정했습니다. 평가 자료로 고르거나 바꾸지 않았습니다.
- 기준 방법(평균값)도 학습 자료로만 계산했습니다. 결측 처리 규칙(습도 0 → 결측)은 값 자체가 불가능해서 정한 규칙이며, 평가 자료에는 해당 행이 없습니다.
- 예측 시점에 알 수 없는 정보(casual, registered, 행 번호)는 입력에서 뺐습니다. 날씨는 관측값이라 실제 서비스의 "예보"보다 정확합니다. 이 점은 한계에 적었습니다.

## 5. 모델

| 항목 | 내용 |
|---|---|
| 알고리즘 | scikit-learn `HistGradientBoostingRegressor` (그래디언트 부스팅 트리) |
| 손실 | 포아송 손실. 대여 건수는 0 이상의 셈 값이라 이 손실을 쓰고, 예측도 0 이상이 됨 |
| 하이퍼파라미터 | learning_rate 0.05, max_iter 500, max_leaf_nodes 31, min_samples_leaf 20, early_stopping 끔, random_state 42 |
| 입력 특성 (11개) | yr, mnth, hr, weekday, workingday, holiday, weathersit, temp_c, feels_like_c, humidity, windspeed_kmh |
| 버전 이름 | `hgb-v1` ([models/hgb-v1/](../models/hgb-v1/)) |
| 파일 | `model.joblib` (joblib 압축 3), sha256 `9911f56a01c312600ac0ce5101e811299274debd19ad6d077f2e6b8e090d47dd` |
| 메타데이터 | [models/hgb-v1/metadata.json](../models/hgb-v1/metadata.json): 학습 자료 sha256, 기간, 라이브러리 버전, 학습 시간, 학습 MAE(17.66) |
| 재현성 | 같은 자료와 설정으로 두 번 학습했을 때 sha256이 같았습니다(실행 기록의 train 단계 두 건) |

모델 파일은 직접 학습한 결과이고 크기가 작아(0.7MB) 저장소에 함께 둡니다. `python -m bikeml run local`을 실행하면 다시 만들 수 있습니다.

모델 버전은 세 곳에서 확인합니다. (1) `metadata.json`의 `model_sha256`, (2) Lambda 환경 변수 `MODEL_SHA256`(받은 파일의 해시가 다르면 예측하지 않음), (3) 모든 응답의 `model_version`, `model_sha256`.
