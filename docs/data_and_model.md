# 데이터와 모델

## 1. 데이터 출처와 사용 조건

| 항목 | 내용 |
|---|---|
| 이름 | Bike Sharing (UCI Machine Learning Repository, id 275) |
| 주소 | https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset |
| 데이터셋 DOI | 10.24432/C5W894 |
| 받은 파일 | `bike+sharing+dataset.zip`(sha256 `b70182d0...845401`) 안의 `hour.csv`, `Readme.txt` |
| 원 출처 | Capital Bikeshare(미국 워싱턴 D.C.) 2011~2012년 대여 기록. 날씨는 freemeteo.com |
| 사용 조건 | CC BY 4.0. 출처를 밝히면 공유, 수정, 재배포가 가능하다 |
| 논문 | Fanaee-T, H., Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, 2(2-3), 113-127. doi:10.1007/s13748-013-0040-3 |

재배포가 허용되므로 원자료와 가공한 자료를 저장소에 함께 두었다. 출처와 바꾼 점은 [data/README.md](../data/README.md)에 적었다. `python -m bikeml run fetch`는 파일의 sha256을 [config/pipeline.json](../config/pipeline.json) 값과 비교한다. 파일이 없거나 값이 다르면 UCI에서 zip을 다시 받아 zip의 해시를 확인하고, 필요한 두 파일만 꺼낸다.

## 2. 데이터 한 건의 의미

`hour.csv`의 한 행은 한 시간 동안 시스템 전체에서 시작된 대여 건수와 그 시간의 달력, 날씨 정보다. 17,379행이고 기간은 2011-01-01 00시부터 2012-12-31 23시까지다.

| 원자료 열 | 서비스 입력 | 단위와 값 | 사용 |
|---|---|---|---|
| dteday + hr | `datetime` | `YYYY-MM-DDTHH:MM` | 연, 월, 시, 요일, 근무일을 서비스가 직접 계산 |
| holiday | `holiday` | true/false (D.C. 공휴일) | 입력 |
| weathersit | `weather` | 1 맑음·구름 조금, 2 안개·흐림, 3 약한 비·눈, 4 폭우·폭설 | 입력 |
| temp | `temp_c` | °C | 입력 (아래 변환) |
| atemp | `feels_like_c` | °C | 입력 (아래 변환) |
| hum | `humidity` | % | 입력 (×100) |
| windspeed | `windspeed_kmh` | km/h로 가정 | 입력 (×67) |
| cnt | 없음 | 대/시간 | 정답 |
| casual, registered | 없음 | 대/시간 | 제외. 둘을 더하면 cnt라서 정답이 새어 들어간다 |
| instant | 없음 | 행 번호 | 제외. 시간 순서 번호이고 서비스 입력에 없다 |
| season | 없음 | 1~4 | 제외. 월로 대신한다 |
| yr, mnth, weekday, workingday | 없음 | | 원자료 값 대신 `datetime`, `holiday`로 다시 계산 |

모델 입력은 11개다: yr(2011=0, 2012=1), mnth, hr, weekday(0=일요일~6=토요일, 원자료와 같은 방식), workingday(주말도 공휴일도 아니면 1), holiday, weathersit, temp_c, feels_like_c, humidity, windspeed_kmh.

달력 값은 학습할 때와 서비스할 때 같은 함수([bikeml/features.py](../bikeml/features.py))로 만든다. `prepare` 단계는 이렇게 계산한 yr, mnth, hr, weekday, workingday, holiday, weathersit가 17,379행 모두 원자료와 같은지 확인하고, 하나라도 다르면 멈춘다. 결과는 모두 0건이다(`results/data/prepare_report.json`의 `derived_feature_mismatches`).

### 단위 변환과 근거

원자료의 기온, 습도, 풍속은 0~1로 정규화되어 있다. 사용자는 °C와 %로 입력하므로 원래 단위로 되돌렸다.

- 습도는 hum × 100, 풍속은 windspeed × 67이다. Readme에 "최댓값 67로 나눔"이라고만 있고 단위는 없다. 최댓값 57이 km/h로 자연스러워 km/h로 가정했다.
- 기온은 두 설명이 서로 다르다. zip 안의 Readme.txt는 "41로 나눔"(체감 50)이고, UCI 웹페이지는 `(t - t_min)/(t_max - t_min)`, t_min=-8, t_max=+39(체감은 -16, +50)다.
- 정규화 값의 최솟값 0.02는 2011년 1월 22~24일과 2012-01-04에 걸쳐 17시간 나온다. 이 값을 Readme 식에 넣으면 +0.8°C, 웹페이지 식에 넣으면 -7.1°C다. Readme 식이 맞다면 두 해 겨울 동안 워싱턴 기온이 한 번도 영하로 내려가지 않았다는 뜻이 되어 맞지 않는다. 그래서 웹페이지 식을 썼다(`temp_c = -8 + 47 × temp`, `feels_like_c = -16 + 66 × atemp`).
- 트리 모델은 값의 순서만 보므로 어느 식을 써도 평가 점수는 같다. 식의 선택은 사용자가 넣은 °C를 모델이 어떻게 해석하는지에만 영향을 준다. 이 점은 한계에 적었다.

## 3. 잘못된 값과 빠진 값

점검 결과는 [results/data/prepare_report.json](../results/data/prepare_report.json)에 자동으로 남는다.

| 점검 | 결과 | 처리 |
|---|---|---|
| 빈 값, 같은 날짜·시각 중복 | 0건 | 있으면 멈춤 |
| cnt = casual + registered | 모든 행에서 성립 | 아니면 멈춤 |
| 정규화 값 0~1, 시각 0~23, 날씨 1~4 | 모든 행에서 성립 | 아니면 멈춤 |
| 습도 0% | 22행, 모두 2011-03-10 | 그날 22시간 중 20시간이 날씨 3(약한 비·눈)이라 습도 0%는 불가능한 값이다. 센서 오류로 보고 결측(NaN)으로 바꿨다. 모델이 결측을 직접 처리한다. 나머지 행의 습도 최솟값은 8%다 |
| 풍속 0 | 2,180행 | 실제 무풍인지 결측인지 구분할 수 없어 그대로 두었다 (한계) |
| 빠진 시간 | 165시간 (학습 125, 평가 40) | 채우지 않았다. cnt 최솟값이 1이라 대여가 0건인 시간은 행 자체가 없는 것으로 보인다. 2012-10-29 01시부터 10-30 12시까지 36시간은 허리케인 샌디로 운영이 멈춘 기간이다 |

서비스 입력은 [bikeml/features.py](../bikeml/features.py)에서 검사한다. 다음 경우 400 오류로 돌려보내고, 몇 번째 입력의 어느 필드가 틀렸는지 알려 준다.
- 필수 필드가 없을 때
- 타입이 틀릴 때(숫자 자리에 문자열)
- 범위를 벗어날 때(25시, 날씨 5, 습도 0%나 130%, 지나치게 큰 수)
- 날짜만 있고 시각이 없을 때

다음 경우에는 계산은 하되 응답에 경고를 붙인다.
- 학습 기간(2011~2012) 밖의 연도는 가까운 연도 수준으로 계산한다.
- 분·초는 버린다.
- 시간대 표시는 무시한다.
- 알 수 없는 필드는 무시한다.

## 4. 학습용과 평가용 분할

시간 순서로 나눴다. 실제 서비스는 과거 자료로 학습해 미래를 예측한다. 무작위로 나누면 같은 날의 앞뒤 시간이 양쪽에 섞여 점수가 실제보다 좋게 나온다.

| 구분 | 기간 | 행 수 | 평균 대여량 |
|---|---|---|---|
| 학습 | 2011-01-01 ~ 2012-08-31 | 14,491 | 179.3 대/시간 |
| 평가 | 2012-09-01 ~ 2012-12-31 | 2,888 | 240.2 대/시간 |

- 하이퍼파라미터는 [config/pipeline.json](../config/pipeline.json)에 미리 고정했다. 평가 자료를 보고 고르거나 바꾸지 않았다.
- 기준 방법(평균값)도 학습 자료로만 계산했다. 습도 0을 결측으로 바꾸는 규칙은 값 자체가 불가능해서 정한 것이고, 평가 자료에는 해당 행이 없다.
- 예측 시점에 알 수 없는 정보(casual, registered, 행 번호)는 입력에서 뺐다. 날씨는 관측값이라 실제 서비스에서 쓸 예보보다 정확하다. 이 점은 한계에 적었다.

## 5. 모델

| 항목 | 내용 |
|---|---|
| 알고리즘 | scikit-learn `HistGradientBoostingRegressor` (그래디언트 부스팅 트리) |
| 손실 | 포아송. 대여 건수는 0 이상의 셈 값이라 이 손실을 쓰고, 예측도 0 이상이 된다 |
| 하이퍼파라미터 | learning_rate 0.05, max_iter 500, max_leaf_nodes 31, min_samples_leaf 20, early_stopping 끔, random_state 42 |
| 버전 이름 | `hgb-v1` ([models/hgb-v1/](../models/hgb-v1/)). `train`은 후보 모델을 `models/_candidate/hgb-v1/`(Git 제외)에 만들고, `evaluate`가 합격 기준을 통과시킨 경우에만 `models/hgb-v1/`로 복사한다. 그래서 평가에 떨어진 모델이 합격한 모델을 덮지 않는다. `evaluate`는 평가할 모델의 학습 자료 sha256, 하이퍼파라미터, 버전이 지금 설정과 같은지 먼저 확인하고, 다르면 멈춘다(분할 기간만 바꾸고 다시 학습하지 않은 경우 등) |
| 파일 | `model.joblib` (joblib 압축 3, 0.7MB), sha256 `9911f56a01c312600ac0ce5101e811299274debd19ad6d077f2e6b8e090d47dd` |
| 메타데이터 | [models/hgb-v1/metadata.json](../models/hgb-v1/metadata.json): 학습 자료 sha256과 기간, 라이브러리 버전, 학습 시간, 학습 MAE(17.66) |
| 재현성 | 같은 자료와 설정으로 여러 번 학습해도 sha256이 같았다. `runs/run_log.jsonl`의 train 단계 기록이 모두 `9911f56a...`이다 |

early_stopping을 끈 이유: 기본값 'auto'는 학습 자료가 10,000행을 넘으면 학습 자료의 일부를 무작위로 떼어 조기 종료에 쓴다. 학습 자료 14,491행에서는 이 동작이 켜지므로, 결과를 고정하려고 명시적으로 껐다.

모델 버전은 네 곳에서 확인한다.
1. `metadata.json`의 `model_sha256`
2. Lambda 환경 변수 `MODEL_SHA256`. 받은 파일의 해시가 다르면 예측하지 않는다.
3. 모든 응답의 `model_version`, `model_sha256`, `runtime`(실행 환경의 Python, scikit-learn, numpy, boto3 버전)
4. 실행 기록 `runs/run_log.jsonl`의 `model`
