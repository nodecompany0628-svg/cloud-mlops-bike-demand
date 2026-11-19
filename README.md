# 날씨로 자전거 시간당 대여량 예측 — 클라우드 MLOps 대체 실습과제

공공 자전거 운영 담당자가 **다음 날 날씨 예보**를 넣으면 **시간대별 대여 건수**를 돌려주는 예측 서비스입니다. 공개 데이터로 모델을 직접 학습했고, AWS Lambda에서 실행합니다. 데이터 준비부터 배포, 검사, 정리까지 명령 하나로 순서대로 실행하고 기록합니다.

> **진행 상태 (2026-11-19)**
> ✅ 실행함: 데이터 받기·점검·분할, 학습, 평가(기준 방법 비교), 서비스 정상 동작 검사(로컬), 단위 테스트, Lambda 배포 패키지 빌드
> ⏳ 실행 전: AWS 배포와 클라우드 검사·평가, 클라우드 오류 재현·복구, 새 폴더 재실행, 자원 정리. AWS 계정이 준비되면 실행하고 이 문서와 [docs/cloud.md](docs/cloud.md), [docs/operations.md](docs/operations.md)를 채웁니다.

## 1. 서비스

| 항목 | 내용 |
|---|---|
| 사용자 | 공공 자전거 운영 담당자 |
| 목적 | 시간대별 수요를 미리 알고 자전거 재배치와 정비 인력을 계획 |
| 입력 (1시간당 1건, 최대 1,000건) | `datetime` 날짜·시각, `holiday` 공휴일 여부, `weather` 날씨 1~4, `temp_c` 기온 °C, `feels_like_c` 체감 기온 °C, `humidity` 습도 %, `windspeed_kmh` 풍속 km/h |
| 출력 | 시간당 예상 대여 건수(대/시간, 0 이상 정수), 모델 버전과 sha256, 경고 |
| 형태 | 하루 한두 번 호출하는 예측 함수(AWS Lambda). 상시 서버와 화면은 두지 않음 |
| 데이터 | UCI Bike Sharing Dataset (워싱턴 D.C. Capital Bikeshare 2011~2012, CC BY 4.0) |
| 모델 | 직접 학습한 그래디언트 부스팅 회귀(`HistGradientBoostingRegressor`, 포아송 손실), 버전 `hgb-v1` |

요청과 응답 예:

```json
{"instances": [{"datetime": "2012-10-01T08:00", "holiday": false, "weather": 1,
                "temp_c": 17.5, "feels_like_c": 21.0, "humidity": 72, "windspeed_kmh": 9.0}]}
```
```json
{"statusCode": 200, "body": {"ok": true, "model_version": "hgb-v1", "model_sha256": "9911f56a…", "n": 1,
  "unit": "대/시간", "predictions": [{"datetime": "2012-10-01T08:00", "predicted_count": 647, "predicted_value": 646.93}]}}
```

입력이 잘못되면 `400`과 함께 몇 번째 입력의 어느 필드가 틀렸는지 알려 줍니다(예: `{"index": 0, "field": "humidity", "message": "필수 입력이 없습니다"}`).

## 2. 구성

```
python -m bikeml run local   fetch → prepare → train → evaluate → check-local
python -m bikeml run cloud   package → deploy → smoke → cloud-eval
                              │          │         │         └ 평가 자료 2,888건을 Lambda로 예측, 로컬과 비교
                              │          │         └ 정상·오류 입력 13개 사례를 Lambda에 보내 검사
                              │          └ S3(모델·패키지) + IAM 역할 + 로그 그룹 + Lambda 함수, 버전 게시
                              └ 리눅스용 라이브러리로 zip 빌드
python -m bikeml status / cleanup   자원 상태 조회 / 삭제 후 재조회
```

| 폴더 | 내용 |
|---|---|
| [bikeml/](bikeml/) | 파이프라인 코드. `features.py`·`service.py`는 Lambda에도 그대로 들어감 |
| [lambda_src/lambda_function.py](lambda_src/lambda_function.py) | Lambda 진입점 (S3에서 모델을 받아 sha256 확인 후 예측) |
| [config/](config/) | 파이프라인 설정, Lambda 설정 |
| [data/](data/) | 원자료(`raw/`), 정리·분할한 자료(`processed/`) |
| [models/hgb-v1/](models/hgb-v1/) | 학습한 모델과 메타데이터 |
| [samples/](samples/) | 서비스 검사 사례, 예시 요청 |
| [results/](results/) | 평가, 서비스 검사, 클라우드 기록 (`failed/`는 실패한 실행) |
| [runs/run_log.jsonl](runs/run_log.jsonl) | 단계별 실행 기록 |
| [tests/](tests/) | 단위 테스트 |
| [docs/](docs/) | 상세 문서 |

## 3. 결과 요약

| 항목 | 결과 | 근거 |
|---|---|---|
| 모델 MAE (평가 2012-09~12, 2,888시간) | **47.53 대/시간** (RMSE 72.95) | [results/evaluation/metrics.json](results/evaluation/metrics.json) |
| 시간대 평균 기준 / 전체 평균 기준 MAE | 94.78 / 170.51 대/시간 → 기준보다 49.9% 감소, 합격 | 같은 파일 |
| 서비스 검사 (로컬 핸들러) | 13/13 통과 | [results/service_check/local.json](results/service_check/local.json) |
| 테스트 | 32개 통과 (모의 AWS로 배포·검사·평가·정리 전 과정 포함) | `python -m pytest -q` |
| Lambda 패키지 | zip 53.14MB, 압축 해제 169.15MB (한도 250MB), 두 번 빌드해 sha256 같음 | [results/cloud/package.json](results/cloud/package.json) |
| 클라우드 실행·평가 | ⏳ 실행 전 | [docs/cloud.md](docs/cloud.md) |

## 4. 필요한 환경과 권한

- Python **3.12** (Lambda 런타임과 같은 버전. [uv](https://docs.astral.sh/uv/)를 쓰면 자동으로 설치됨), Git
- 확인한 환경: Windows 11, Python 3.12.13, pip 26.3, uv 0.11.16
- 라이브러리 버전: [requirements.txt](requirements.txt) (로컬), [requirements-lambda.txt](requirements-lambda.txt) (Lambda 패키지). 두 파일의 scikit-learn, numpy 버전이 같아야 모델을 불러올 수 있음
- 클라우드 단계: AWS 자격 증명(로컬 프로필)과 [docs/iam/operator-policy.json](docs/iam/operator-policy.json) 범위의 권한. AWS CLI는 없어도 됨(boto3 사용)
- 로컬 단계(`run local`, 테스트)는 AWS 없이 실행됩니다.

## 5. 설치

```bash
git clone https://github.com/nodecompany0628-svg/cloud-mlops-bike-demand.git
cd cloud-mlops-bike-demand

uv venv --python 3.12 --seed .venv        # 또는: python3.12 -m venv .venv
.venv\Scripts\activate                    # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 6. 실행과 검증

```bash
# 로컬: 데이터 확인 → 정리·분할 → 학습 → 평가 → 서비스 검사
python -m bikeml run local
python -m pytest -q

# AWS 자원 준비와 배포 → 클라우드 검사 → 클라우드 평가
#   프로필은 --profile 또는 환경 변수 AWS_PROFILE 로 지정
python -m bikeml --profile <프로필> run cloud

# 한 번에 전부
python -m bikeml --profile <프로필> run all

# 단계 하나씩도 가능
python -m bikeml run evaluate
python -m bikeml --profile <프로필> run deploy smoke

# 자원 상태 조회, 정리(삭제 후 재조회)
python -m bikeml --profile <프로필> status
python -m bikeml --profile <프로필> cleanup

# 실행 기록 보기
python -m bikeml runs
```

AWS CLI로 직접 호출하려면:

```bash
aws lambda invoke --function-name bike-demand-predict --region ap-northeast-2 \
  --cli-binary-format raw-in-base64-out \
  --payload file://samples/generated/request_2012-12-03.json out.json
```

한 단계가 실패하면 뒤 단계는 실행하지 않고, 실패한 결과는 `results/failed/<run_id>_<단계>/`에 따로 남습니다. 평가가 합격 기준에 못 미치면 배포하지 않습니다.

## 7. 결과 파일 위치

| 파일 | 내용 |
|---|---|
| `results/data/prepare_report.json` | 원자료 점검, 결측 처리, 단위 변환, 분할 요약 |
| `results/evaluation/metrics.json` | 모델·기준 방법 지표, 월별 지표, 합격 판정 |
| `results/evaluation/predictions_test.csv` | 평가 자료 전체의 실제값과 예측값 |
| `results/evaluation/worst_errors.csv`, `worst_case_context.csv` | 큰 오류 15건, 최대 오류의 앞뒤 7일 |
| `results/evaluation/daily_totals.csv` | 일별 합계 비교 |
| `results/service_check/local.json`, `cloud.json` | 서비스 검사 결과 (로컬 / Lambda) |
| `results/cloud/package.json`, `deploy.json`, `cloud_eval.json`, `predictions_cloud.csv`, `status*.json`, `cleanup.json` | 패키지, 배포, 클라우드 평가, 자원 상태, 정리 |
| `runs/run_log.jsonl` | 단계별 실행 기록 (시각, 성공 여부, 커밋, 모델 버전, 입력·설정 해시, 결과 위치) |

## 8. 문서

| 문서 | 내용 |
|---|---|
| [docs/data_and_model.md](docs/data_and_model.md) | 데이터 출처·사용 조건, 한 건의 의미, 단위, 잘못된 값 처리, 분할 기준, 모델과 버전 |
| [docs/evaluation.md](docs/evaluation.md) | 평가 기준, 결과, 정상 동작 검사, 오류 사례 분석, 한계 |
| [docs/cloud.md](docs/cloud.md) | 클라우드 구성과 선택 이유, 설정, 실행 기록 형식, 오류 재현·복구 |
| [docs/operations.md](docs/operations.md) | 새 폴더 재실행, 권한, 비용, 자원 정리 |

## 9. 참고 자료와 구현 범위

참고 자료
- 데이터: Fanaee-T, H., Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*. UCI Machine Learning Repository, Bike Sharing Dataset (CC BY 4.0).
- scikit-learn 문서: `HistGradientBoostingRegressor`, 포아송 손실
- AWS 문서: Lambda .zip 배포 패키지와 크기 한도, Lambda 실행 역할, S3 버킷 버전 관리·수명 주기, CloudWatch Logs 보관 기간

이 저장소에서 작성한 부분
- `bikeml/`, `lambda_src/`, `tests/` 전체 코드: 입력 검사와 특성 변환, 데이터 점검·분할, 학습, 기준 방법 비교 평가, 서비스 검사, Lambda 핸들러, 배포 패키지 빌드, AWS 자원 준비·배포·호출·조회·정리, 실행 기록과 실패 처리
- 설정(`config/`), 서비스 검사 사례(`samples/service_cases.json`), 문서(`docs/`)
- 외부 코드를 복사해 쓰지 않았고, 라이브러리는 위 버전 파일에 적은 공개 패키지만 씁니다.
- 코드와 문서 작성에 AI 코딩 도구(Claude Code)를 함께 사용했습니다.
