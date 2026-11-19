# 날씨로 자전거 시간당 대여량 예측

클라우드 MLOps 대체 실습과제. 공공 자전거 운영 담당자가 다음 날의 날씨 예보를 넣으면 시간대별 대여 건수를 돌려주는 예측 서비스다. 공개 데이터로 모델을 직접 학습했고 AWS Lambda에서 실행한다. 데이터 준비부터 학습, 평가, 배포, 검사까지는 `run` 명령으로 단계를 순서대로 실행하고, 자원 정리는 `cleanup` 명령으로 한다. 단계마다 실행 기록을 남긴다.

진행 상태 (2026-11-20)

- 실행함: 데이터 받기와 점검, 분할, 학습, 평가, 로컬 서비스 검사, 테스트, Lambda 배포 패키지 빌드, 실패 처리 예시(엄격한 합격 기준으로 평가 실패)
- 모의 AWS(moto)로만 확인함: 자원 생성과 배포, 버전 게시, `live` 별칭(release), 상태 조회, 정리 흐름. 이 테스트에서는 배포 전 확인을 대체하고, 작은 가짜 zip을 올리고, Lambda 호출을 로컬 핸들러로 대신한다.
- 실행 전: 실제 AWS 배포와 클라우드 검사·평가, 클라우드 오류 재현과 복구, 새 폴더 재실행(클라우드 포함), 자원 정리. AWS 계정이 준비되면 실행하고 이 문서와 [docs/cloud.md](docs/cloud.md), [docs/operations.md](docs/operations.md)를 채운다.

## 1. 서비스

| 항목 | 내용 |
|---|---|
| 사용자 | 공공 자전거 운영 담당자 |
| 목적 | 시간대별 수요를 미리 보고 자전거 재배치와 정비 인력을 계획 |
| 입력 (1시간당 1건, 최대 1,000건) | `datetime` 날짜와 시각, `holiday` 공휴일 여부, `weather` 날씨 1~4, `temp_c` 기온 °C, `feels_like_c` 체감 기온 °C, `humidity` 습도 %, `windspeed_kmh` 풍속 km/h |
| 출력 | 시간당 예상 대여 건수(대/시간, 0 이상 정수), 모델 버전과 sha256, 실행 환경 라이브러리 버전, 경고 |
| 형태 | 하루 한두 번 호출하는 예측 함수(AWS Lambda 직접 호출). 상시 서버와 화면은 두지 않는다 |
| 데이터 | UCI Bike Sharing (워싱턴 D.C. Capital Bikeshare 2011~2012, CC BY 4.0) |
| 모델 | 직접 학습한 `HistGradientBoostingRegressor`(포아송 손실), 버전 `hgb-v1` |

요청 예

```json
{"instances": [{"datetime": "2012-10-01T08:00", "holiday": false, "weather": 1,
                "temp_c": 17.5, "feels_like_c": 21.0, "humidity": 72, "windspeed_kmh": 9.0}]}
```

응답 예 (로컬 실행 결과)

```json
{"statusCode": 200, "body": {"ok": true, "model_version": "hgb-v1", "model_sha256": "f6186c0a...", "n": 1,
  "unit": "대/시간", "predictions": [{"datetime": "2012-10-01T08:00", "predicted_count": 647, "predicted_value": 646.93}],
  "runtime": {"python": "3.12.13", "scikit-learn": "1.9.1", "numpy": "2.5.5", "boto3": "1.44.6", "execution_env": "local"}}}
```

입력이 잘못되면 400과 함께 몇 번째 입력의 어느 필드가 틀렸는지 돌려준다. 예: `{"index": 0, "field": "humidity", "message": "필수 입력이 없습니다"}`

## 2. 구성

| 명령 | 하는 일 |
|---|---|
| `python -m bikeml run local` | fetch, prepare, train, evaluate, check-local |
| `python -m bikeml run cloud` | package, deploy, smoke, cloud-eval, release |
| `python -m bikeml run all` | local과 cloud를 이어서 실행 |
| `python -m bikeml status` | AWS 자원 상태 조회 |
| `python -m bikeml cleanup` | AWS 자원 삭제 후 다시 조회 |

| 단계 | 하는 일 |
|---|---|
| fetch | UCI에서 원자료를 받고 sha256 확인 (이미 있으면 확인만) |
| prepare | 원자료 점검, 단위 변환, 결측 처리, 시간 순서 분할 |
| train | 모델 학습. 결과는 후보 폴더 `models/_candidate/`에 둔다 |
| evaluate | 정리한 자료가 설정의 분할 기간과 맞는지, 평가할 모델이 지금 학습 자료, 하이퍼파라미터, 특성 코드로 학습됐는지 확인한 뒤 기준 방법과 비교하고 합격 기준을 본다. 합격하면 후보 모델을 `models/hgb-v1/`로 복사하고, 불합격이면 멈춘다 |
| check-local | 서비스 검사 사례 12개를 Lambda 핸들러에 로컬로 보냄 |
| package | 리눅스용 라이브러리로 Lambda zip 빌드 |
| deploy | 배포 전 확인(모델, 학습 자료, 분할 기간, 특성 코드, 라이브러리 버전, 평가, 검사, zip, 런타임과 아키텍처)을 한 뒤 권한 경계를 확인하고 IAM 역할, S3, 로그 그룹, Lambda 함수를 만들고 버전 게시 |
| smoke | 같은 검사 사례 12개를 Lambda에 보냄 |
| cloud-eval | 평가 자료 2,888건을 Lambda로 예측해 로컬 결과와 비교 |
| release | smoke와 cloud-eval을 모두 통과한 버전에만 `live` 별칭을 붙임 (리전, 리비전, 코드, 모델 경로까지 같은지 확인) |

| 폴더 | 내용 |
|---|---|
| [bikeml/](bikeml/) | 파이프라인 코드. `features.py`, `service.py`는 Lambda에도 들어간다 |
| [lambda_src/lambda_function.py](lambda_src/lambda_function.py) | Lambda 진입점. S3에서 모델을 받아 sha256과 입력 개수를 확인한 뒤 예측 |
| [config/](config/) | 파이프라인 설정, Lambda 설정, 실패 예시용 설정(`demo_strict_criteria.json`) |
| [data/](data/) | 원자료(`raw/`), 정리·분할한 자료(`processed/`), 출처 표기([data/README.md](data/README.md)) |
| [models/hgb-v1/](models/hgb-v1/) | 평가를 통과한 모델과 메타데이터 (후보 폴더 `models/_candidate/`는 Git 제외) |
| [samples/](samples/) | 서비스 검사 사례, 예시 요청 |
| [results/](results/) | 평가, 서비스 검사, 클라우드 기록. 실패한 실행은 `results/failed/` |
| [runs/run_log.jsonl](runs/run_log.jsonl) | 단계별 실행 기록 |
| [tests/](tests/) | 테스트 |
| [docs/](docs/) | 상세 문서 |

## 3. 결과 요약

| 항목 | 결과 | 근거 |
|---|---|---|
| 모델 MAE (평가 2012-09~12, 2,888시간) | 47.53 대/시간 (RMSE 72.95) | [results/evaluation/metrics.json](results/evaluation/metrics.json) |
| 기준 방법 MAE | 시간대 평균 94.78, 전체 평균 170.51 대/시간. 시간대 평균보다 49.9% 낮아 합격 | 같은 파일 |
| 서비스 검사 (로컬 핸들러) | 12/12 통과 | [results/service_check/local.json](results/service_check/local.json) |
| 실패 처리 예시 | 합격 기준을 90%로 올린 설정에서 evaluate 실패, check-local 건너뜀, 정상 결과는 그대로 | [results/failed/](results/failed/), [runs/run_log.jsonl](runs/run_log.jsonl) |
| 테스트 | 116개 통과 | `python -m pytest -q` |
| Lambda 패키지 | zip 64.29MB, 압축 해제 198.94MB (한도 250MB) | [results/cloud/package.json](results/cloud/package.json) |
| 실제 AWS 실행 | 실행 전 | [docs/cloud.md](docs/cloud.md) |

## 4. 필요한 환경과 권한

- Python 3.12 (Lambda 런타임과 같은 버전. [uv](https://docs.astral.sh/uv/)를 쓰면 자동으로 받는다), Git
- 확인한 환경: Windows 11, Python 3.12.13, pip 26.3, uv 0.11.16
- 라이브러리 버전: [requirements.txt](requirements.txt)(파이프라인), [requirements-dev.txt](requirements-dev.txt)(테스트: pytest, moto), [requirements-lambda.txt](requirements-lambda.txt)(Lambda 패키지). 세 파일 모두 간접 의존성까지 버전을 고정했다(Windows 전용 패키지 2개는 `sys_platform == "win32"` 조건). requirements.txt와 requirements-lambda.txt의 scikit-learn, numpy 버전이 같아야 저장한 모델을 Lambda에서 불러올 수 있다.
- 클라우드 단계: AWS 자격 증명을 로컬 프로필로 등록하고 [docs/iam/operator-policy.json](docs/iam/operator-policy.json) 범위의 권한을 준다. AWS CLI는 없어도 된다(boto3 사용).
- 프로필을 지정하지 않으면 클라우드 단계는 AWS에 접속하기 전에 멈춘다. 기본 자격 증명으로 다른 계정에 배포하는 일을 막기 위해서다. 프로필(`--profile`, `AWS_PROFILE`, 설정의 `cloud.profile`)이 하나라도 지정되어 있으면 그 프로필을 쓴다. 설정 파일은 공개 저장소에 올라가므로 프로필 이름은 `--profile`이나 `AWS_PROFILE`로 주는 것을 권한다. 프로필 없이 환경 변수의 키(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)를 쓰려면 `BIKEML_USE_ENV_CREDENTIALS=1`도 함께 지정해야 하고, 이때는 환경 변수의 키만 쓴다(키가 없으면 멈춘다).
- 로컬 단계와 테스트는 AWS 없이 실행된다.

## 5. 자원 준비와 설정

1. 관리자(또는 루트)가 [docs/iam/lambda-boundary.json](docs/iam/lambda-boundary.json)으로 `bike-demand-predict-boundary`라는 고객 관리형 정책을 만든다. Lambda 실행 역할의 권한 경계로 쓰인다. 루트나 관리자 키로 배포하면 도구가 없을 때 직접 만든다.
2. IAM 사용자를 만들고, [docs/iam/operator-policy.json](docs/iam/operator-policy.json)으로 고객 관리형 정책을 만들어 붙인다. 이 정책은 2,048자를 넘어 사용자 인라인 정책으로는 붙일 수 없다(고객 관리형 정책 한도는 6,144자). 이 정책으로는 권한 경계가 붙은 역할만 만들거나 바꿀 수 있어서, 역할 정책을 넓혀 계정 권한을 얻는 길이 막혀 있다.
3. 이 PC에 프로필로 등록한다: `aws configure --profile mlops` 또는 `~/.aws/credentials`에 직접 적는다. 키는 저장소에 넣지 않는다.
4. 리전과 이름은 [config/pipeline.json](config/pipeline.json)의 `cloud`에서, 메모리·제한 시간·동시 실행은 [config/lambda.json](config/lambda.json)에서 바꾼다. 함수, 역할, 권한 경계, 버킷 접두어 이름을 바꾸면 [docs/iam/operator-policy.json](docs/iam/operator-policy.json)의 자원 이름과 [docs/iam/lambda-boundary.json](docs/iam/lambda-boundary.json)의 함수 이름, 버킷 접두어도 같이 바꾸고, 관리자가 경계 정책을 그 내용으로 고쳐야 한다. 경계 정책의 내용이 설정과 다르면 `deploy`는 자원을 만들기 전에 멈춘다.
5. S3 버킷, IAM 역할, 로그 그룹, Lambda 함수는 `deploy` 단계가 만든다. 버킷 이름은 `bike-demand-mlops-<리전>-<임의 8자리>`이고, 다음 실행부터는 같은 접두어의 버킷을 찾아 다시 쓴다.

## 6. 설치

Windows (PowerShell)

```powershell
git clone https://github.com/nodecompany0628-svg/cloud-mlops-bike-demand.git
cd cloud-mlops-bike-demand
uv venv --python 3.12 --seed .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

macOS, Linux

```bash
git clone https://github.com/nodecompany0628-svg/cloud-mlops-bike-demand.git
cd cloud-mlops-bike-demand
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

uv 대신 Python 3.12의 `python -m venv .venv`를 써도 된다. 아래 명령의 `python`은 가상환경의 python이다. Windows에서 가상환경을 활성화하지 않았다면 `.venv\Scripts\python`으로 바꿔 쓴다.

Windows에서는 저장소 경로가 길면(약 110자 이상) 경로 길이 제한 때문에 moto 설치가 실패한다. 짧은 경로(예: `C:\work`)에 받거나 Windows 긴 경로 지원을 켠다. moto가 없으면 모의 AWS 테스트는 건너뛰고 나머지 테스트와 파이프라인은 그대로 동작한다.

## 7. 실행과 검증

```
python -m bikeml run local
python -m pytest -q

python -m bikeml --profile mlops run cloud
python -m bikeml --profile mlops status
python -m bikeml --profile mlops cleanup

python -m bikeml runs
```

- 전체 옵션 `--config`, `--profile`, `--region`, `--lambda-config`는 `run`, `status`, `cleanup`, `runs` 앞에 쓴다.
- `status`, `cleanup`은 지금 리전 하나만 보지 않고, 상태 파일, 실행 기록, 배포 기록, 버킷 이름에서 찾은 모든 리전과 기록에 남은 버킷까지 조회하고 정리한다. 자세한 규칙은 [docs/cloud.md](docs/cloud.md) 2절에 있다.
- 단계 하나씩도 실행할 수 있다: `python -m bikeml run evaluate`, `python -m bikeml --profile mlops run deploy smoke`
- 한 단계가 실패하면 뒤 단계는 실행하지 않고, 실패한 결과는 `results/failed/<run_id>_<단계>/`에 따로 남는다. 실행 중 Ctrl+C로 멈춰도 그 단계는 실패로 기록된다.
- 실패 처리 예시 재현: `python -m bikeml --config config/demo_strict_criteria.json run evaluate check-local`
- `deploy`는 정리한 자료가 설정의 분할 기간과 맞고, 모델이 지금 학습 자료, 설정, 특성 코드로 학습됐고, 지금 합격 기준과 평가 자료로 `evaluate`를 통과했고, 지금 Lambda 코드로 `check-local`을 통과했으며, zip이 지금 코드와 런타임, 아키텍처로 만든 것일 때만 진행한다.
- 사용자는 `live` 별칭으로 호출한다. `live`는 `release` 단계가 smoke와 cloud-eval을 모두 통과한 버전에만 옮긴다.

AWS CLI로 직접 호출하는 방법. 검사를 통과한 버전을 가리키는 `live` 별칭으로 부른다.

```
aws lambda invoke --function-name bike-demand-predict --qualifier live --region ap-northeast-2 --profile mlops --cli-binary-format raw-in-base64-out --payload file://samples/generated/request_2012-12-03.json out.json
```

## 8. 결과 파일 위치

| 파일 | 내용 |
|---|---|
| `results/data/prepare_report.json` | 원자료 점검, 결측 처리, 단위 변환, 분할 요약 |
| `results/evaluation/metrics.json` | 모델과 기준 방법의 지표, 월별 지표, 합격 기준과 판정, 평가 자료 sha256 |
| `results/evaluation/predictions_test.csv` | 평가 자료 전체의 실제값과 예측값 |
| `results/evaluation/worst_errors.csv`, `worst_case_context.csv` | 큰 오류 15건, 최대 오류의 앞뒤 7일 |
| `results/evaluation/daily_totals.csv` | 일별 합계 비교 |
| `results/service_check/local.json`, `cloud.json` | 서비스 검사 결과 (로컬, Lambda), 실행 환경 버전 |
| `results/cloud/package.json` | 패키지 해시, 크기, 들어간 라이브러리와 코드 해시 |
| `results/cloud/deploy.json`, `deploy_override.json` | 정상 배포 기록, 오류 재현용 배포 기록 |
| `results/cloud/cloud_eval.json`, `predictions_cloud.csv` | 클라우드 평가 (호출이 실패하거나 예외로 끝나면 그때까지의 호출 기록 `cloud_eval_calls.json`이 실패 폴더에 남음) |
| `results/cloud/release.json` | `live` 별칭을 옮긴 기록 (이전 버전 포함) |
| `results/cloud/status*.json`, `cleanup.json` | 자원 상태, 정리 |
| `results/failed/<run_id>_<단계>/` | 실패한 단계의 결과와 `error.txt` |
| `runs/run_log.jsonl` | 단계별 실행 기록 (시각, 성공 여부, 커밋, 모델 버전, 입력·설정 해시, 결과 위치) |

## 9. 문서

| 문서 | 내용 |
|---|---|
| [docs/data_and_model.md](docs/data_and_model.md) | 데이터 출처와 사용 조건, 한 건의 의미, 단위, 잘못된 값 처리, 분할 기준, 모델과 버전 |
| [docs/evaluation.md](docs/evaluation.md) | 평가 기준, 결과, 정상 동작 검사, 오류 사례 분석, 한계 |
| [docs/cloud.md](docs/cloud.md) | 클라우드 구성과 선택 이유, 설정, 실행 기록과 실패 처리, 오류 재현과 복구 |
| [docs/operations.md](docs/operations.md) | 새 폴더 재실행, 권한, 비용, 자원 정리 |

## 10. 참고 자료와 구현 범위

참고 자료

- Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894 (CC BY 4.0)
- Fanaee-T, H., Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, 2(2-3), 113-127. https://doi.org/10.1007/s13748-013-0040-3
- scikit-learn, HistGradientBoostingRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- AWS Lambda 할당량: https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html
- AWS Lambda 런타임: https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.html
- AWS Lambda 예약 동시성: https://docs.aws.amazon.com/lambda/latest/dg/configuration-concurrency.html
- Amazon S3 수명 주기 규칙: https://docs.aws.amazon.com/AmazonS3/latest/userguide/intro-lifecycle-rules.html
- Amazon S3 HeadObject (권한에 따른 403, 404): https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html
- AWS Lambda 가격: https://aws.amazon.com/lambda/pricing/

데이터와 그 데이터로 학습한 모델은 CC BY 4.0의 출처 표기 조건을 따른다([data/README.md](data/README.md)).

이 저장소에서 구현한 부분

- `bikeml/`, `lambda_src/`, `tests/`의 전체 코드: 입력 검사와 특성 변환, 데이터 점검과 분할, 학습, 기준 방법 비교 평가, 서비스 검사, Lambda 핸들러, 배포 패키지 빌드, AWS 자원 준비·배포·호출·조회·정리, 실행 기록과 실패 처리
- 설정(`config/`), 서비스 검사 사례(`samples/service_cases.json`), 문서(`docs/`)
- 외부 코드를 복사해 쓰지 않았고, 라이브러리는 버전 파일에 적은 공개 패키지만 쓴다.
- 코드와 문서 작성에 AI 코딩 도구(Claude Code)를 함께 사용했다.
