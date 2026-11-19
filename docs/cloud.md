# 클라우드 실행

진행 상태: 구성과 코드는 완성했고 배포 패키지 빌드(`package`)는 로컬에서 실행했다. 자원 생성, 배포, 버전 게시, 별칭, 상태 조회, 정리의 흐름은 모의 AWS(moto) 테스트로 확인했다. 이 테스트에서는 배포 전 확인을 대체하고, 작은 가짜 zip을 올리고, Lambda 호출을 로컬 핸들러로 대신한다. 실제 AWS에서의 배포(`deploy`), 클라우드 검사(`smoke`), 클라우드 평가(`cloud-eval`), 오류 재현, 정리는 아직 실행하지 않았다. 실행하면 3절과 5절, 그리고 [operations.md](operations.md)를 실제 기록으로 채운다.

## 1. 구성과 선택 이유

자원 (리전 ap-northeast-2, 서울)

| 자원 | 이름 | 내용 |
|---|---|---|
| S3 버킷 | `bike-demand-mlops-ap-northeast-2-<임의 8자리>` | 비공개, 버전 관리. `models/hgb-v1/<모델 sha 12자리>/model.joblib`, `metadata.json`, `packages/lambda-<zip sha 12자리>.zip` |
| Lambda 함수 | `bike-demand-predict` | python3.12, x86_64, 1024MB, 30초. smoke와 cloud-eval을 통과한 버전에 `live` 별칭 |
| IAM 역할 | `bike-demand-predict-role` | Lambda 실행 역할 |
| 로그 그룹 | `/aws/lambda/bike-demand-predict` | 7일 보관 |

흐름

1. `package`: 운영 담당자 PC에서 리눅스용 라이브러리로 zip(64MB)을 만든다.
2. `deploy`: 배포 전 확인을 한 뒤 버킷, 역할, 로그 그룹을 준비하고 모델과 zip을 S3에 올린다. 함수를 만들거나 갱신하고 새 버전을 게시한다.
3. Lambda는 콜드 스타트 때 S3에서 고정된 객체 버전의 모델을 한 번 받아 sha256과 입력 개수를 확인하고, 이후 요청은 메모리의 모델로 예측한다.
4. `smoke`: 검사 사례 12개를 같은 실행에서 `deploy`가 게시한 버전에 보낸다(`smoke`만 따로 실행하면 가장 최근에 게시된 버전).
5. `cloud-eval`: 평가 자료 2,888건을 같은 버전에 500건씩 보내 예측을 받고 로컬 결과와 비교한다.
6. `release`: 그 버전이 smoke와 cloud-eval을 모두 통과했을 때만 `live` 별칭을 그 버전으로 옮긴다. 버전 번호만 보지 않고, 검사 때 기록한 버전의 정체(리전, 리비전 ID, 코드 sha256, 모델 경로와 sha256)가 지금 버전과 모두 같은지 확인한다. 함수를 지웠다 다시 만들면 버전 번호가 1부터 다시 시작하므로 번호만으로는 같은 버전인지 알 수 없기 때문이다. 이전에 `live`가 가리키던 버전도 기록한다.
7. 응답과 기록은 PC의 `results/`, `runs/`에 남기고 Git에 올린다. 함수 로그는 CloudWatch Logs에 남는다.

| 선택 | 이유 |
|---|---|
| AWS Lambda | 운영 담당자가 하루 한두 번 다음 날 24시간 예보를 넣어 호출하는 용도다. 상시 서버나 화면이 필요 없고, 호출할 때만 실행되어 요청이 없으면 비용이 없다. 모델이 0.7MB로 작아 함수 하나로 충분하다. Lambda에서의 실제 처리 시간과 메모리 사용량은 배포 후 REPORT 로그로 기록한다 |
| 직접 호출 (`lambda:InvokeFunction`), 공개 주소 없음 | 사용자가 담당자 몇 명뿐이라 공개 HTTP 주소를 열 이유가 없다. IAM 권한이 있는 사람만 호출할 수 있어 인증을 따로 만들지 않았다 |
| `live` 별칭 | 사용자는 `live`로 호출한다. `live`는 smoke와 cloud-eval을 모두 통과한 버전만 가리키므로, 배포나 검사 도중 실패해 `$LATEST`나 새 버전이 어중간한 상태여도 사용자 호출에는 영향이 없다 |
| 모델은 S3에 두고 실행할 때 내려받음 | 모델 버전을 코드 패키지와 따로 관리한다. 모델 경로에 모델 sha256 앞 12자리를 넣어(`models/hgb-v1/<sha12>/model.joblib`) 내용이 다른 모델은 다른 객체가 되게 하고, Lambda 환경 변수 `MODEL_URI`에 S3 객체 버전 ID까지 적는다(`...model.joblib?versionId=...`). 그래서 다시 학습해 올려도 이미 게시한 Lambda 버전이나 `live`가 가리키는 모델 파일은 덮이거나 수명 주기 규칙으로 지워지지 않는다 |
| 같은 파일은 다시 올리지 않음 | 업로드 전에 S3의 현재 객체에 적힌 sha256을 보고, 같으면 그 객체 버전을 그대로 쓴다. 함수 설정의 설명은 고정 문구이고 커밋 정보는 버전 설명에만 넣는다. 그래서 모델, 코드, 설정이 그대로면 Lambda 설정도 그대로이고, AWS가 새 버전 번호를 만들지 않고 기존 번호를 돌려준다 |
| 버전 관리 반영 대기 | AWS 문서는 버킷에서 버전 관리를 처음 켠 뒤 반영에 최대 15분이 걸릴 수 있다고 한다. 업로드 결과에 객체 버전이 없거나 그 버전을 읽지 못하면 30초 간격으로 최대 30번(15분) 다시 올린다 |
| zip 패키지 (컨테이너 이미지 아님) | 이 PC에 Docker가 없다. scikit-learn, scipy, numpy의 리눅스용 휠을 `pip --platform manylinux...`로 받아 zip을 만든다. Lambda는 코드 폴더(/var/task)에 쓸 수 없어 콜드 스타트마다 라이브러리를 다시 컴파일하게 되므로 .pyc를 미리 만들어 넣는다. Python 3.12에서 빌드할 때만 만들고, 해시 기반 무효화를 쓰며, 소스 경로는 `/var/task/...`로 기록한다. 압축 해제 198.94MB로 Lambda 한도 250MB 안이다. zip이 직접 업로드 한도 50MB를 넘어 S3를 거쳐 올린다 |
| python3.12, x86_64 | 로컬 학습 환경과 같은 Python, 라이브러리 버전으로 맞춰 저장한 모델을 그대로 불러온다. python3.12 런타임은 Amazon Linux 2023(glibc 2.34) 기반이라 manylinux_2_28 휠이 동작한다. AWS 런타임 표 기준 지원 종료는 2028-10-31이다 |
| 서울 리전 | 사용자(한국)와 가깝고 필요한 서비스가 모두 있다. 데이터가 공개 자료라 위치 제약이 없다 |
| 메모리 1024MB, 제한 시간 30초 | 메모리를 늘리면 CPU도 비례해서 늘어 scikit-learn을 불러오는 콜드 스타트가 빨라진다. 실제 사용량은 실행 후 REPORT 로그로 확인한다 |
| 예약 동시성 2 | 실수로 반복 호출해도 동시에 2개까지만 돌아 비용과 영향 범위가 제한된다. 새 계정은 동시 실행 할당량이 낮아 예약이 거절될 수 있는데, 그 경우만 기록에 남기고 계속한다. 권한 오류 같은 다른 오류는 실패로 처리한다 |
| 배포마다 함수 버전 게시 | 버전마다 코드 sha256, 환경 변수(모델 객체 버전 포함), 버전 설명(git 커밋, 모델 버전, 패키지 sha)이 고정된다. 커밋하지 않은 코드로 배포하면 설명에 `+dirty`가 붙는다. 검사와 평가는 `$LATEST`가 아니라 같은 실행의 `deploy`가 게시한 번호(따로 실행하면 가장 최근 번호)로 호출해서, 어느 코드와 모델이 답했는지 기록에 남긴다 |

### 설정

| 파일 | 내용 |
|---|---|
| [config/pipeline.json](../config/pipeline.json) | 리전, 함수와 역할 이름, 버킷 접두어, 태그, 분할 기간, 하이퍼파라미터, 합격 기준. `cloud.bucket`을 적으면 그 버킷을 쓰는데, 이름이 `bike-demand-mlops-`로 시작하고 이 과제의 태그가 붙은 버킷이어야 한다. 버킷뿐 아니라 IAM 역할, Lambda 함수, 로그 그룹도 이 과제의 태그(`project`, `owner`)가 없으면 배포, release, 정리에서 건드리지 않고 멈춘다. 예외로, 이름이 `bike-demand-mlops-<리전>-<16진수 8자리>` 형식이고 태그도 객체도 없는 빈 버킷은 이 도구가 버킷을 만든 직후 태그를 붙이기 전에 멈춰 남은 것으로 보고, 배포는 태그를 붙여 이어받고 정리는 지운다 |
| [config/lambda.json](../config/lambda.json) | 런타임, 메모리, 제한 시간, 임시 저장소, 예약 동시성, 로그 보관 일수 |
| Lambda 환경 변수 | `MODEL_URI`(S3 경로와 객체 버전), `MODEL_VERSION`, `MODEL_SHA256`, `JOBLIB_MULTIPROCESSING=0` |

`JOBLIB_MULTIPROCESSING=0`을 두는 이유: Lambda에는 /dev/shm이 없어 joblib을 불러올 때 경고가 난다. 예측은 한 프로세스에서 하므로 joblib 병렬 처리를 끈다.

Lambda 실행 역할 `bike-demand-predict-role`의 권한은 두 가지뿐이다([bikeml/cloud.py](../bikeml/cloud.py)의 `role_policy`).
- 자기 로그 그룹에 로그 쓰기 (`logs:CreateLogStream`, `logs:PutLogEvents`)
- 버킷의 `models/*` 읽기 (`s3:GetObject`, `s3:GetObjectVersion`)

### 배포 전 확인

`deploy`는 AWS에 접속하기 전에 다음을 확인하고, 하나라도 틀리면 멈춘다.

1. 모델 파일의 sha256이 metadata.json과 같고, 정리한 자료가 설정의 분할 기간과 맞으며, 모델이 지금 학습 자료(`train.csv` sha256), 하이퍼파라미터, 버전, 특성 코드로 학습됐다.
2. 이 모델로 `evaluate`를 통과한 기록(`results/evaluation/metrics.json`)이 있고, 그때의 합격 기준과 평가 자료 sha256이 지금 설정, 지금 파일과 같다.
3. 이 모델로 `check-local`을 통과한 기록(`results/service_check/local.json`)이 있고, 그때의 Lambda 코드 해시가 지금 코드와 같다.
4. `build/lambda_package.zip`의 해시가 `results/cloud/package.json`과 같다.
5. zip에 넣은 Lambda 코드와 `requirements-lambda.txt`가 지금 파일과 같고, zip의 런타임과 아키텍처가 Lambda 설정과 같다.

그래서 평가에 떨어진 모델, 검사하지 않은 코드, 이전 코드로 만든 zip은 배포되지 않는다. 또 `train`은 후보 모델을 `models/_candidate/`에 만들고 `evaluate`를 통과해야만 `models/hgb-v1/`로 복사하므로, 평가에 떨어진 모델이 이미 합격한 모델을 덮지 않는다. `evaluate`와 `deploy`는 모델의 학습 자료 sha256, 하이퍼파라미터, 버전, 특성 코드가 지금과 같은지, 정리한 자료가 설정의 분할 기간과 맞는지도 확인한다.

## 2. 데이터와 결과의 위치

| 대상 | 위치 |
|---|---|
| 원자료, 정리한 자료 | Git: `data/raw/`, `data/processed/` |
| 모델 | Git: `models/hgb-v1/`, 클라우드: `s3://<버킷>/models/hgb-v1/<모델 sha 12자리>/` |
| 배포 패키지 | 로컬: `build/lambda_package.zip` (Git 제외), 클라우드: `s3://<버킷>/packages/lambda-<sha12>.zip` |
| 요청 예시 | `samples/service_cases.json`, `samples/generated/request_2012-12-03.json` |
| 클라우드 응답과 평가 | Git: `results/service_check/cloud.json`, `results/cloud/cloud_eval.json`, `results/cloud/predictions_cloud.csv` |
| 배포 기록 | Git: `results/cloud/package.json`, `results/cloud/deploy.json`. 오류 재현용 배포는 `deploy_override.json`에 따로 남겨 정상 배포 기록을 덮지 않는다 |
| 함수 로그 | CloudWatch Logs `/aws/lambda/bike-demand-predict` (7일). 필요한 부분은 Git 기록으로 옮긴다 |
| 실행 기록 | Git: `runs/run_log.jsonl` |

Lambda는 S3에서 받은 코드의 사본을 따로 보관한다(기본 저장 방식). 그래서 수명 주기 규칙으로 `packages/`의 zip이 지워져도 배포된 함수에는 영향이 없다.

`deploy`는 자원을 만들기 전에 계정(sha256 해시)과 리전을 `.state/regions.json`(Git 제외)에 적는다. `status`와 `cleanup`은 지금 리전, 이 파일에서 같은 계정의 리전, 배포 기록(`results/cloud/deploy.json`, `deploy_override.json`)의 리전을 모두 조회하고 정리한다. 다른 계정의 배포 기록이 남아 있으면 `cleanup`은 실패로 기록하고 그 계정의 프로필로 다시 실행하라고 알린다. 다른 리전에서 배포하다 중간에 실패해 남은 자원도 놓치지 않기 위해서다. IAM 역할은 리전과 상관없는 자원이라 모든 리전을 정리한 뒤 마지막에 지우고, 역할의 인라인 정책은 리전마다 따로 둔다.

공개 저장소에 계정 정보가 남지 않게 했다([bikeml/runlog.py](../bikeml/runlog.py)의 `sanitize`).
- 버킷 이름에는 계정 ID 대신 임의 문자열을 넣는다. 다음 실행부터는 같은 접두어의 버킷을 목록에서 찾아 쓴다.
- 기록에는 호출자의 이름 대신 종류(`root`, `user`, `assumed-role`)만 남기고, 화면에도 종류만 보여 준다.
- 기록과 결과 파일을 저장할 때 다음을 바꾼다. JSON 안에서 이스케이프된 경로도 포함한다.
  - 계정 ID는 `<ACCOUNT_ID>`로 바꾼다.
  - 저장소 경로는 `<REPO>`, 홈 폴더 경로는 `~`로 바꾼다.
  - 그 밖의 `Users\이름`은 `<USER>`로 바꾼다.
  - AWS 오류 메시지 속 IAM 사용자나 세션 이름(`user/이름`, `assumed-role/...`)은 `<NAME>`으로 바꾼다.

## 3. 실제 입력에서 결과까지 (실행 결과)

실행 전이다. 실행 순서와 확인할 것은 다음과 같다.

1. `python -m bikeml --profile <프로필> run cloud`: package, deploy, smoke, cloud-eval, release를 순서대로 실행한다.
2. deploy: 배포 전 확인, 버킷·역할·로그 그룹 생성, 모델과 패키지 업로드, 함수 생성, 버전 게시. 결과는 `results/cloud/deploy.json`
3. smoke: 로컬과 같은 12개 사례를 Lambda에 보낸다. 응답, 요청 ID, 실행 시간, 메모리 사용량(REPORT 로그), Lambda 안의 라이브러리 버전을 남긴다. 결과는 `results/service_check/cloud.json`
4. cloud-eval: 평가 자료 2,888건을 500건씩 6번 보내 MAE를 계산하고 로컬 예측과 같은지 비교한다. 결과는 `results/cloud/cloud_eval.json`
5. release: 두 검사를 모두 통과했고 정체가 같은 버전에 `live` 별칭을 붙인다. 결과는 `results/cloud/release.json`(버전 정체와 이전 `live` 버전 포함)

## 4. 실행 기록과 실패 처리

[bikeml/runlog.py](../bikeml/runlog.py)가 단계마다 `runs/run_log.jsonl`에 한 줄을 남긴다. `python -m bikeml runs`로 표를 볼 수 있다.

| 필드 | 내용 |
|---|---|
| `run_id`, `step` | 실행 묶음 ID(시각과 임의값), 단계 이름 |
| `started_at`, `finished_at`, `duration_s` | 시각(실행한 PC의 시간대 오프셋 포함, 이 PC는 +09:00), 걸린 시간 |
| `status` | success, failed, skipped |
| `git.commit`, `git.code_dirty` | 코드 커밋, 커밋하지 않은 코드 변경 여부 |
| `model` | 모델 이름, 버전, sha256 |
| `inputs`, `configs` | 입력 파일과 설정 파일의 경로, sha256 |
| `args` | 명령 인자(설정 파일, 프로필, 리전, Lambda 설정 파일 등) |
| `cloud` | 리전, 함수, 게시 버전, 별칭, 코드 sha256, 모델과 패키지의 S3 경로 |
| `result_location` | 결과 파일 위치 (실패하면 `results/failed/...`) |
| `summary`, `error`, `env` | 주요 수치, 오류 메시지, Python과 라이브러리 버전 |

- 각 단계는 결과를 임시 폴더에 먼저 쓴다. 성공하면 제자리(`results/`, `models/` 등)로 옮긴다. 옮기다 실패하거나 Ctrl+C로 멈추면 이미 옮긴 파일을 원래대로 되돌린다.
- 단계가 실패하면 결과를 `results/failed/<run_id>_<단계>/`로 옮긴다. 정상 결과를 덮어쓰지 않고, 오류 추적은 그 폴더의 `error.txt`에 남는다. Ctrl+C로 멈춘 경우도 실패로 기록한다.
- 한 단계가 실패하면 뒤 단계는 실행하지 않고 `skipped`로 기록한다.
- 외부 모델은 쓰지 않는다. Lambda 런타임에 들어 있는 boto3는 AWS가 관리해서 버전을 고정할 수 없다. 대신 응답의 `runtime`과 서비스 검사 결과의 `runtime`에 실제 버전이 기록된다. 이 boto3는 모델 파일을 내려받는 데만 쓴다.

실패 처리 예시 (AWS 없이 실행함)

합격 기준만 90%로 올린 설정([config/demo_strict_criteria.json](../config/demo_strict_criteria.json))으로 평가를 다시 돌렸다.

```
python -m bikeml --config config/demo_strict_criteria.json run evaluate check-local
```

| 항목 | 결과 |
|---|---|
| run_id | `20261119T230144-db75ff` |
| evaluate | failed. `RuntimeError: 합격 기준 미달: {'beats_all_baselines': True, 'mae_reduction_vs_hour_baseline': 0.499, 'meets_reduction': False}` |
| check-local | skipped (`이전 단계 실패로 실행하지 않음: evaluate`) |
| 실패한 결과 | `results/failed/20261119T230144-db75ff_evaluate/` (metrics.json의 `passed: false`, error.txt) |
| 정상 결과 | `results/evaluation/metrics.json`은 그대로 (`passed: true`, 기준 0.3). 후보 모델은 `models/hgb-v1/`로 옮겨지지 않았다 |
| 배포 | 정상 설정으로는 배포 전 확인을 통과한다. 엄격한 설정으로 `deploy`를 실행하면 "evaluate 이후 합격 기준이나 평가 자료가 바뀜"으로 멈춘다 |

같은 설정으로 이전 코드에서 실행한 기록(`20261119T212432-8dab61`, `20261119T215025-2b3b55`, `20261119T221433-76275e`, `20261119T223856-cfa909`)도 `results/failed/`에 남아 있다.

## 5. 오류 재현과 복구

실행 전이다. 과제 예시 중 "잘못된 모델 경로"를 클라우드에서 재현할 계획이다.

1. 재현: `python -m bikeml --profile <프로필> run deploy smoke cloud-eval release --model-uri-override s3://<버킷>/models/hgb-v1/model.pkl` (실제 파일은 `model.joblib`). 이 배포 기록은 `results/cloud/deploy_override.json`에 따로 남는다.
2. 예상 영향: 모든 요청이 `500 MODEL_LOAD_FAILED`, smoke 실패, cloud-eval과 release는 실행하지 않음(skipped). release가 실행되지 않으므로 `live` 별칭은 이전 정상 버전을 그대로 가리킨다.
3. 원인 조사: 응답 메시지와 CloudWatch 로그를 보고, 운영자 권한으로 `aws s3api head-object`를 실행해 비교한다. S3는 `s3:ListBucket` 권한이 없는 요청자에게 없는 객체를 404가 아니라 403(Forbidden)으로 알린다. Lambda 역할에는 이 권한이 없어서 403으로 보일 것으로 예상하며, 이 점도 확인한다.
4. 수정: 경로를 바로잡아 다시 배포하고, 배포 단계에서 `MODEL_URI`가 가리키는 객체가 있는지와 sha256이 맞는지 확인하는 코드를 추가한다.
5. 재검사: 실패했던 같은 명령과 정상 입력(smoke 12건, cloud-eval 2,888건)을 다시 실행해 결과를 남긴다.
