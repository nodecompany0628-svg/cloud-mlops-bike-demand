# 클라우드 실행

진행 상태: 구성과 코드는 완성했다. 배포 패키지 빌드(`package`)는 로컬에서 실행했다. 배포부터 정리까지의 전 과정은 모의 AWS(moto) 테스트로 확인했다. 실제 AWS에서의 배포(`deploy`), 클라우드 검사(`smoke`), 클라우드 평가(`cloud-eval`), 오류 재현, 정리는 아직 실행하지 않았다. 실행하면 3절과 5절, 그리고 [operations.md](operations.md)를 실제 기록으로 채운다.

## 1. 구성과 선택 이유

```
운영 담당자 PC (python -m bikeml ...)          AWS ap-northeast-2 (서울)
  package    리눅스용 zip 빌드 (53MB)
  deploy     모델과 패키지 업로드 ----------->  S3 bike-demand-mlops-ap-northeast-2-<임의 8자리> (비공개, 버전 관리)
                                                  models/hgb-v1/model.joblib, metadata.json
                                                  packages/lambda-<sha 12자리>.zip
             함수 생성 또는 갱신, 버전 게시 -->  Lambda bike-demand-predict (python3.12, 1024MB, 30초)
  smoke      검사 사례 12개 호출 ------------->    콜드 스타트 때 S3에서 모델을 한 번 받아 sha256 확인
  cloud-eval 평가 자료 2,888건, 500건씩 ----->    입력 검사, 특성 변환, 예측
             <------------ 예측(대/시간) ----    로그: CloudWatch Logs /aws/lambda/bike-demand-predict (7일 보관)
  results/, runs/ 에 기록 (Git)
```

| 선택 | 이유 |
|---|---|
| AWS Lambda | 운영 담당자가 하루 한두 번 다음 날 24시간 예보를 넣어 호출하는 용도다. 상시 서버나 화면이 필요 없고, 호출할 때만 실행되어 요청이 없으면 비용이 없다. 모델이 작아(0.7MB) 1,000건 예측이 로컬 PC에서 23ms 걸린다. Lambda에서의 실제 시간은 배포 후 REPORT 로그로 기록한다 |
| 직접 호출 (`lambda:InvokeFunction`), 공개 주소 없음 | 사용자가 담당자 몇 명뿐이라 공개 HTTP 주소를 열 이유가 없다. IAM 권한이 있는 사람만 호출할 수 있어 인증을 따로 만들지 않았다 |
| 모델은 S3에 두고 실행할 때 내려받음 | 모델 버전을 코드 패키지와 따로 관리한다. 모델만 바꿀 때는 환경 변수 `MODEL_URI`만 바꾸면 된다. 버킷 버전 관리를 켜서 덮어써도 이전 파일이 남는다 |
| zip 패키지 (컨테이너 이미지 아님) | 이 PC에 Docker가 없다. scikit-learn, scipy, numpy의 리눅스용 휠을 `pip --platform manylinux...`로 받아 zip을 만든다. 압축 해제 169MB로 Lambda 한도 250MB 안이다. zip이 직접 업로드 한도 50MB를 넘어 S3를 거쳐 올린다 |
| python3.12, x86_64 | 로컬 학습 환경과 같은 Python, 라이브러리 버전으로 맞춰 저장한 모델을 그대로 불러온다. python3.12 런타임은 Amazon Linux 2023(glibc 2.34) 기반이라 manylinux_2_28 휠이 동작한다. AWS 런타임 표 기준 지원 종료는 2028-10-31이다 |
| 서울 리전 | 사용자(한국)와 가깝고 필요한 서비스가 모두 있다. 데이터가 공개 자료라 위치 제약이 없다 |
| 메모리 1024MB, 제한 시간 30초 | 메모리를 늘리면 CPU도 비례해서 늘어 scikit-learn을 불러오는 콜드 스타트가 빨라진다. 실제 사용량은 실행 후 REPORT 로그로 확인한다 |
| 예약 동시성 2 | 실수로 반복 호출해도 동시에 2개까지만 돌아 비용과 영향 범위가 제한된다. 새 계정은 동시 실행 할당량이 낮아 예약이 거절될 수 있는데, 그 경우만 기록에 남기고 계속한다. 권한 오류 같은 다른 오류는 실패로 처리한다 |
| 배포마다 함수 버전 게시 | 버전마다 코드 sha256과 설명(git 커밋, 모델 버전, 패키지 sha)이 고정된다. 커밋하지 않은 코드로 배포하면 설명에 `+dirty`가 붙는다. 검사와 평가는 `$LATEST`가 아니라 가장 최근에 게시한 번호로 호출해서, 어느 코드와 모델이 답했는지 기록에 남긴다. 코드와 설정이 그대로면 AWS가 새 번호를 만들지 않고 기존 번호를 돌려준다 |

### 설정

| 파일 | 내용 |
|---|---|
| [config/pipeline.json](../config/pipeline.json) | 리전, 함수와 역할 이름, 버킷 접두어, 태그, 분할 기간, 하이퍼파라미터, 합격 기준. `cloud.bucket`을 적으면 그 버킷을 쓴다 |
| [config/lambda.json](../config/lambda.json) | 런타임, 메모리, 제한 시간, 임시 저장소, 예약 동시성, 로그 보관 일수 |
| Lambda 환경 변수 | `MODEL_URI`(S3 경로), `MODEL_VERSION`, `MODEL_SHA256`, `JOBLIB_MULTIPROCESSING=0` |

`JOBLIB_MULTIPROCESSING=0`을 두는 이유: Lambda에는 /dev/shm이 없어 joblib을 불러올 때 경고가 난다. 예측은 한 프로세스에서 하므로 joblib 병렬 처리를 끈다.

Lambda 실행 역할 `bike-demand-predict-role`의 권한은 두 가지뿐이다([bikeml/cloud.py](../bikeml/cloud.py)의 `role_policy`).
- 자기 로그 그룹에 로그 쓰기 (`logs:CreateLogStream`, `logs:PutLogEvents`)
- 버킷의 `models/*` 읽기 (`s3:GetObject`)

### 배포 전 확인

`deploy`는 AWS에 접속하기 전에 다음을 확인한다. 하나라도 틀리면 멈춘다.

1. 모델 파일의 sha256이 metadata.json과 같다.
2. 이 모델로 `evaluate`를 통과한 기록(`results/evaluation/metrics.json`)이 있다.
3. 이 모델로 `check-local`을 통과한 기록(`results/service_check/local.json`)이 있다.
4. `build/lambda_package.zip`의 해시가 `results/cloud/package.json`과 같다.
5. zip에 넣은 Lambda 코드와 `requirements-lambda.txt`가 지금 파일과 같다.

그래서 평가에 떨어진 모델이나 이전 코드로 만든 zip은 배포되지 않는다.

## 2. 데이터와 결과의 위치

| 대상 | 위치 |
|---|---|
| 원자료, 정리한 자료 | Git: `data/raw/`, `data/processed/` |
| 모델 | Git: `models/hgb-v1/`, 클라우드: `s3://<버킷>/models/hgb-v1/` |
| 배포 패키지 | 로컬: `build/lambda_package.zip` (Git 제외), 클라우드: `s3://<버킷>/packages/lambda-<sha12>.zip` |
| 요청 예시 | `samples/service_cases.json`, `samples/generated/request_2012-12-03.json` |
| 클라우드 응답과 평가 | Git: `results/service_check/cloud.json`, `results/cloud/cloud_eval.json`, `results/cloud/predictions_cloud.csv` |
| 배포 기록 | Git: `results/cloud/package.json`, `results/cloud/deploy.json` |
| 함수 로그 | CloudWatch Logs `/aws/lambda/bike-demand-predict` (7일). 필요한 부분은 Git 기록으로 옮긴다 |
| 실행 기록 | Git: `runs/run_log.jsonl` |

Lambda는 S3에서 받은 코드의 사본을 따로 보관한다(기본 저장 방식). 그래서 수명 주기 규칙으로 `packages/`의 zip이 지워져도 배포된 함수에는 영향이 없다.

공개 저장소에 계정 정보가 남지 않게 했다.
- 버킷 이름에는 계정 ID 대신 임의 문자열을 넣는다. 다음 실행부터는 같은 접두어의 버킷을 목록에서 찾아 쓴다.
- 기록에는 호출자의 이름 대신 종류(`root`, `user`, `assumed-role`)만 남긴다.
- 계정 ID, 저장소 경로, PC 사용자 이름은 기록과 결과 파일을 저장할 때 `<ACCOUNT_ID>`, `<REPO>`, `<USER>`로 바꾼다([bikeml/runlog.py](../bikeml/runlog.py)의 `sanitize`).

## 3. 실제 입력에서 결과까지 (실행 결과)

실행 전이다. 실행 순서와 확인할 것은 다음과 같다.

1. `python -m bikeml --profile <프로필> run cloud`: package, deploy, smoke, cloud-eval을 순서대로 실행한다.
2. deploy: 배포 전 확인, 버킷·역할·로그 그룹 생성, 모델과 패키지 업로드, 함수 생성, 버전 게시. 결과는 `results/cloud/deploy.json`
3. smoke: 로컬과 같은 12개 사례를 Lambda에 보낸다. 응답, 요청 ID, 실행 시간, 메모리 사용량(REPORT 로그), Lambda 안의 라이브러리 버전을 남긴다. 결과는 `results/service_check/cloud.json`
4. cloud-eval: 평가 자료 2,888건을 500건씩 6번 보내 MAE를 계산하고 로컬 예측과 같은지 비교한다. 결과는 `results/cloud/cloud_eval.json`

## 4. 실행 기록과 실패 처리

[bikeml/runlog.py](../bikeml/runlog.py)가 단계마다 `runs/run_log.jsonl`에 한 줄을 남긴다. `python -m bikeml runs`로 표를 볼 수 있다.

| 필드 | 내용 |
|---|---|
| `run_id`, `step` | 실행 묶음 ID(시각과 임의값), 단계 이름 |
| `started_at`, `finished_at`, `duration_s` | 시각(KST, 오프셋 포함), 걸린 시간 |
| `status` | success, failed, skipped |
| `git.commit`, `git.code_dirty` | 코드 커밋, 커밋하지 않은 코드 변경 여부 |
| `model` | 모델 이름, 버전, sha256 |
| `inputs`, `configs` | 입력 파일과 설정 파일의 경로, sha256 |
| `args` | 명령 인자(프로필, 리전, Lambda 설정 파일 등) |
| `cloud` | 리전, 함수, 게시 버전, 코드 sha256, 모델과 패키지의 S3 경로 |
| `result_location` | 결과 파일 위치 (실패하면 `results/failed/...`) |
| `summary`, `error`, `env` | 주요 수치, 오류 메시지, Python과 라이브러리 버전 |

- 각 단계는 결과를 임시 폴더에 먼저 쓴다. 성공하면 제자리(`results/`, `models/` 등)로 옮긴다. 옮기다 실패하면 이미 옮긴 파일을 원래대로 되돌린다.
- 단계가 실패하면 결과를 `results/failed/<run_id>_<단계>/`로 옮긴다. 정상 결과를 덮어쓰지 않고, 오류 추적은 그 폴더의 `error.txt`에 남는다.
- 한 단계가 실패하면 뒤 단계는 실행하지 않고 `skipped`로 기록한다. 예를 들어 클라우드 검사가 실패하면 클라우드 평가를 하지 않는다.
- 외부 모델은 쓰지 않는다. Lambda 런타임에 들어 있는 boto3는 AWS가 관리해서 버전을 고정할 수 없다. 대신 응답의 `runtime`에 실제 버전이 기록된다. 이 boto3는 모델 파일을 내려받는 데만 쓴다.

## 5. 오류 재현과 복구

실행 전이다. 과제 예시 중 "잘못된 모델 경로"를 클라우드에서 재현할 계획이다.

1. 재현: `python -m bikeml --profile <프로필> run deploy smoke cloud-eval --model-uri-override s3://<버킷>/models/hgb-v1/model.pkl` (실제 파일은 `model.joblib`)
2. 예상 영향: 모든 요청이 `500 MODEL_LOAD_FAILED`, smoke 실패, cloud-eval은 실행하지 않음(skipped)
3. 원인 조사: 응답 메시지와 CloudWatch 로그를 보고, 운영자 권한으로 `aws s3api head-object`를 실행해 비교한다. Lambda 역할에는 `s3:ListBucket`이 없어서, 없는 객체도 404가 아니라 403(Forbidden)으로 보일 것으로 예상한다. S3는 ListBucket 권한이 없는 요청자에게 없는 객체를 403으로 알린다. 이 점도 확인한다.
4. 수정: 경로를 바로잡아 다시 배포하고, 배포 단계에서 `MODEL_URI`가 가리키는 객체가 있는지와 sha256이 맞는지 확인하는 코드를 추가한다.
5. 재검사: 실패했던 같은 명령과 정상 입력(smoke 12건, cloud-eval 2,888건)을 다시 실행해 결과를 남긴다.
