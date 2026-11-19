# 클라우드 실행

> **진행 상태**: 구성과 코드는 완성했고, 배포 패키지 빌드(`package`)까지 로컬에서 실행했습니다.
> AWS 배포(`deploy`), 클라우드 검사(`smoke`), 클라우드 평가(`cloud-eval`), 오류 재현, 정리는 **아직 실행 전**입니다.
> 실행하면 이 문서의 "실행 결과" 칸과 [operations.md](operations.md)를 실제 기록으로 채웁니다.

## 1. 구성과 선택 이유

```
 운영 담당자 PC (python -m bikeml ...)                     AWS ap-northeast-2 (서울)
 ─────────────────────────────────────            ───────────────────────────────────────────────
  package  리눅스용 zip 빌드 (53MB)
  deploy   ── 모델, 패키지 업로드 ───────────────▶  S3  bike-demand-mlops-<해시>-ap-northeast-2  (비공개, 버전 관리)
           │                                         ├ models/hgb-v1/model.joblib, metadata.json
           │                                         └ packages/lambda-<sha12>.zip
           └── 함수 생성·갱신, 버전 게시 ─────────▶  Lambda  bike-demand-predict  (python3.12, 1024MB, 30초)
  smoke    ── invoke(JSON) ──────────────────────▶    │ 콜드 스타트 때 S3 에서 모델을 한 번 받아 sha256 확인
  cloud-eval  (평가 자료 2,888건, 500건씩)              │ 입력 검사 → 특성 변환 → 예측
           ◀──────────── 예측(대/시간) ──────────    └ 로그 → CloudWatch Logs /aws/lambda/bike-demand-predict (7일 보관)
  results/, runs/ 에 기록 (Git)
```

| 선택 | 이유 |
|---|---|
| **AWS Lambda** (서버리스 함수) | 운영 담당자가 하루 한두 번 "내일 24시간 예보"를 넣어 호출하는 용도입니다. 상시 서버나 화면이 필요 없고, 호출할 때만 실행되어 요청이 없으면 비용이 0입니다. 모델이 작아(0.7MB) 한 번에 1,000건까지 1초 안팎으로 예측할 수 있습니다. |
| 직접 호출(`lambda:InvokeFunction`), 공개 URL 없음 | 사용자가 담당자 몇 명뿐이라 공개 HTTP 주소를 열 이유가 없습니다. IAM 권한이 있는 사람만 호출하므로 인증을 따로 만들 필요가 없습니다. 나중에 HTTP가 필요하면 핸들러가 `{"body": "..."}` 형태도 이미 받습니다. |
| **S3**에 모델 보관, 실행 때 내려받기 | 모델 버전을 코드 패키지와 떼어 관리합니다. 모델만 바꿀 때는 환경 변수 `MODEL_URI`만 바꾸면 됩니다. 버킷 버전 관리를 켜서 덮어써도 이전 파일이 남습니다. |
| zip 패키지 (컨테이너 이미지 아님) | 이 PC에 Docker가 없습니다. scikit-learn, scipy, numpy의 리눅스용 휠을 `pip --platform manylinux…`로 받아 zip을 만듭니다. 압축 해제 169MB로 Lambda 한도 250MB 안입니다. zip이 50MB를 넘어 S3를 거쳐 올립니다. |
| python3.12, x86_64 | 로컬 학습 환경과 같은 Python과 라이브러리 버전으로 맞춰, 저장한 모델을 그대로 불러올 수 있게 했습니다. |
| 서울 리전 | 사용자(한국)와 가깝고, 필요한 서비스가 모두 있습니다. 데이터는 공개 자료라 위치 제약이 없습니다. |
| 메모리 1024MB, 제한 시간 30초 | 메모리를 늘리면 CPU도 비례해 늘어 scikit-learn 불러오기(콜드 스타트)가 빨라집니다. 실제 사용량은 실행 후 REPORT 로그로 확인합니다. |
| 동시 실행 2 (예약 동시성) | 실수로 반복 호출해도 동시에 2개까지만 돌아 비용과 영향 범위를 제한합니다. 계정 한도 때문에 설정이 거절되면 기록에 남깁니다. |
| 배포할 때마다 함수 버전 게시 | 각 버전은 코드 sha256과 설명(`git 커밋, 모델 버전, 패키지 sha`)이 고정됩니다. 검사와 평가는 `$LATEST`가 아니라 방금 게시한 번호로 호출해서, 어느 코드와 모델이 답했는지 기록에 남습니다. 패키지 zip은 파일 순서와 시각을 고정해 같은 코드·라이브러리면 sha256이 같습니다. pip가 만드는 Windows 실행 파일(`bin/`)은 빌드마다 내용이 달라 뺐습니다. |

### 실행 설정

| 파일 | 내용 |
|---|---|
| [config/pipeline.json](../config/pipeline.json) | 리전, 함수·역할 이름, 버킷 접두어, 태그, 분할 기간, 모델 하이퍼파라미터, 합격 기준 |
| [config/lambda.json](../config/lambda.json) | 런타임, 메모리, 제한 시간, 임시 저장소, 예약 동시성, 로그 보관 일수 |
| Lambda 환경 변수 | `MODEL_URI`(S3 경로), `MODEL_VERSION`, `MODEL_SHA256` |

Lambda 실행 역할(`bike-demand-predict-role`)의 권한은 두 가지뿐입니다([bikeml/cloud.py](../bikeml/cloud.py) `role_policy`).
- 자기 로그 그룹에 로그 쓰기 (`logs:CreateLogStream`, `logs:PutLogEvents`)
- 버킷의 `models/*` 읽기 (`s3:GetObject`)

## 2. 데이터와 결과의 위치

| 대상 | 위치 |
|---|---|
| 원자료, 정리한 자료 | Git: `data/raw/`, `data/processed/` |
| 모델 | Git: `models/hgb-v1/`, 클라우드: `s3://<버킷>/models/hgb-v1/` |
| 배포 패키지 | 로컬: `build/lambda_package.zip` (Git 제외), 클라우드: `s3://<버킷>/packages/lambda-<sha12>.zip` |
| 요청 예시 | `samples/service_cases.json`, `samples/generated/request_2012-12-03.json` |
| 클라우드 응답과 평가 | Git: `results/service_check/cloud.json`, `results/cloud/cloud_eval.json`, `results/cloud/predictions_cloud.csv` |
| 배포 기록 | Git: `results/cloud/package.json`, `results/cloud/deploy.json` |
| 함수 로그 | CloudWatch Logs `/aws/lambda/bike-demand-predict` (7일). 필요한 부분은 Git의 기록으로 옮김 |
| 실행 기록 | Git: `runs/run_log.jsonl` |

버킷 이름에는 계정 ID 대신 계정 ID의 sha256 앞 10자리를 넣습니다. 공개 저장소의 기록에 계정 ID가 드러나지 않게 하려는 것이며, 기록을 쓸 때 계정 ID 문자열은 `<ACCOUNT_ID>`로 바꿉니다([bikeml/runlog.py](../bikeml/runlog.py) `sanitize`).

## 3. 실제 입력에서 결과까지 (실행 결과)

⏳ **실행 전**. 실행 순서와 확인할 것:

1. `python -m bikeml run cloud` → package → deploy → smoke → cloud-eval이 순서대로 실행됩니다.
2. `deploy`: 버킷·역할·로그 그룹 생성, 모델·패키지 업로드, 함수 생성, 버전 게시 → `results/cloud/deploy.json`
3. `smoke`: 로컬과 같은 13개 사례를 Lambda에 보냅니다. 응답, 요청 ID, 실행 시간, 메모리 사용량(REPORT 로그) → `results/service_check/cloud.json`
4. `cloud-eval`: 평가 자료 2,888건을 500건씩 6번 보내 MAE를 계산하고, 로컬 예측과 같은지 비교합니다 → `results/cloud/cloud_eval.json`

## 4. 실행 기록과 실패 처리

[bikeml/runlog.py](../bikeml/runlog.py)가 단계마다 `runs/run_log.jsonl`에 한 줄을 남깁니다. `python -m bikeml runs`로 표를 볼 수 있습니다.

| 필드 | 내용 |
|---|---|
| `run_id`, `step` | 실행 묶음 ID(시각+임의값), 단계 이름 |
| `started_at`, `finished_at`, `duration_s` | 시각(KST, 오프셋 포함), 걸린 시간 |
| `status` | success / failed / skipped |
| `git.commit`, `git.code_dirty` | 코드 커밋, 커밋하지 않은 코드 변경 여부 |
| `model` | 모델 이름, 버전, sha256 |
| `inputs`, `configs` | 입력 파일과 설정 파일의 경로·sha256 |
| `args` | 명령 인자(프로필, 리전, Lambda 설정 파일 등) |
| `cloud` | 리전, 함수, 게시 버전, 코드 sha256, 모델·패키지 S3 경로 |
| `result_location` | 결과 파일 위치 (실패하면 `results/failed/…`) |
| `summary`, `error`, `env` | 주요 수치, 오류 메시지, Python·라이브러리 버전 |

- 각 단계는 결과를 임시 폴더에 먼저 씁니다. 성공하면 제자리(`results/`, `models/` …)로 옮기고, **실패하면 `results/failed/<run_id>_<단계>/`로 옮겨** 정상 결과를 덮어쓰지 않습니다. 오류 추적은 그 폴더의 `error.txt`에 남습니다.
- 한 단계가 실패하면 **뒤 단계는 실행하지 않고** `skipped`로 기록합니다. 예: 평가가 합격 기준에 못 미치면 배포하지 않고, 클라우드 검사가 실패하면 클라우드 평가를 하지 않습니다.
- 외부 모델은 쓰지 않습니다. Lambda 런타임에 들어 있는 boto3는 AWS가 관리해서 버전을 고정할 수 없습니다. 이 boto3는 모델 파일을 내려받는 데만 쓰입니다.

## 5. 오류 재현과 복구

⏳ **실행 전**. 계획은 다음과 같습니다. 과제 예시 중 "잘못된 모델 경로"를 클라우드에서 재현합니다.

1. 재현: `python -m bikeml run deploy smoke cloud-eval --model-uri-override s3://<버킷>/models/hgb-v1/model.pkl`
   (실제 파일은 `model.joblib`. 오타가 난 경로로 배포)
2. 기대 영향: 모든 요청이 `500 MODEL_LOAD_FAILED`, smoke 실패, cloud-eval은 실행하지 않음(skipped)
3. 원인 조사: 응답 메시지, CloudWatch 로그, 운영자 권한으로 `s3api head-object`를 실행해 비교. Lambda 역할에는 `s3:ListBucket`이 없어서, 없는 객체도 404가 아니라 403(Forbidden)으로 보일 것으로 예상합니다. 이것도 확인합니다.
4. 수정: 경로를 바로잡아 다시 배포하고, 배포 전에 모델 객체가 있는지와 sha256이 맞는지 확인하는 단계를 코드에 추가합니다.
5. 재검사: 실패했던 같은 명령과 정상 입력(smoke 13건, cloud-eval 2,888건)을 다시 실행해 결과를 남깁니다.
