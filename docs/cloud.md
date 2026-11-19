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
2. `deploy`: 배포 전 확인을 한 뒤 권한 경계, 역할, 버킷, 로그 그룹을 차례로 준비하고 모델과 zip을 S3에 올린다. 함수를 만들거나 갱신하고 새 버전을 게시한다. 자원을 만들기 전에 거부할 이유를 먼저 모두 확인한다. 태그가 없는 남의 버킷, 역할, Lambda 함수, 로그 그룹, 이름이 맞는 버킷이 여러 개, 다른 리전에 있는 설정 버킷, 권한 경계가 없는 기존 역할, 내용이 설정과 다른 경계 정책, 운영자 키인데 아직 없는 경계 정책 중 하나라도 있으면 아무것도 만들지 않고 멈춘다.
3. Lambda는 콜드 스타트 때 S3에서 고정된 객체 버전의 모델을 한 번 받아 sha256과 입력 개수를 확인하고, 이후 요청은 메모리의 모델로 예측한다.
4. `smoke`: 검사 사례 12개를 같은 실행에서 `deploy`가 게시한 버전에 보낸다(`smoke`만 따로 실행하면 가장 최근에 게시된 버전). 같은 실행에서 `deploy`가 역할을 새로 만들었으면, 먼저 정상 요청 1건을 보내 새 역할 권한이 아직 반영되지 않아 모델 읽기가 403으로 실패하는지 본다. 그렇다면 10초 간격으로 다시 보내며, 첫 요청을 포함해 최대 6번(기다림은 최대 50초) 보낸다. 역할이 이미 있던 배포에서는 기다리지 않으므로, 잘못된 모델 경로로 생긴 403을 권한 대기로 착각하지 않는다. 결과는 `readiness`에 남는다.
5. `cloud-eval`: 평가 자료 2,888건을 같은 버전에 500건씩 보내 예측을 받고 로컬 결과와 비교한다.
6. `release`: 그 버전이 smoke와 cloud-eval을 모두 통과했을 때만 `live` 별칭을 그 버전으로 옮긴다. 버전 번호만 보지 않고, 실행 기록에서 버전의 정체(리전, 리비전 ID, 코드 sha256, 모델 경로와 sha256)가 같은 smoke와 cloud-eval의 가장 최근 기록이 둘 다 성공인지 확인한다. 같은 버전을 다시 검사해 실패했다면 옮기지 않는다. 함수를 지웠다 다시 만들면 버전 번호가 1부터 다시 시작하므로 번호만으로는 같은 버전인지 알 수 없기 때문이다. 이전에 `live`가 가리키던 버전도 기록한다.
7. 응답과 기록은 PC의 `results/`, `runs/`에 남기고 Git에 올린다. 함수 로그는 CloudWatch Logs에 남는다.

| 선택 | 이유 |
|---|---|
| AWS Lambda | 운영 담당자가 하루 한두 번 다음 날 24시간 예보를 넣어 호출하는 용도다. 상시 서버나 화면이 필요 없고, 호출할 때만 실행되어 요청이 없으면 비용이 없다. 모델이 0.7MB로 작아 함수 하나로 충분하다. Lambda에서의 실제 처리 시간과 메모리 사용량은 배포 후 REPORT 로그로 기록한다 |
| 직접 호출 (`lambda:InvokeFunction`), 공개 주소 없음 | 사용자가 담당자 몇 명뿐이라 공개 HTTP 주소를 열 이유가 없다. IAM 권한이 있는 사람만 호출할 수 있어 인증을 따로 만들지 않았다 |
| `live` 별칭 | 사용자는 `live`로 호출한다. `live`는 smoke와 cloud-eval을 모두 통과한 버전만 가리키므로, 배포나 검사 도중 실패해 `$LATEST`나 새 버전이 어중간한 상태여도 사용자 호출이 검사하지 않은 버전으로 가지 않는다. 메모리, 제한 시간, 환경 변수는 게시한 버전마다 고정되지만, 예약 동시성(2)과 실행 역할의 권한은 모든 버전이 함께 쓴다. 그래서 검사 중에는 `live` 호출과 동시 실행 자리를 나눠 쓰고, 예약 동시성 변경은 `live`에도 바로 적용된다. 역할 권한은 이 과제의 모든 버킷을 대상으로 잡아 두어, 버킷이 바뀌어도 `live`가 쓰던 모델을 계속 읽을 수 있다 |
| 모델은 S3에 두고 실행할 때 내려받음 | 모델 버전을 코드 패키지와 따로 관리한다. 모델 경로에 모델 sha256 앞 12자리를 넣어(`models/hgb-v1/<sha12>/model.joblib`) 내용이 다른 모델은 다른 객체가 되게 하고, Lambda 환경 변수 `MODEL_URI`에 S3 객체 버전 ID까지 적는다(`...model.joblib?versionId=...`). 그래서 다시 학습해 올려도 이미 게시한 Lambda 버전이나 `live`가 가리키는 모델 파일은 덮이거나 수명 주기 규칙으로 지워지지 않는다 |
| 같은 파일은 다시 올리지 않음 | 업로드 전에 S3의 현재 객체에 적힌 sha256을 보고, 같으면 그 객체 버전을 그대로 쓴다. 함수 설정의 설명은 고정 문구이고 커밋 정보는 버전 설명에만 넣는다. 그래서 모델, 코드, 설정이 그대로면 Lambda 설정도 그대로다. 이 경우 AWS 문서에 따르면 PublishVersion은 새 번호를 만들지 않고 기존 번호를 돌려주며, 버전 설명은 처음 게시할 때의 커밋으로 남는다. 이번 배포의 커밋은 `deploy.json`과 실행 기록에 남는다. 이 동작은 실제 AWS에서 확인할 부분이다(moto는 매번 새 번호를 만든다) |
| 버전 관리 반영 대기 | AWS 문서는 버킷에서 버전 관리를 처음 켠 뒤 반영에 최대 15분이 걸릴 수 있다고 한다. 업로드 결과에 객체 버전이 없거나 그 버전을 읽지 못하면 30초 간격으로 다시 올리며, 처음을 포함해 최대 30번 올린다(기다림은 최대 14분 30초) |
| zip 패키지 (컨테이너 이미지 아님) | 이 PC에 Docker가 없다. scikit-learn, scipy, numpy의 리눅스용 휠을 `pip --platform manylinux...`로 받아 zip을 만든다. Lambda는 코드 폴더(/var/task)에 쓸 수 없어 콜드 스타트마다 라이브러리를 다시 컴파일하게 되므로 .pyc를 미리 만들어 넣는다. Python 3.12에서 빌드할 때만 만들고, 소스와 대조하지 않는 방식(unchecked-hash)을 써서 압축 해제 시각과 상관없이 그대로 쓰이게 하며, 소스 경로는 `/var/task/...`로 기록한다. /var/task는 읽기 전용이라 소스가 바뀔 일이 없다. 컴파일한 수와 실패한 파일은 `package.json`에 남는다. 압축 해제 198.94MB로 Lambda 한도 250MB 안이다. zip이 직접 업로드 한도 50MB를 넘어 S3를 거쳐 올린다 |
| python3.12, x86_64 | 로컬 학습 환경과 같은 Python, 라이브러리 버전으로 맞춰 저장한 모델을 그대로 불러온다. python3.12 런타임은 Amazon Linux 2023(glibc 2.34) 기반이라 manylinux_2_28 휠이 동작한다. AWS 런타임 표 기준 지원 종료는 2028-10-31이다 |
| 서울 리전 | 사용자(한국)와 가깝고 필요한 서비스가 모두 있다. 데이터가 공개 자료라 위치 제약이 없다 |
| 메모리 1024MB, 제한 시간 30초 | 메모리를 늘리면 CPU도 비례해서 늘어 scikit-learn을 불러오는 콜드 스타트가 빨라진다. 실제 사용량은 실행 후 REPORT 로그로 확인한다 |
| 예약 동시성 2 | 실수로 반복 호출해도 동시에 2개까지만 돌아 비용과 영향 범위가 제한된다. 새 계정은 동시 실행 할당량이 낮아 예약이 거절될 수 있는데, 그 경우만 기록에 남기고 계속한다. 권한 오류 같은 다른 오류는 실패로 처리한다 |
| 배포마다 함수 버전 게시 | 버전마다 코드 sha256, 환경 변수(모델 객체 버전 포함), 버전 설명(git 커밋, 모델 버전, 패키지 sha)이 고정된다. 커밋하지 않은 코드로 배포하면 설명에 `+dirty`가 붙는다. 검사와 평가는 `$LATEST`가 아니라 같은 실행의 `deploy`가 게시한 번호(따로 실행하면 가장 최근 번호)로 호출해서, 어느 코드와 모델이 답했는지 기록에 남긴다 |

### 설정

| 파일 | 내용 |
|---|---|
| [config/pipeline.json](../config/pipeline.json) | 리전, 함수와 역할 이름, 버킷 접두어, 태그, 분할 기간, 학습 스레드 수, 하이퍼파라미터, 합격 기준. `cloud.bucket`을 적으면 그 버킷을 쓴다(이름이 `bike-demand-mlops-`로 시작하고, 이 과제의 태그가 있고, 지금 리전에 있어야 함. 없으면 그 이름으로 만든다). 이때도 자동으로 만든 버킷은 조회와 정리에 포함된다 |
| [config/lambda.json](../config/lambda.json) | 런타임, 메모리, 제한 시간, 임시 저장소, 예약 동시성, 로그 보관 일수 |
| Lambda 환경 변수 | `MODEL_URI`(S3 경로와 객체 버전), `MODEL_VERSION`, `MODEL_SHA256`, `JOBLIB_MULTIPROCESSING=0` |

자원 구분: 버킷, IAM 역할, Lambda 함수, 로그 그룹은 이 과제의 태그(`project`, `owner`)가 있을 때만 배포, release, 정리에서 바꾸거나 지운다. 예외는 태그도 객체도 없는 빈 버킷 중 두 경우뿐이다. 하나는 이름이 `bike-demand-mlops-<리전>-<16진수 8자리>` 형식인 버킷이고, 다른 하나는 `cloud.bucket`에 적은 이름이면서 실행 기록에 이 도구가 쓴 기록이 있는 버킷이다. 둘 다 이 도구가 버킷을 만든 직후 태그를 붙이기 전에 멈춰 남은 것으로 보고, 배포는 태그를 붙여 이어받고 정리는 지운다. 기록이 없는 설정 버킷은 사용자가 직접 만든 것일 수 있으므로 건드리지 않는다.

`JOBLIB_MULTIPROCESSING=0`을 두는 이유: Lambda에는 /dev/shm이 없어 joblib을 불러올 때 경고가 난다. 예측은 한 프로세스에서 하므로 joblib 병렬 처리를 끈다.

Lambda 실행 역할 `bike-demand-predict-role`은 권한 경계 `bike-demand-predict-boundary`([docs/iam/lambda-boundary.json](iam/lambda-boundary.json))를 달고 만들어진다. 운영자 정책은 이 경계가 붙은 역할만 만들거나 인라인 정책을 바꿀 수 있게 조건을 걸었다. 그래서 운영자가 역할 정책에 넓은 권한을 넣고 Lambda 코드를 바꿔 실행하더라도, 실제 권한은 경계 안(로그 쓰기, 모델 읽기)을 넘지 못한다. 경계 정책은 관리자가 미리 만들고, 루트나 관리자 키로 배포하면 없을 때 도구가 만든다. 운영자 정책에는 정책을 만들거나 지우는 권한이 없다. 그 권한이 있으면 경계를 바꾸거나 없애 권한을 넓힐 수 있기 때문이다. 그래서 운영자 키로 정리하면 도구가 만든 경계 정책도 남고 결과에 `kept`로 기록된다(IAM 정책은 비용이 없다). 이미 있는 역할에 이 경계가 없으면 배포를 멈춘다. 이미 있는 경계 정책의 내용(기본 버전)이 지금 설정의 함수 이름과 버킷 접두어로 만든 문서와 달라도 멈춘다. 이름을 바꾸고 경계 정책을 그대로 두면 Lambda가 모델을 읽지 못하기 때문이다. 역할의 인라인 정책 권한은 두 가지뿐이다([bikeml/cloud.py](../bikeml/cloud.py)의 `role_policy`).
- 자기 로그 그룹에 로그 쓰기 (`logs:CreateLogStream`, `logs:PutLogEvents`)
- 이 과제의 모든 버킷(`bike-demand-mlops-*`)의 `models/*` 읽기 (`s3:GetObject`, `s3:GetObjectVersion`). 지금 버킷만 허용하면, 버킷을 바꿔 배포할 때 아직 이전 버전을 가리키는 `live`가 모델을 못 읽게 되기 때문이다

### 배포 전 확인

`deploy`는 AWS에 접속하기 전에 다음을 확인하고, 하나라도 틀리면 멈춘다.

1. 모델 파일의 sha256이 metadata.json과 같고, 정리한 자료가 설정의 분할 기간과 맞으며, 모델이 지금 학습 자료(`train.csv` sha256), 하이퍼파라미터, 버전, 특성 코드로 학습됐다.
2. 이 모델로 `evaluate`를 통과한 기록(`results/evaluation/metrics.json`)이 있고, 그때의 합격 기준과 평가 자료 sha256이 지금 설정, 지금 파일과 같다.
3. 이 모델로 `check-local`을 통과한 기록(`results/service_check/local.json`)이 있고, 그때의 Lambda 코드 해시와 검사 사례 파일(`samples/service_cases.json`) 해시가 지금과 같다.
4. `build/lambda_package.zip`의 해시가 `results/cloud/package.json`과 같다.
5. zip에 넣은 Lambda 코드와 `requirements-lambda.txt`가 지금 파일과 같고, zip의 런타임과 아키텍처가 Lambda 설정과 같다.
6. 모델을 만들 때의 라이브러리 버전(metadata.json의 scikit-learn, numpy, joblib)이 `requirements-lambda.txt`에 고정한 버전과 같다.

그래서 평가에 떨어진 모델, 검사하지 않은 코드, 이전 코드로 만든 zip, Lambda에서 불러올 수 없는 라이브러리 버전의 모델은 배포되지 않는다. 또 `train`은 후보 모델을 `models/_candidate/`에 만들고 `evaluate`를 통과해야만 `models/hgb-v1/`로 복사하므로, 평가에 떨어진 모델이 이미 합격한 모델을 덮지 않는다. `evaluate`와 `deploy`는 모델의 학습 자료 sha256, 하이퍼파라미터, 버전, 특성 코드가 지금과 같은지, 정리한 자료가 설정의 분할 기간과 맞는지도 확인한다.

## 2. 데이터와 결과의 위치

| 대상 | 위치 |
|---|---|
| 원자료, 정리한 자료 | Git: `data/raw/`, `data/processed/` |
| 모델 | Git: `models/hgb-v1/`, 클라우드: `s3://<버킷>/models/hgb-v1/<모델 sha 12자리>/` |
| 배포 패키지 | 로컬: `build/lambda_package.zip` (Git 제외), 클라우드: `s3://<버킷>/packages/lambda-<sha12>.zip` |
| 요청 예시 | `samples/service_cases.json`, `samples/generated/request_2012-12-03.json` |
| 클라우드 응답과 평가 | Git: `results/service_check/cloud.json`, `results/cloud/cloud_eval.json`, `results/cloud/predictions_cloud.csv`. 클라우드 평가 중 호출이 실패하거나 호출 자체가 예외로 끝나면 그때까지의 호출 기록 `cloud_eval_calls.json`이 실패 폴더에 남는다 |
| 배포 기록 | Git: `results/cloud/package.json`, `results/cloud/deploy.json`. 오류 재현용 배포는 `deploy_override.json`에 따로 남겨 정상 배포 기록을 덮지 않는다 |
| 함수 로그 | CloudWatch Logs `/aws/lambda/bike-demand-predict` (7일). 필요한 부분은 Git 기록으로 옮긴다 |
| 실행 기록 | Git: `runs/run_log.jsonl` |

Lambda는 S3에서 받은 코드의 사본을 따로 보관한다(기본 저장 방식). 그래서 수명 주기 규칙으로 `packages/`의 zip이 지워져도 배포된 함수에는 영향이 없다.

`status`와 `cleanup`이 조회하는 범위는 다음과 같다.
- 리전: 지금 리전, 이 PC에서 같은 계정으로 배포를 시작했던 리전(`.state/regions.json`), 실행 기록(`runs/run_log.jsonl`)에 남은 모든 리전, 배포 기록(`results/cloud/deploy.json`, `deploy_override.json`)의 리전, 이 계정에 있는 `bike-demand-mlops-<리전>-<8자리>` 버킷 이름에 든 리전. 버킷 이름에서 뽑은 리전은 실제 AWS 리전 목록에 있는 것만 쓴다. `.state/`가 없는 새 폴더에서도 버킷 이름과 실행 기록으로 다른 리전을 찾을 수 있다.
- 버킷: 이름 형식이 맞는 버킷, 설정의 `cloud.bucket`, 실행 기록과 배포 기록에 남은 버킷 이름. 배포는 버킷 이름을 정하자마자 그 단계의 기록에 넣어 두므로, 배포가 중간에 실패했거나 설정에서 `cloud.bucket`을 지운 뒤에도 그 버킷을 찾는다. 기록은 단계가 끝날 때 쓰이므로 프로세스가 강제로 죽으면 남지 않고, 이때 자동 이름 버킷은 이름 형식으로 찾는다. 다른 계정의 버킷이라 위치를 조회할 수 없으면 건너뛴다.

`deploy`는 자원을 만들기 전에 계정과 리전을 `.state/regions.json`(Git 제외)에 적는다. 계정 ID는 그대로 적지 않고, 이 PC에서 만든 임의 키(`.state/local_key`, Git 제외)로 HMAC-SHA256을 계산한 값만 적는다.

다음 경우에는 결과 파일을 남긴 뒤 `status`와 `cleanup`을 실패로 기록한다.
- 이 계정의 자원이 있다고 알려진 리전(상태 파일, 버킷 이름, 설정이나 기록에 남은 버킷의 위치로 찾은 리전)에 접속하지 못했다. 이유가 인증 오류여도 마찬가지다.
- 그 밖의 리전에 일시적인 네트워크나 권한 오류로 접속하지 못했다. 켜지 않은 리전처럼 인증 오류(`InvalidClientTokenId` 등)로 접속할 수 없는 리전은 `regions_unreachable`에 남기고 건너뛴다.
- 상태를 확인하지 못한 자원(`exists: null`)이 있다. `status`는 권한 경계 정책도 여기에 넣는다.
- 실행 기록에 읽을 수 없는 줄(UTF-8 JSON 객체가 아니거나, `run_id`, `step`, `status`가 문자열이 아니거나, `cloud`가 객체가 아니거나, 그 안의 리전과 버킷이 문자열도 null도 아닌 줄)이 있다. 그 줄에 적힌 리전이나 버킷을 놓쳤을 수 있기 때문이다. 빈 줄은 건너뛴다. `release`도 이때 검사 결과를 확인할 수 없으므로 멈춘다.
- 지우다가 오류가 난 자원이 있다(`cleanup`). 결과 파일의 `errors`에 남는다.
- 다른 계정의 배포 기록이 남아 있다(`cleanup`). 이 계정의 자원을 지운 뒤 그 계정의 프로필로 다시 실행하라고 알린다.

IAM 역할은 리전과 상관없는 자원이라, 모든 리전의 지역 자원이 지워졌고 위의 첫째, 둘째, 넷째 경우가 아닐 때만 마지막에 지운다. 역할의 인라인 정책은 리전마다 따로 둔다.

공개 저장소에 계정 정보가 남지 않게 했다([bikeml/runlog.py](../bikeml/runlog.py)의 `sanitize`).
- 버킷 이름에는 계정 ID 대신 임의 문자열을 넣는다. 다음 실행부터는 같은 접두어의 버킷을 목록에서 찾아 쓴다.
- 기록에는 호출자의 이름 대신 종류(`root`, `user`, `assumed-role`)만 남기고, 화면에도 종류만 보여 준다.
- 기록과 결과 파일을 저장할 때 다음을 바꾼다. JSON 안에서 이스케이프된 경로도 포함한다.
  - 계정 ID는 `<ACCOUNT_ID>`로 바꾼다.
  - 저장소 경로는 `<REPO>`, 홈 폴더 경로는 `~`로 바꾼다. 저장소와 홈 폴더 밖에 있는 설정 파일을 `--config`나 `--lambda-config`로 지정하면 그 파일이 있는 폴더도 `<EXTERNAL>`로 바꾼다.
  - 경로 밖에서 나온 이 PC 사용자 이름(`Users\이름` 형태)은 `<USER>`로 바꾼다. 경로는 폴더 경계까지 맞을 때만 바꾸므로 이름이 비슷한 다른 폴더는 건드리지 않는다.
  - AWS 오류 메시지 속 IAM 사용자나 세션 이름(`user/이름`, `assumed-role/...`)은 `<NAME>`으로, ARN 속 12자리 계정 ID는 `<ACCOUNT_ID>`로 바꾼다. 계정 ID를 알기 전에 난 오류도 마찬가지다.
  - AWS 오류 문장에서 프로필 이름이 나오는 자리(`profile (이름)`, `profile "이름"`, `profile 이름`, `profiles: [...]`)는 `<PROFILE>`로 바꾼다. 실행 인자의 `profile`도 이름 대신 `<PROFILE>`로 남긴다. 이 자리 밖에 같은 글자가 있어도 바꾸지 않는다.

## 3. 실제 입력에서 결과까지 (실행 결과)

실행 전이다. 실행 순서와 확인할 것은 다음과 같다.

1. `python -m bikeml --profile <프로필> run cloud`: package, deploy, smoke, cloud-eval, release를 순서대로 실행한다.
2. deploy: 배포 전 확인, 권한 경계 확인, 역할·버킷·로그 그룹 생성, 모델과 패키지 업로드, 함수 생성, 버전 게시. 결과는 `results/cloud/deploy.json`
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
| `inputs`, `configs` | 입력 파일과 설정 파일의 경로(저장소 기준 상대 경로. 저장소 밖 파일은 `<EXTERNAL>/파일 이름`), sha256 |
| `args` | 명령 인자(설정 파일, 프로필, 리전, Lambda 설정 파일 등). 설정 파일 경로는 `inputs`와 같은 방식으로, 프로필은 `<PROFILE>`로 남긴다 |
| `cloud` | 리전, 함수, 게시 버전, 별칭, 코드 sha256, 모델과 패키지의 S3 경로 |
| `result_location` | 결과 파일 위치 (실패하면 `results/failed/...`) |
| `summary`, `error`, `env` | 주요 수치, 오류 메시지, Python과 라이브러리 버전 |

- 각 단계는 결과를 임시 폴더에 먼저 쓴다. 성공하면 제자리(`results/`, `models/` 등)로 옮긴다. 옮기다 실패하거나 Ctrl+C로 멈추면 이미 옮긴 파일을 원래대로 되돌린다.
- 단계가 실패하면 결과를 `results/failed/<run_id>_<단계>/`로 옮긴다. 정상 결과를 덮어쓰지 않고, 오류 추적은 그 폴더의 `error.txt`에 남는다. Ctrl+C로 멈춘 경우도 실패로 기록한다.
- 한 단계가 실패하면 뒤 단계는 실행하지 않고 `skipped`로 기록한다.
- 외부 모델은 쓰지 않는다. boto3는 zip에 넣지 않고 Lambda 런타임에 들어 있는 것을 쓰기로 했다(모델 파일을 내려받는 데만 쓰므로 패키지를 키우지 않으려고). 그래서 boto3 버전은 AWS가 정하며, 응답의 `runtime`과 서비스 검사 결과의 `runtime`에 실제 버전이 기록된다.

실패 처리 예시 (AWS 없이 실행함)

합격 기준만 90%로 올린 설정([config/demo_strict_criteria.json](../config/demo_strict_criteria.json))으로 평가를 다시 돌렸다.

```
python -m bikeml --config config/demo_strict_criteria.json run evaluate check-local
```

| 항목 | 결과 |
|---|---|
| run_id | `20261120T035711-1523d6` |
| evaluate | failed. `RuntimeError: 합격 기준 미달: {'beats_all_baselines': True, 'mae_reduction_vs_hour_baseline': 0.499, 'meets_reduction': False}` |
| check-local | skipped (`이전 단계 실패로 실행하지 않음: evaluate`) |
| 실패한 결과 | `results/failed/20261120T035711-1523d6_evaluate/` (metrics.json의 `passed: false`, error.txt) |
| 정상 결과 | `results/evaluation/metrics.json`은 그대로 (`passed: true`, 기준 0.3). 후보 모델은 `models/hgb-v1/`로 옮겨지지 않았다 |
| 배포 | 정상 설정으로는 배포 전 확인을 통과한다. 엄격한 설정으로 `deploy`를 실행하면 "evaluate 이후 합격 기준이나 평가 자료가 바뀜"으로 멈춘다 |

`results/failed/`에는 이전 코드로 같은 예시를 실행한 실패 기록도 남아 있다. 그중 하나(`20261119T235636-890eb9`)는 테스트 파일을 커밋하기 전에 실행해 `code_dirty: true`로 기록되어 있다.

## 5. 오류 재현과 복구

실행 전이다. 과제 예시 중 "잘못된 모델 경로"를 클라우드에서 재현할 계획이다.

1. 재현: `python -m bikeml --profile <프로필> run deploy smoke cloud-eval release --model-uri-override s3://<버킷>/models/hgb-v1/model.pkl` (실제 파일은 `model.joblib`). 이 배포 기록은 `results/cloud/deploy_override.json`에 따로 남는다.
2. 예상 영향: 모든 요청이 `500 MODEL_LOAD_FAILED`, smoke 실패, cloud-eval과 release는 실행하지 않음(skipped). release가 실행되지 않으므로 `live` 별칭은 이전 정상 버전을 그대로 가리킨다.
3. 원인 조사: 응답 메시지와 CloudWatch 로그를 보고, 운영자 권한으로 `aws s3api head-object`를 실행해 비교한다. S3는 `s3:ListBucket` 권한이 없는 요청자에게 없는 객체를 404가 아니라 403(Forbidden)으로 알린다. Lambda 역할에는 이 권한이 없어서 403으로 보일 것으로 예상하며, 이 점도 확인한다.
4. 수정: 경로를 바로잡아 다시 배포하고, 배포 단계에서 `MODEL_URI`가 가리키는 객체가 있는지와 sha256이 맞는지 확인하는 코드를 추가한다.
5. 재검사: 실패했던 같은 명령과 정상 입력(smoke 12건, cloud-eval 2,888건)을 다시 실행해 결과를 남긴다.
