# 데이터 출처와 이용 조건

## 원자료 (`raw/`)

- 이름: Bike Sharing
- 제공: UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- 인용: Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894
- 논문: Fanaee-T, H., Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, 2(2-3), 113-127. https://doi.org/10.1007/s13748-013-0040-3
- 이용 조건: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- 원 출처: Capital Bikeshare(워싱턴 D.C.) 2011~2012년 대여 기록, 날씨는 freemeteo.com

`hour.csv`와 `Readme.txt`는 받은 그대로이며 바꾸지 않았다. sha256은 [config/pipeline.json](../config/pipeline.json)에 있다.

## 가공한 자료 (`processed/`)

`python -m bikeml run prepare`가 원자료로 만든 파생 자료다. 원자료에서 바꾼 점은 다음과 같다.

- dteday와 hr을 합쳐 `datetime`(`YYYY-MM-DDTHH:00`) 한 열로 만듦
- weathersit을 `weather`로, holiday를 true/false로 바꿈
- 기온, 체감 기온, 습도, 풍속을 정규화 값에서 실제 단위(°C, %, km/h)로 되돌려 `temp_c`, `feels_like_c`, `humidity`, `windspeed_kmh`로 둠
- 습도 0% 22행(2011-03-10)을 결측으로 바꿈
- instant, season, yr, mnth, weekday, workingday, casual, registered 열을 뺌. yr, mnth, weekday, workingday는 `datetime`과 `holiday`에서 다시 계산할 수 있다
- 2012-08-31까지를 `train.csv`, 2012-09-01부터를 `test.csv`로 나눔

자세한 기준은 [docs/data_and_model.md](../docs/data_and_model.md)에 있다. 가공한 자료도 CC BY 4.0 조건을 따른다.
