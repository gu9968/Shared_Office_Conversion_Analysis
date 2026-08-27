# Data

원본 데이터는 사용자 이용 로그 및 식별 정보를 포함하고 있어 GitHub 저장소에 공개하지 않습니다.

## 로컬 재현 시 필요한 원본 파일

아래 파일을 `data/raw/`에 직접 배치해야 합니다.

```text
data/raw/
├── site_area.csv
├── trial_register.csv
├── trial_visit_info.csv
├── trial_access_log.csv
└── trial_payment.csv
```

## 공개 데이터

`sample/prediction_input_sample.csv`은 서비스 실행 형식을 보여주기 위한 **익명화된 시연용 샘플 데이터**입니다.
실제 원본 고객 식별자는 포함하지 않습니다.

## Git 제외 권장 경로

```gitignore
data/raw/
data/interim/
data/processed/
```
