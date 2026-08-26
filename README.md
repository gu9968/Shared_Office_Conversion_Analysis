# 🏢 공유오피스 결제 전환 예측

## 시연 영상

[![시연영상](VIDEO_THUMBNAIL_URL)](VIDEO_URL)

---

## 🎯 프로젝트 배경

### Why?

| 문제                         | 해결                     | 결과                      |
| -------------------------- | ---------------------- | ----------------------- |
| 3일 체험 고객의 실제 결제 가능성 판단 어려움 | 고객 방문·체류 행동 데이터 분석     | 결제 전환 주요 행동 패턴 도출       |
| 모든 고객에게 동일한 영업·마케팅 진행      | 머신러닝 기반 결제 가능성 예측      | 고객별 전환 가능성 사전 분류        |
| 예측 과정의 반복적인 수작업            | Streamlit + n8n 자동화 구축 | 데이터 입력부터 예측 결과 저장까지 자동화 |

---

## 기술 스택

![Python](https://img.shields.io/badge/Python-3776AB?style=flat\&logo=python\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat\&logo=pandas\&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat\&logo=scikitlearn\&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat\&logo=streamlit\&logoColor=white)
![n8n](https://img.shields.io/badge/n8n-EA4B71?style=flat\&logo=n8n\&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat\&logo=docker\&logoColor=white)

---

## 프로젝트 구조

```text
shared-office-conversion-prediction/
├── app.py                    # Streamlit 예측 서비스
├── prediction_api.py         # 예측 API
├── notebooks/
│   ├── 01_eda.ipynb          # 탐색적 데이터 분석
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb     # 모델 학습 및 평가
├── models/
│   └── model.pkl
├── n8n/
│   └── workflow.json
├── requirements.txt
└── README.md
```

---

## 파이프라인 구조

| 단계          | 내용                                     | 결과             |
| ----------- | -------------------------------------- | -------------- |
| 📊 **분석**   | 체험 고객 방문·체류 데이터 EDA                    | 결제 전환 행동 패턴 도출 |
| 🔧 **전처리**  | 결측치 처리 · 파생변수 생성 · Feature Engineering | 모델 입력 데이터 생성   |
| 🤖 **예측**   | XGBoost 기반 결제 전환 분류                    | 고객별 결제 가능성 예측  |
| 🖥️ **서비스** | Streamlit 예측 화면 구현                     | 신규 고객 실시간 예측   |
| ⚙️ **자동화**  | n8n 데이터 감지 → API 호출 → 결과 저장            | 예측 프로세스 자동화    |

---

## 🏗️ 아키텍처

<!-- GitHub에 이미지 업로드 후 아래 URL 교체 -->

<img src="ARCHITECTURE_IMAGE_URL" />

체험 고객의 행동 데이터를 전처리한 뒤 머신러닝 모델로 결제 가능성을 예측하고,
Streamlit과 n8n을 통해 예측 결과를 활용할 수 있도록 구성했습니다.

---

## n8n 워크플로우

<!-- GitHub에 이미지 업로드 후 아래 URL 교체 -->

<img src="N8N_WORKFLOW_IMAGE_URL" />

1. Google Drive에서 신규 데이터 파일을 감지합니다.
2. 데이터를 전처리합니다.
3. Prediction API로 데이터를 전달합니다.
4. 머신러닝 모델이 결제 전환 가능성을 예측합니다.
5. 예측 결과를 자동 저장합니다.

---

## 📈 주요 분석 결과

* **평균 체류 시간(`avg_stay_hour`)**이 주요 예측 변수로 나타남
* **최초 방문 시간(`first_visit_hour`)**이 결제 전환과 연관됨
* **일평균 출입 횟수(`avg_daily_enter`)**가 주요 변수로 확인됨
* 단순 방문 여부보다 **방문 빈도와 체류 패턴의 조합**이 결제 전환 판단에 중요하게 나타남
* 최종 모델 ROC-AUC 약 **0.62~0.63**
* 비즈니스 활용을 고려하여 예측 Threshold를 **0.35**로 조정

---

## 주요 트러블슈팅

| 문제                         | 해결                          |
| -------------------------- | --------------------------- |
| 고객별 방문 기록 형태가 달라 직접 비교 어려움 | 방문일수·체류시간·출입횟수 등 파생변수 생성    |
| 기본 Threshold에서 타깃 고객 탐지 한계 | Threshold를 0.35로 조정         |
| 분석 코드 수동 실행 필요             | Prediction API + n8n 자동화 구축 |
| 분석 결과를 비개발자가 활용하기 어려움      | Streamlit 기반 예측 서비스 구현      |

---

## 한계점 및 개선 방향

| 한계점                  | 개선 방향                                  |
| -------------------- | -------------------------------------- |
| ROC-AUC 0.62~0.63 수준 | 추가 행동 데이터 확보 및 Feature Engineering 고도화 |
| 고객 행동 데이터 중심 예측      | 고객 속성·마케팅 접점 데이터 추가                    |
| 로컬 Docker 기반 n8n 운영  | Cloud 환경으로 이전                          |
| 모델 결과 중심 설명          | SHAP 기반 개별 고객 예측 근거 제공                 |
