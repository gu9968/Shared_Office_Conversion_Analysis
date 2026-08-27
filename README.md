# 🏢 공유오피스 3일 체험 고객 결제 전환 예측

공유오피스 **3일 무료체험 고객의 이용 행동을 분석하고 결제 전환 가능성을 예측하는 머신러닝 프로젝트**입니다.

방문·체류·출입 행동을 기반으로 전환 요인을 분석하고,
**XGBoost → FastAPI → Streamlit → n8n**으로 이어지는 예측 및 자동화 환경을 구현했습니다.

---

## 🎬 Demo

[![시연영상](https://img.youtube.com/vi/_-W1URhU9to/0.jpg)](https://youtu.be/_-W1URhU9to)

> 고객 데이터 입력 → 결제 가능성 예측 → n8n 자동 처리 → 예측 결과 저장

---

## 🔎 Project Overview

| 구분         | 내용                            |
| ---------- | ----------------------------- |
| **목표**     | 무료체험 고객 중 결제 전환 가능성이 높은 고객 식별 |
| **분석 대상**  | 공유오피스 3일 무료체험 고객              |
| **주요 데이터** | 방문일수 · 체류시간 · 출입횟수 · 최초 방문시점  |
| **최종 모델**  | XGBoost Classifier            |
| **모델링 표본** | 5,629명                        |
| **서비스**    | Streamlit + FastAPI           |
| **자동화**    | n8n + Google Drive            |

---

## 📚 Analysis Process

프로젝트 분석 과정은 단계별 Notebook으로 구성했습니다.

| Notebook                      | 내용                                       |
| ----------------------------- | ---------------------------------------- |
| `00_project_overview.ipynb`   | 프로젝트 배경 및 문제 정의                          |
| `01_data_preprocessing.ipynb` | 원본 데이터 정제 및 사용자 단위 테이블 생성                |
| `02_eda_analysis.ipynb`       | 고객 행동 변수 생성 및 결제 전환 패턴 분석                |
| `03_model_experiment.ipynb`   | Logistic · Random Forest · XGBoost 모델 비교 |
| `04_model_tuning_final.ipynb` | Feature Engineering · XGBoost 튜닝 · 모델 저장 |
| `05_streamlit_prepare.ipynb`  | 서비스용 모델 검증 및 예측 샘플 생성                    |

```text
Raw Data
   ↓
Preprocessing
   ↓
EDA & Feature Engineering
   ↓
Model Experiment
   ↓
XGBoost Tuning
   ↓
Streamlit / FastAPI
   ↓
n8n Automation
```

---

## 📊 Key Findings

체험 고객의 이용 행동을 분석한 결과 다음 변수가 결제 전환 예측에 주요하게 활용되었습니다.

* `avg_stay_hour` — 평균 체류 시간
* `first_visit_hour` — 최초 방문 시간
* `avg_daily_enter` — 일평균 입실 횟수
* `visit_days` — 체험기간 방문 일수
* `first_visit_delay` — 신청 후 첫 방문까지 소요 시간

특히 단순 방문 여부보다 **방문 빈도와 체류 패턴의 조합**에서 결제 전환 차이가 확인되었습니다.

---

## 🤖 Prediction Model

기본 행동 변수와 상호작용·비율·시간대·로그 변수를 생성하여
최종적으로 **42개 Feature**를 모델에 사용했습니다.

### Final Model

**XGBoost Classifier**

```python
XGBClassifier(
    n_estimators=700,
    learning_rate=0.02,
    max_depth=2,
    gamma=0.3,
    colsample_bytree=0.7,
    scale_pos_weight=2.0
)
```

| 항목                   |                결과 |
| -------------------- | ----------------: |
| ROC-AUC              | 약 **0.62 ~ 0.63** |
| Feature              |           **42개** |
| Prediction Threshold |          **0.35** |

운영 목적상 단순 Accuracy보다 **결제 가능 고객을 탐지하는 Recall**을 중요하게 보고 Threshold를 조정했습니다.

---

## 🖥️ Prediction Service

Streamlit에서 고객 행동 데이터를 입력하면 FastAPI를 통해 모델에 전달되고,
고객별 **결제 전환 확률과 예측 결과**를 확인할 수 있도록 구현했습니다.

```text
Customer Data
      ↓
  Streamlit
      ↓
   FastAPI
      ↓
   XGBoost
      ↓
Payment Probability
```

---

## ⚙️ n8n Automation

![n8n workflow](./assets/n8n_workflow.png)

Google Drive에 신규 고객 데이터가 업로드되면 예측 프로세스가 자동으로 실행됩니다.

```text
Google Drive Trigger
        ↓
파일 다운로드
        ↓
CSV 데이터 추출
        ↓
데이터 집계
        ↓
FastAPI 예측 요청
        ↓
예측 결과 분리
        ↓
CSV 파일 생성
        ↓
Google Drive 저장
```

이를 통해 모델을 직접 실행하지 않아도
**파일 업로드만으로 고객별 결제 전환 예측 결과를 생성**할 수 있도록 구성했습니다.

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat\&logo=python\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat\&logo=pandas\&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat\&logo=scikitlearn\&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat\&logo=fastapi\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat\&logo=streamlit\&logoColor=white)
![n8n](https://img.shields.io/badge/n8n-EA4B71?style=flat\&logo=n8n\&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat\&logo=docker\&logoColor=white)

---

## 📁 Repository Structure

```text
shared-office-conversion-prediction/
│
├── notebooks/
│   ├── 00_project_overview.ipynb
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_model_experiment.ipynb
│   ├── 04_model_tuning_final.ipynb
│   └── 05_streamlit_prepare.ipynb
│
├── app/
│   ├── app.py
│   └── prediction_api.py
│
├── models/
│   ├── best_tuned_model.pkl
│   ├── best_tuned_threshold.pkl
│   ├── feature_cols.json
│   ├── feature_medians.json
│   └── model_metrics.json
│
├── assets/
│   └── n8n_workflow.png
│
├── data/
│   └── README.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔒 Data

분석에 사용한 원본 데이터는 사용자 이용 로그 및 식별 정보를 포함하고 있어
**GitHub 저장소에는 공개하지 않습니다.**

로컬에서 Notebook을 재실행하려면 아래 파일을 `data/raw/`에 별도로 배치해야 합니다.

```text
data/raw/
├── site_area.csv
├── trial_register.csv
├── trial_visit_info.csv
├── trial_access_log.csv
└── trial_payment.csv
```

`data/raw/`, `data/interim/`, `data/processed/`는 `.gitignore`를 통해 Git 추적 대상에서 제외합니다.
