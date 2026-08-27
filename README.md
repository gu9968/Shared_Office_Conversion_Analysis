# 🏢 공유오피스 체험 고객 결제 전환 예측

공유오피스 **3일 체험 고객의 이용 행동을 분석하고 결제 전환 가능성을 예측**하는 머신러닝 프로젝트입니다.

방문·체류·출입 행동을 기반으로 전환에 영향을 미치는 요인을 분석하고,
**XGBoost → FastAPI → Streamlit → n8n**으로 이어지는 예측·자동화 환경을 구현했습니다.

---

## 🎬 Demo

> 실제 체험 고객 데이터 입력 → 결제 가능성 예측 → n8n 자동 처리 과정

**[시연 영상 보기](GITHUB_VIDEO_URL)**

---

## 🔎 Project Overview

| 구분             | 내용                                    |
| -------------- | ------------------------------------- |
| **Goal**       | 3일 체험 고객 중 실제 결제 가능성이 높은 고객 식별        |
| **Data**       | 방문일수, 체류시간, 출입횟수, 최초 방문시간 등 체험 행동 데이터 |
| **Model**      | XGBoost Classifier                    |
| **Service**    | Streamlit 기반 고객별 결제 가능성 예측            |
| **Automation** | Google Drive → FastAPI → 예측 결과 저장 자동화 |

---

## 📊 Analysis

체험 고객의 이용 패턴을 기준으로 결제 전환 차이를 분석했습니다.

주요 분석 변수:

* `avg_stay_hour` : 평균 체류 시간
* `first_visit_hour` : 최초 방문 시간
* `avg_daily_enter` : 일평균 출입 횟수
* `visit_days` : 방문 일수
* `first_visit_delay` : 체험 등록 후 첫 방문까지 소요 시간

### 주요 발견

* **방문 빈도가 높고 체류 시간이 짧은 고객군의 결제율이 가장 높게 나타남**
* 최초 방문 시점과 일평균 출입 횟수에서도 전환 고객과 비전환 고객 간 차이가 확인됨
* 단순 방문 여부보다 **체험 기간 동안의 이용 행동 패턴**이 결제 전환 예측에 중요하게 작용함

---

## 🤖 Prediction Model

최종 모델은 **XGBoost Classifier**를 사용했습니다.

| 항목                   |                           결과 |
| -------------------- | ---------------------------: |
| ROC-AUC              |            약 **0.62 ~ 0.63** |
| Prediction Threshold |                     **0.35** |
| 주요 변수                | 평균 체류시간 · 최초 방문시간 · 일평균 출입횟수 |

모델 결과는 단순 결제 여부 판정보다
**전환 가능성이 높은 체험 고객을 사전에 선별하는 용도**에 초점을 맞췄습니다.

---

## 🖥️ Prediction Service

Streamlit을 통해 모델 결과를 바로 확인할 수 있도록 구현했습니다.

```text
고객 행동 데이터 입력
        ↓
     FastAPI
        ↓
   XGBoost Model
        ↓
 결제 전환 확률 계산
        ↓
 Streamlit 결과 표시
```

---

## ⚙️ n8n Automation

![n8n 자동화 워크플로우](./assets/n8n_workflow.png)

### 자동화 흐름

```text
Google Drive Trigger
        ↓
파일 다운로드
        ↓
CSV 데이터 추출
        ↓
데이터 집계
        ↓
FastAPI HTTP 요청
        ↓
예측 결과 분리
        ↓
CSV 파일 생성
        ↓
Google Drive 업로드
```

Google Drive에 새로운 체험 고객 CSV가 등록되면 n8n이 파일을 감지합니다.

이후 고객 데이터를 FastAPI 예측 API로 전달하고,
반환된 **결제 확률 및 예측 결과를 CSV로 생성하여 Google Drive에 자동 저장**하도록 구성했습니다.

---

## 🧩 System Flow

```text
                  ┌─────────────┐
                  │ Trial Data  │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │    n8n      │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   FastAPI   │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   XGBoost   │
                  └──────┬──────┘
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
         ┌───────────┐       ┌───────────┐
         │ Streamlit │       │ CSV Result│
         └───────────┘       └───────────┘
```

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

## 📁 Repository

```text
shared-office-conversion-prediction/
│
├── app.py
├── prediction_api.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── models/
│   └── model.pkl
│
├── assets/
│   └── n8n_workflow.png
│
├── requirements.txt
└── README.md
```
