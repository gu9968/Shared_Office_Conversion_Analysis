# Shared Office Conversion Prediction

공유오피스 **3일 체험 고객의 행동 데이터를 분석하고, 결제 전환 가능성을 예측하는 머신러닝 프로젝트**입니다.

체험 기간 동안의 방문 횟수, 체류 시간, 최초 방문 시점 등의 행동 데이터를 기반으로 결제 전환에 영향을 미치는 주요 요인을 분석하고, 머신러닝 모델을 활용해 고객별 결제 가능성을 예측했습니다.

또한 **Streamlit 기반 예측 서비스**와 **n8n 자동화 파이프라인**을 구축하여 데이터 입력부터 예측 결과 저장까지의 과정을 자동화했습니다.

---

## 1. Demo

> 프로젝트 시연 영상

<!-- 영상 업로드 후 아래 링크를 수정하세요. -->

[🎥 Demo Video](YOUR_DEMO_VIDEO_URL)

### 주요 시연 내용

* 고객 행동 데이터 입력
* 머신러닝 기반 결제 가능성 예측
* Streamlit 예측 결과 확인
* n8n을 활용한 데이터 처리 및 예측 자동화

---

## 2. Project Background

### Problem

공유오피스에서는 신규 고객 유치를 위해 **3일 무료 체험 프로그램**을 운영하고 있습니다.

하지만 모든 체험 고객이 실제 결제로 이어지는 것은 아니기 때문에,

> **어떤 행동을 보이는 고객이 실제 결제로 전환될 가능성이 높은가?**

를 데이터 기반으로 파악할 필요가 있었습니다.

### Solution

체험 고객의 방문 행동 데이터를 기반으로 다음 과정을 수행했습니다.

1. 체험 고객 행동 데이터 전처리
2. 결제 전환 여부에 따른 행동 패턴 분석
3. 주요 전환 영향 변수 탐색
4. 머신러닝 기반 결제 가능성 예측
5. Streamlit 기반 예측 서비스 구현
6. n8n 기반 데이터 처리 및 예측 자동화

### Result

분석 결과 결제 전환과 관련된 주요 변수로 다음과 같은 행동 특성이 확인되었습니다.

* 평균 체류 시간
* 최초 방문 시간
* 일평균 출입 횟수
* 방문 일수
* 최초 방문까지 걸린 시간

특히 **방문 횟수와 체류 패턴**이 결제 전환 가능성을 구분하는 주요 행동 변수로 나타났습니다.

이를 기반으로 머신러닝 모델을 구축하고 신규 체험 고객의 행동 데이터를 입력하면 **결제 전환 가능성을 예측할 수 있는 시스템**을 구현했습니다.

---

## 3. Tech Stack

### Data Analysis

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Visualization

* Matplotlib
* Seaborn

### Application

* Streamlit

### Automation

* n8n
* Docker

### Collaboration / Environment

* Git
* GitHub
* VS Code
* Jupyter Notebook

---

## 4. Project Structure

```text
shared-office-conversion-prediction/
│
├── README.md
│
├── requirements.txt
│
├── .gitignore
│
├── app.py
├── prediction_api.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   └── model.pkl
│
├── images/
│   ├── architecture.png
│   ├── pipeline.png
│   └── streamlit_demo.png
│
└── n8n/
    └── workflow.json
```

> 실제 GitHub 업로드 파일 구조에 맞춰 변경합니다.

---

## 5. Data Pipeline

```text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Exploratory Data Analysis
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Prediction
   ↓
Streamlit / n8n
```

### Feature Engineering

고객의 체험 행동을 설명하기 위해 주요 파생 변수를 생성했습니다.

예시:

```text
visit_days
total_stay_time
first_visit_delay
avg_stay_hour
avg_daily_enter
n_sites_visited
first_visit_hour
```

---

## 6. Machine Learning

결제 전환 여부를 예측하기 위해 여러 머신러닝 모델을 비교하고 최종 모델을 선정했습니다.

### Final Model

**XGBoost Classifier**

주요 설정:

```python
XGBClassifier(
    learning_rate=0.02,
    max_depth=2,
    gamma=0.3,
    colsample_bytree=0.7,
    n_estimators=700
)
```

분류 임계값은 비즈니스 목적에 맞춰 조정했습니다.

```text
Threshold = 0.35
```

### Model Performance

대표적인 평가 지표:

```text
ROC-AUC ≈ 0.62 ~ 0.63
```

단순 정확도보다 **결제 가능성이 높은 고객을 구분하는 능력**을 중심으로 모델을 평가했습니다.

---

## 7. Feature Importance

모델 분석 결과 주요 변수는 다음과 같습니다.

| Rank | Feature          | Description |
| ---- | ---------------- | ----------- |
| 1    | avg_stay_hour    | 평균 체류 시간    |
| 2    | first_visit_hour | 최초 방문 시간    |
| 3    | avg_daily_enter  | 일평균 출입 횟수   |

이를 통해 단순한 방문 여부보다 **고객의 구체적인 이용 행동 패턴**이 결제 전환과 연관되어 있음을 확인했습니다.

---

## 8. Streamlit Application

분석 결과를 실제로 활용할 수 있도록 Streamlit 기반 서비스를 구현했습니다.

### Pages

#### 1. Project Overview

프로젝트 목적과 분석 배경을 확인할 수 있습니다.

#### 2. Model Performance

모델의 성능과 주요 분석 결과를 확인할 수 있습니다.

#### 3. Conversion Prediction

신규 고객의 행동 데이터를 입력하면 머신러닝 모델이 결제 전환 가능성을 예측합니다.

```text
Customer Data
      ↓
Preprocessing
      ↓
ML Model
      ↓
Conversion Probability
```

---

## 9. Automation with n8n

반복적인 데이터 처리 및 예측 과정을 자동화하기 위해 **n8n Workflow**를 구성했습니다.

### Workflow

```text
Google Drive
     ↓
New Data Detection
     ↓
Data Preprocessing
     ↓
Prediction API
     ↓
ML Model
     ↓
Prediction Result
     ↓
Result Storage
```

신규 데이터가 입력되면 사람이 직접 분석 코드를 실행하지 않아도 자동으로 예측 프로세스가 실행되도록 구성했습니다.

---

## 10. Architecture

```text
                   ┌──────────────┐
                   │  Input Data  │
                   └──────┬───────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Data Processing │
                 │     Python      │
                 └────────┬────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ XGBoost Model │
                  └───────┬───────┘
                          │
               ┌──────────┴───────────┐
               │                      │
               ▼                      ▼
        ┌─────────────┐        ┌─────────────┐
        │  Streamlit  │        │     n8n     │
        │ Prediction  │        │ Automation  │
        └─────────────┘        └─────────────┘
```

---

## 11. Key Insights

분석을 통해 다음과 같은 비즈니스 인사이트를 도출했습니다.

### ① 고객 행동 패턴과 결제 전환

단순 방문 여부보다 **방문 빈도와 체류 시간의 조합**이 결제 전환과 더 밀접하게 나타났습니다.

### ② 최초 방문 행동

체험 등록 이후 실제 첫 방문까지의 시간과 최초 방문 시간대 역시 결제 전환을 구분하는 주요 변수로 확인되었습니다.

### ③ ML을 활용한 고객 우선순위화

결제 가능성이 높은 고객을 사전에 예측하면 모든 체험 고객에게 동일한 마케팅을 수행하는 대신,

**전환 가능성이 높은 고객에게 상담·프로모션·CRM 자원을 우선 배분하는 방식**으로 활용할 수 있습니다.

---

## 12. Expected Business Impact

본 프로젝트의 예측 모델은 단순히 고객의 결제 여부를 예측하는 데 그치지 않고 CRM 및 영업 전략에 활용할 수 있습니다.

```text
Trial Customer
      ↓
Behavior Data
      ↓
Conversion Prediction
      ↓
Customer Segmentation
      ↓
CRM / Sales Action
```

예를 들어,

* 고전환 예상 고객 → 상담 및 계약 유도
* 중간 전환 예상 고객 → 추가 혜택 제공
* 저전환 예상 고객 → 행동 패턴 기반 리텐션 전략

과 같은 방식으로 고객 대응 전략을 차별화할 수 있습니다.

---

## 13. What I Learned

이 프로젝트를 통해 다음 과정을 경험했습니다.

* 실제 비즈니스 데이터를 활용한 문제 정의
* EDA를 통한 고객 행동 패턴 분석
* Feature Engineering
* Classification Model 구축 및 평가
* Threshold 조정을 통한 모델 활용 방식 설계
* Streamlit 기반 ML 서비스 구현
* n8n 기반 데이터 자동화 Workflow 설계
* 분석 결과를 실제 비즈니스 액션으로 연결하는 과정

---

## 14. Future Improvements

향후 다음과 같은 방향으로 프로젝트를 개선할 수 있습니다.

* 추가 고객 행동 데이터 확보
* 모델 성능 개선
* SHAP을 활용한 개별 고객 예측 근거 제공
* CRM 시스템 연계
* 자동 재학습 Pipeline 구축
* 실시간 예측 API 구축

---

## Author

Data Analysis & Machine Learning Project
