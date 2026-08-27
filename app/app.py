from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


st.set_page_config(
    page_title="공유오피스 결제 전환 예측",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 1. 프로젝트 경로
# ============================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "best_tuned_model.pkl"
THRESHOLD_PATH = MODEL_DIR / "best_tuned_threshold.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"
FEATURE_MEDIANS_PATH = MODEL_DIR / "feature_medians.json"

API_BASE_URL = os.getenv(
    "PREDICTION_API_URL",
    "http://localhost:8000",
).rstrip("/")

LATEST_RESULT_URL = f"{API_BASE_URL}/latest"

DEFAULT_THRESHOLD = 0.35
EPS = 1e-6


# ============================================================
# 2. 모델 자산
# ============================================================
@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다.\n확인 경로: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    threshold = (
        float(joblib.load(THRESHOLD_PATH))
        if THRESHOLD_PATH.exists()
        else DEFAULT_THRESHOLD
    )

    model_feature_cols = model.get_booster().feature_names
    if not model_feature_cols:
        raise ValueError("모델 내부 피처명을 불러오지 못했습니다.")
    model_feature_cols = [str(c) for c in model_feature_cols]

    if FEATURE_COLS_PATH.exists():
        with FEATURE_COLS_PATH.open("r", encoding="utf-8") as f:
            feature_cols = [str(c) for c in json.load(f)]
        if feature_cols != model_feature_cols:
            raise ValueError(
                "feature_cols.json과 모델 내부 피처 순서가 일치하지 않습니다."
            )
    else:
        feature_cols = model_feature_cols

    feature_medians = {}
    if FEATURE_MEDIANS_PATH.exists():
        with FEATURE_MEDIANS_PATH.open("r", encoding="utf-8") as f:
            feature_medians = {
                str(k): float(v)
                for k, v in json.load(f).items()
            }

    return model, threshold, feature_cols, feature_medians


try:
    model, THRESHOLD, feature_cols, feature_medians = load_assets()
except Exception as error:
    st.error("모델 자산을 불러오지 못했습니다.")
    st.code(str(error))
    st.stop()


BASE_FEATURES = [
    "avg_stay_hour",
    "avg_daily_enter",
    "visit_days",
    "first_visit_delay",
    "consecutive_group_2일",
    "consecutive_group_3일",
    "first_visit_hour",
    "n_sites_visited",
    "area_pyeong",
    "is_post_covid",
]

FEATURE_KR = {
    "customer_id": "고객 ID",
    "trial_date": "무료체험 신청일",
    "avg_stay_hour": "평균 체류시간",
    "avg_daily_enter": "하루 평균 입실횟수",
    "visit_days": "방문일수",
    "first_visit_delay": "첫 방문 지연일",
    "consecutive_group_2일": "2일 연속 방문 여부",
    "consecutive_group_3일": "3일 연속 방문 여부",
    "first_visit_hour": "첫 방문 시각",
    "n_sites_visited": "방문 지점 수",
    "area_pyeong": "지점 면적",
    "is_post_covid": "엔데믹 이후 여부",
    "payment_probability": "결제확률",
    "payment_probability_percent": "결제확률(%)",
    "prediction_value": "예측값",
    "prediction": "예측 결과",
}

COLUMN_ALIASES = {
    "고객 ID": "customer_id",
    "체험 날짜": "trial_date",
    "평균 체류 시간": "avg_stay_hour",
    "평균 일일 입실": "avg_daily_enter",
    "방문일수": "visit_days",
    "첫 방문 지연": "first_visit_delay",
    "연속_그룹_2일": "consecutive_group_2일",
    "연속_그룹_3일": "consecutive_group_3일",
    "첫 방문 시간": "first_visit_hour",
    "방문한 사이트 수": "n_sites_visited",
    "면적(평)": "area_pyeong",
    "코로나 이후": "is_post_covid",
}


# ============================================================
# 3. 피처 생성 / 예측
# ============================================================
def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            c: COLUMN_ALIASES.get(str(c).strip(), str(c).strip())
            for c in df.columns
        }
    )


def make_features(input_df: pd.DataFrame):
    df = normalize_input_columns(input_df.copy())

    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        return None, missing

    for c in BASE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "trial_date" in df.columns:
        trial_date = pd.to_datetime(df["trial_date"], errors="coerce")
        df["trial_month"] = trial_date.dt.month.fillna(1).astype(int)
        df["trial_dayofweek"] = trial_date.dt.dayofweek.fillna(0).astype(int)
    else:
        df["trial_month"] = 1
        df["trial_dayofweek"] = 0

    df["is_weekend_trial"] = (df["trial_dayofweek"] >= 5).astype(int)

    df["stay_x_visit"] = df["avg_stay_hour"] * df["visit_days"]
    df["enter_x_visit"] = df["avg_daily_enter"] * df["visit_days"]
    df["stay_x_enter"] = df["avg_stay_hour"] * df["avg_daily_enter"]
    df["enter_per_visit_day"] = df["avg_daily_enter"] / (df["visit_days"] + EPS)
    df["stay_per_enter"] = df["avg_stay_hour"] / (df["avg_daily_enter"] + EPS)
    df["delay_per_visit_day"] = df["first_visit_delay"] / (df["visit_days"] + EPS)
    df["visit_delay_interaction"] = df["visit_days"] * df["first_visit_delay"]

    df["is_fast_visit"] = (df["first_visit_delay"] <= 1).astype(int)
    df["is_delayed_visit"] = (df["first_visit_delay"] >= 3).astype(int)
    df["is_frequent_user"] = (df["avg_daily_enter"] >= 2).astype(int)
    df["is_long_stay"] = (df["avg_stay_hour"] >= 2).astype(int)
    df["is_short_frequent"] = (
        (df["avg_stay_hour"] < 1)
        & (df["avg_daily_enter"] >= 2)
    ).astype(int)
    df["is_long_frequent"] = (
        (df["avg_stay_hour"] >= 2)
        & (df["avg_daily_enter"] >= 2)
    ).astype(int)

    # 최종 학습 노트북과 동일
    df["is_morning"] = (
        (df["first_visit_hour"] >= 6)
        & (df["first_visit_hour"] < 12)
    ).astype(int)
    df["is_afternoon"] = (
        (df["first_visit_hour"] >= 12)
        & (df["first_visit_hour"] < 18)
    ).astype(int)
    df["is_evening"] = (
        (df["first_visit_hour"] >= 18)
        & (df["first_visit_hour"] < 24)
    ).astype(int)

    df["first_visit_hour_sin"] = np.sin(
        2 * np.pi * df["first_visit_hour"] / 24
    )
    df["first_visit_hour_cos"] = np.cos(
        2 * np.pi * df["first_visit_hour"] / 24
    )
    df["trial_month_sin"] = np.sin(
        2 * np.pi * df["trial_month"] / 12
    )
    df["trial_month_cos"] = np.cos(
        2 * np.pi * df["trial_month"] / 12
    )

    df["post_covid_x_visit_days"] = (
        df["is_post_covid"] * df["visit_days"]
    )
    df["post_covid_x_avg_stay"] = (
        df["is_post_covid"] * df["avg_stay_hour"]
    )
    df["post_covid_x_daily_enter"] = (
        df["is_post_covid"] * df["avg_daily_enter"]
    )

    for c in [
        "avg_stay_hour",
        "avg_daily_enter",
        "first_visit_delay",
        "area_pyeong",
        "n_sites_visited",
        "stay_x_visit",
    ]:
        df[f"log1p_{c}"] = np.log1p(df[c].clip(lower=0))

    X = df.reindex(columns=feature_cols).copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    for c in feature_cols:
        if c in feature_medians:
            X[c] = X[c].fillna(feature_medians[c])

    return X, []


def predict_payment(input_df: pd.DataFrame):
    normalized = normalize_input_columns(input_df.copy())
    X, missing = make_features(normalized)

    if missing:
        return None, missing

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    result = normalized.copy()
    result["payment_probability"] = probabilities
    result["payment_probability_percent"] = (probabilities * 100).round(1)
    result["prediction_value"] = predictions
    result["prediction"] = np.where(
        predictions == 1,
        "결제 예상",
        "미결제 예상",
    )

    return result, []


# ============================================================
# 4. UI
# ============================================================
st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 2.2rem;}
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg,#0F3D67 0%,#2563EB 100%);
        margin-bottom: 1.6rem;
    }
    .hero h1 {color:white!important; margin:0 0 .6rem 0;}
    .hero p {color:#DBEAFE!important; margin:0; line-height:1.7;}
    .info-box {
        border:1px solid #D9E2EC;
        border-radius:14px;
        padding:1.1rem 1.2rem;
        background:#FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## 공유오피스 분석")
st.sidebar.caption("무료체험 결제 전환 예측")

menu = st.sidebar.radio(
    "메뉴",
    [
        "프로젝트 소개",
        "모델 성능",
        "결제 전환 예측",
        "자동 예측 결과",
    ],
    label_visibility="collapsed",
)


if menu == "프로젝트 소개":
    st.markdown(
        """
        <div class="hero">
            <h1>공유오피스 3일 체험 고객 결제 전환 예측</h1>
            <p>
            무료체험 고객의 방문·체류·입실 행동을 분석하고
            XGBoost 모델을 활용해 고객별 결제 가능성을 예측합니다.
            예측 결과는 전환 가능 고객 선별과 영업 우선순위 설정에 활용합니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 신청자", "9,624명")
    c2.metric("최종 분석 표본", "5,629명")
    c3.metric("최종 모델", "XGBoost")
    c4.metric("모델 피처", f"{len(feature_cols)}개")

    st.subheader("분석 흐름")
    st.write(
        "데이터 정제 → 고객 행동 분석 → 피처 엔지니어링 → "
        "모델 비교·튜닝 → Streamlit/FastAPI → n8n 자동화"
    )

    st.subheader("주요 분석 변수")
    st.dataframe(
        pd.DataFrame(
            {
                "변수": [
                    "avg_stay_hour",
                    "first_visit_hour",
                    "avg_daily_enter",
                    "visit_days",
                    "first_visit_delay",
                ],
                "설명": [
                    "평균 체류시간",
                    "최초 방문 시각",
                    "하루 평균 입실횟수",
                    "체험기간 방문일수",
                    "신청 후 첫 방문까지 소요일",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


elif menu == "모델 성능":
    st.title("모델 성능")

    metrics = [
        ("Accuracy", "0.44"),
        ("Precision", "0.40"),
        ("Recall", "0.94"),
        ("F1-score", "0.56"),
        ("ROC-AUC", "0.63"),
    ]

    cols = st.columns(5)
    for col, (name, value) in zip(cols, metrics):
        col.metric(name, value)

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("최종 모델", "XGBoost")
    c2.metric("Threshold", f"{THRESHOLD:.2f}")
    c3.metric("입력 피처", f"{len(feature_cols)}개")

    st.info(
        "운영상 결제 가능 고객을 놓치지 않는 방향을 우선해 "
        "Threshold 0.35를 적용했습니다."
    )

    with st.expander("42개 모델 피처 보기"):
        st.dataframe(
            pd.DataFrame(
                {
                    "번호": range(1, len(feature_cols) + 1),
                    "피처": feature_cols,
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


elif menu == "결제 전환 예측":
    st.title("결제 전환 예측")

    manual_tab, csv_tab = st.tabs(
        ["고객 1명 입력", "CSV 일괄 예측"]
    )

    with manual_tab:
        with st.form("manual_prediction"):
            c1, c2 = st.columns(2)

            with c1:
                trial_date = st.date_input("무료체험 신청일")
                visit_days = st.slider("방문일수", 1, 3, 2)
                avg_stay_hour = st.number_input(
                    "평균 체류시간(시간)",
                    min_value=0.0,
                    value=1.5,
                    step=0.1,
                )
                avg_daily_enter = st.number_input(
                    "하루 평균 입실횟수",
                    min_value=0.0,
                    value=2.0,
                    step=0.1,
                )
                first_visit_delay = st.slider(
                    "첫 방문 지연일",
                    0,
                    3,
                    0,
                )

            with c2:
                first_visit_hour = st.slider(
                    "첫 방문 시각",
                    0,
                    23,
                    9,
                )
                n_sites_visited = st.number_input(
                    "방문 지점 수",
                    min_value=1,
                    value=1,
                    step=1,
                )
                area_pyeong = st.number_input(
                    "주 이용 지점 면적(평)",
                    min_value=1.0,
                    value=100.0,
                    step=10.0,
                )
                consecutive_days = st.selectbox(
                    "최대 연속 방문",
                    ["연속 없음", "2일 연속", "3일 연속"],
                )
                is_post_covid = st.selectbox(
                    "체험 시기",
                    [1, 0],
                    format_func=lambda x: (
                        "엔데믹 이후" if x == 1 else "코로나 기간"
                    ),
                )

            submitted = st.form_submit_button(
                "결제 가능성 예측",
                use_container_width=True,
            )

        if submitted:
            input_df = pd.DataFrame(
                [
                    {
                        "trial_date": str(trial_date),
                        "avg_stay_hour": avg_stay_hour,
                        "avg_daily_enter": avg_daily_enter,
                        "visit_days": visit_days,
                        "first_visit_delay": first_visit_delay,
                        "consecutive_group_2일": int(
                            consecutive_days in ["2일 연속", "3일 연속"]
                        ),
                        "consecutive_group_3일": int(
                            consecutive_days == "3일 연속"
                        ),
                        "first_visit_hour": first_visit_hour,
                        "n_sites_visited": n_sites_visited,
                        "area_pyeong": area_pyeong,
                        "is_post_covid": is_post_covid,
                    }
                ]
            )

            result, missing = predict_payment(input_df)

            if missing:
                st.error("누락 컬럼: " + ", ".join(missing))
            else:
                probability = float(
                    result.loc[0, "payment_probability"]
                )
                prediction = result.loc[0, "prediction"]

                c1, c2, c3 = st.columns(3)
                c1.metric("결제 확률", f"{probability:.1%}")
                c2.metric("분류 기준", f"{THRESHOLD:.2f}")
                c3.metric("예측 결과", prediction)

                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

    with csv_tab:
        uploaded = st.file_uploader(
            "고객 CSV 파일",
            type=["csv"],
        )

        if uploaded is not None:
            try:
                input_df = pd.read_csv(uploaded)
                result, missing = predict_payment(input_df)

                if missing:
                    st.error(
                        "예측에 필요한 컬럼이 부족합니다: "
                        + ", ".join(missing)
                    )
                else:
                    pay_count = int(
                        (result["prediction_value"] == 1).sum()
                    )
                    nonpay_count = int(
                        (result["prediction_value"] == 0).sum()
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("전체 고객", f"{len(result):,}명")
                    c2.metric("결제 예상", f"{pay_count:,}명")
                    c3.metric("미결제 예상", f"{nonpay_count:,}명")

                    fig = px.histogram(
                        result,
                        x="payment_probability",
                        nbins=20,
                        title="결제 확률 분포",
                    )
                    fig.add_vline(
                        x=THRESHOLD,
                        line_dash="dash",
                        annotation_text=f"Threshold {THRESHOLD:.2f}",
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                    st.dataframe(
                        result,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "예측 결과 CSV 다운로드",
                        data=result.to_csv(
                            index=False,
                            encoding="utf-8-sig",
                        ),
                        file_name="prediction_result.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            except Exception as error:
                st.error("CSV 예측 중 오류가 발생했습니다.")
                st.code(str(error))


elif menu == "자동 예측 결과":
    st.title("n8n 자동 예측 결과")
    st.caption(
        "n8n이 FastAPI /predict를 호출하면 가장 최근 결과를 조회합니다."
    )

    if st.button(
        "최신 결과 불러오기",
        use_container_width=True,
    ):
        try:
            response = requests.get(
                LATEST_RESULT_URL,
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()

            result_df = pd.DataFrame(
                payload.get("results", [])
            )

            if result_df.empty:
                st.info("저장된 예측 결과가 없습니다.")
            else:
                pay_count = int(
                    (result_df["prediction_value"] == 1).sum()
                )

                c1, c2 = st.columns(2)
                c1.metric("최근 예측 고객", f"{len(result_df):,}명")
                c2.metric("결제 예상", f"{pay_count:,}명")

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True,
                )

        except requests.RequestException as error:
            st.error(
                "FastAPI에서 최신 결과를 가져오지 못했습니다. "
                "API가 실행 중인지 확인하세요."
            )
            st.code(str(error))
