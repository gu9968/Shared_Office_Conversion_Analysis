from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import xgboost as xgb


st.set_page_config(
    page_title="공유오피스 결제 전환 예측",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(
    os.getenv(
        "COWORKING_MODEL_DIR",
        str(PROJECT_ROOT / "models"),
    )
).expanduser()
RUNTIME_DIR = Path(
    os.getenv(
        "COWORKING_RUNTIME_DIR",
        str(PROJECT_ROOT / "runtime"),
    )
).expanduser()

MODEL_PATH = MODEL_DIR / "best_tuned_model.pkl"
THRESHOLD_PATH = MODEL_DIR / "best_tuned_threshold.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"
FEATURE_MEDIANS_PATH = MODEL_DIR / "feature_medians.json"

DEFAULT_THRESHOLD = 0.35
EPS = 1e-6
REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("PREDICTION_API_TIMEOUT", "10")
)

API_BASE_URL = os.getenv(
    "PREDICTION_API_URL",
    "http://localhost:8000",
).rstrip("/")

LATEST_RESULT_URL = f"{API_BASE_URL}/latest"
LOCAL_LATEST_RESULT_PATH = RUNTIME_DIR / "latest_prediction_result.csv"


st.markdown(
    """
    <style>
    :root { color-scheme: light; }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: #F4F7FB !important;
        color: #172033 !important;
    }

    [data-testid="stHeader"] {
        background: rgba(244,247,251,.92) !important;
        border-bottom: 1px solid #D9E2EC;
    }

    .block-container {
        max-width: 1320px;
        padding: 2.4rem 2.5rem 5rem;
    }

    h1, h2, h3 {
        color: #102A43 !important;
        letter-spacing: -.035em;
    }

    section[data-testid="stSidebar"] {
        width: 285px !important;
        background: #FFFFFF !important;
        border-right: 1px solid #D9E2EC;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        width: 100%;
        min-height: 46px;
        display: flex;
        align-items: center;
        padding: .75rem .95rem !important;
        border-radius: 10px;
        border: 1px solid transparent;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"]
    label [data-baseweb="radio"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"]
    label:has(input:checked) {
        background: linear-gradient(
            135deg,
            #2563EB 0%,
            #1D4ED8 100%
        ) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"]
    label:has(input:checked) p {
        color: #FFFFFF !important;
    }

    .hero-section {
        background:
            radial-gradient(
                circle at 92% 10%,
                rgba(96,165,250,.26),
                transparent 34%
            ),
            linear-gradient(
                135deg,
                #0F3D67 0%,
                #155E9D 54%,
                #2563EB 100%
            );
        border-radius: 24px;
        padding: 2.8rem 3rem;
        margin-bottom: 1.8rem;
    }

    .hero-badge {
        display: inline-block;
        padding: .42rem .78rem;
        border-radius: 999px;
        background: rgba(255,255,255,.15);
        color: #DBEAFE;
        font-size: .82rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .hero-title {
        color: #FFFFFF;
        font-size: 2.55rem;
        font-weight: 850;
        line-height: 1.22;
    }

    .hero-description {
        color: #DBEAFE;
        font-size: 1.05rem;
        line-height: 1.8;
        max-width: 760px;
    }

    div[data-testid="stMetric"],
    [data-testid="stDataFrame"],
    [data-testid="stPlotlyChart"],
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #D9E2EC;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(16,42,67,.045);
    }

    div[data-testid="stMetric"] {
        padding: 1.3rem 1.35rem;
    }

    .status-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 5px solid #2563EB;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
        color: #1E3A8A;
        font-weight: 650;
    }

    .empty-box {
        background: #FFFFFF;
        border: 1px dashed #93C5FD;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: #627D98;
    }

    .section-caption {
        color: #64788C;
        font-size: .95rem;
        margin-bottom: 12px;
    }

    .insight-box {
        min-height: 185px;
        background: #FFFFFF;
        border: 1px solid #D9E2EC;
        border-radius: 18px;
        padding: 1.45rem 1.55rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(16,42,67,.045);
    }

    .insight-title {
        color: #486581;
        font-size: .92rem;
        font-weight: 700;
        margin-bottom: .6rem;
    }

    .insight-value {
        color: #1D4ED8;
        font-size: 1.65rem;
        font-weight: 850;
        line-height: 1.25;
        margin-bottom: .7rem;
    }

    .insight-text {
        color: #334E68;
        font-size: .96rem;
        line-height: 1.65;
    }

    .process-box {
        min-height: 108px;
        background: #FFFFFF;
        border: 1px solid #D9E2EC;
        border-radius: 14px;
        padding: 1.15rem 1rem;
        box-shadow: 0 6px 18px rgba(16,42,67,.04);
    }

    .process-number {
        color: #2563EB;
        font-size: .82rem;
        font-weight: 800;
        margin-bottom: .45rem;
    }

    .process-label {
        color: #102A43;
        font-size: .96rem;
        font-weight: 750;
        line-height: 1.4;
    }

    .prediction-success,
    .prediction-warning {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 1rem 0 1.2rem;
        font-weight: 650;
        line-height: 1.65;
    }

    .prediction-success {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 5px solid #10B981;
        color: #065F46;
    }

    .prediction-warning {
        background: #FFF7ED;
        border: 1px solid #FED7AA;
        border-left: 5px solid #F59E0B;
        color: #9A3412;
    }

    @media (max-width: 900px) {
        .hero-section {
            padding: 2rem 1.5rem;
        }

        .hero-title {
            font-size: 2rem;
        }

        .insight-box,
        .process-box {
            min-height: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_prediction_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}"
        )

    loaded_model = joblib.load(MODEL_PATH)
    model_feature_cols = loaded_model.get_booster().feature_names

    if not model_feature_cols:
        raise ValueError("모델의 피처명을 불러오지 못했습니다.")

    model_feature_cols = [
        str(column) for column in model_feature_cols
    ]

    if THRESHOLD_PATH.exists():
        loaded_threshold = float(joblib.load(THRESHOLD_PATH))
    else:
        loaded_threshold = DEFAULT_THRESHOLD

    if not 0 <= loaded_threshold <= 1:
        raise ValueError(
            "Threshold는 0 이상 1 이하이어야 합니다. "
            f"현재 값: {loaded_threshold}"
        )

    if FEATURE_COLS_PATH.exists():
        with FEATURE_COLS_PATH.open(encoding="utf-8") as file:
            saved_feature_cols = [
                str(column) for column in json.load(file)
            ]

        if saved_feature_cols != model_feature_cols:
            raise ValueError(
                "feature_cols.json의 피처 순서가 "
                "모델 내부 피처와 일치하지 않습니다."
            )

        loaded_feature_cols = saved_feature_cols
    else:
        loaded_feature_cols = model_feature_cols

    loaded_feature_medians: dict[str, float] = {}

    if FEATURE_MEDIANS_PATH.exists():
        with FEATURE_MEDIANS_PATH.open(encoding="utf-8") as file:
            raw_medians = json.load(file)

        loaded_feature_medians = {
            str(column): float(value)
            for column, value in raw_medians.items()
            if value is not None
        }

    return (
        loaded_model,
        loaded_threshold,
        loaded_feature_cols,
        loaded_feature_medians,
    )


try:
    model, THRESHOLD, feature_cols, feature_medians = (
        load_prediction_assets()
    )
except Exception as error:
    st.error("예측 자산을 불러오지 못했습니다.")
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


def make_features(input_df: pd.DataFrame):
    df = input_df.copy()
    missing_cols = [
        column
        for column in BASE_FEATURES
        if column not in df.columns
    ]

    if missing_cols:
        return None, missing_cols

    for column in BASE_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if "trial_date" in df.columns:
        trial_date = pd.to_datetime(
            df["trial_date"],
            errors="coerce",
        )
        df["trial_month"] = trial_date.dt.month.fillna(1)
        df["trial_dayofweek"] = (
            trial_date.dt.dayofweek.fillna(0)
        )
    else:
        df["trial_month"] = 1
        df["trial_dayofweek"] = 0

    df["is_weekend_trial"] = (
        df["trial_dayofweek"] >= 5
    ).astype(int)

    df["stay_x_visit"] = (
        df["avg_stay_hour"] * df["visit_days"]
    )
    df["enter_x_visit"] = (
        df["avg_daily_enter"] * df["visit_days"]
    )
    df["stay_x_enter"] = (
        df["avg_stay_hour"] * df["avg_daily_enter"]
    )
    df["enter_per_visit_day"] = (
        df["avg_daily_enter"]
        / (df["visit_days"] + EPS)
    )
    df["stay_per_enter"] = (
        df["avg_stay_hour"]
        / (df["avg_daily_enter"] + EPS)
    )
    df["delay_per_visit_day"] = (
        df["first_visit_delay"]
        / (df["visit_days"] + EPS)
    )
    df["visit_delay_interaction"] = (
        df["visit_days"] * df["first_visit_delay"]
    )
    df["is_fast_visit"] = (
        df["first_visit_delay"] <= 1
    ).astype(int)
    df["is_delayed_visit"] = (
        df["first_visit_delay"] >= 3
    ).astype(int)
    df["is_frequent_user"] = (
        df["avg_daily_enter"] >= 2
    ).astype(int)
    df["is_long_stay"] = (
        df["avg_stay_hour"] >= 2
    ).astype(int)

    df["is_short_frequent"] = (
        (df["avg_stay_hour"] < 1)
        & (df["avg_daily_enter"] >= 2)
    ).astype(int)

    df["is_long_frequent"] = (
        (df["avg_stay_hour"] >= 2)
        & (df["avg_daily_enter"] >= 2)
    ).astype(int)

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

    df["log1p_avg_stay_hour"] = np.log1p(
        df["avg_stay_hour"].clip(lower=0)
    )
    df["log1p_avg_daily_enter"] = np.log1p(
        df["avg_daily_enter"].clip(lower=0)
    )
    df["log1p_first_visit_delay"] = np.log1p(
        df["first_visit_delay"].clip(lower=0)
    )
    df["log1p_area_pyeong"] = np.log1p(
        df["area_pyeong"].clip(lower=0)
    )
    df["log1p_n_sites_visited"] = np.log1p(
        df["n_sites_visited"].clip(lower=0)
    )
    df["log1p_stay_x_visit"] = np.log1p(
        df["stay_x_visit"].clip(lower=0)
    )

    missing_model_features = [
        column
        for column in feature_cols
        if column not in df.columns
    ]

    if missing_model_features:
        return None, missing_model_features

    X = df[feature_cols].copy()

    for column in feature_cols:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    for column, median in feature_medians.items():
        if column in X.columns:
            X[column] = X[column].fillna(median)

    return X, []


def predict_payment(input_df: pd.DataFrame):
    X, missing_cols = make_features(input_df)

    if missing_cols:
        return None, missing_cols

    dmatrix = xgb.DMatrix(
        data=X.to_numpy(dtype=np.float32),
        feature_names=feature_cols,
        missing=np.nan,
    )

    probabilities = model.get_booster().predict(dmatrix)
    predictions = (probabilities >= THRESHOLD).astype(int)

    result = input_df.copy()
    result["payment_probability"] = probabilities
    result["payment_probability_percent"] = (
        probabilities * 100
    ).round(1)
    result["prediction_value"] = predictions
    result["prediction"] = np.where(
        predictions == 1,
        "결제예상",
        "미결제예상",
    )

    return result, []


def fetch_latest_result():
    # FastAPI와 Streamlit이 같은 프로젝트 폴더에서 실행되는 경우
    # API가 저장한 최신 결과 CSV를 우선 직접 읽습니다.
    if LOCAL_LATEST_RESULT_PATH.exists():
        result_df = pd.read_csv(
            LOCAL_LATEST_RESULT_PATH
        )

        modified_at = datetime.fromtimestamp(
            LOCAL_LATEST_RESULT_PATH.stat().st_mtime
        ).astimezone()

        return (
            result_df,
            modified_at.isoformat(),
            None,
            "로컬 결과 파일",
        )

    # 로컬 파일이 없으면 FastAPI의 /latest를 조회합니다.
    response = requests.get(
        LATEST_RESULT_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code == 404:
        return (
            None,
            None,
            "아직 자동 예측 결과가 없습니다. "
            "Google Drive 업로드 후 n8n 실행 기록을 확인하세요.",
            "FastAPI",
        )

    response.raise_for_status()
    payload = response.json()

    result_df = pd.DataFrame(
        payload.get("results", [])
    )
    updated_at = payload.get("updated_at")

    return (
        result_df,
        updated_at,
        None,
        "FastAPI",
    )

def format_updated_at(value):
    if not value:
        return "-"

    timestamp = pd.to_datetime(value, errors="coerce")

    if pd.isna(timestamp):
        return str(value)

    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def show_result_dashboard(result_df, updated_at):
    if result_df is None or result_df.empty:
        st.markdown(
            """
            <div class="empty-box">
                표시할 예측 결과가 없습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    probability_col = "payment_probability"
    probability_percent_col = (
        "payment_probability_percent"
    )
    prediction_value_col = "prediction_value"
    prediction_label_col = "prediction"

    total_count = len(result_df)
    pay_count = int(
        (result_df[prediction_value_col] == 1).sum()
    )
    nonpay_count = total_count - pay_count
    pay_rate = (
        pay_count / total_count
        if total_count
        else 0
    )

    st.markdown(
        f"""
        <div class="status-box">
            마지막 자동 예측 결과 ·
            {format_updated_at(updated_at)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 고객", f"{total_count:,}명")
    c2.metric("결제 예상", f"{pay_count:,}명")
    c3.metric("미결제 예상", f"{nonpay_count:,}명")
    c4.metric("예상 전환율", f"{pay_rate:.1%}")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        summary_df = pd.DataFrame(
            {
                "예측 결과": [
                    "결제 예상",
                    "미결제 예상",
                ],
                "고객 수": [
                    pay_count,
                    nonpay_count,
                ],
            }
        )

        fig = px.pie(
            summary_df,
            names="예측 결과",
            values="고객 수",
            hole=0.56,
            title="결제 예측 구성",
            color="예측 결과",
            color_discrete_map={
                "결제 예상": "#2563EB",
                "미결제 예상": "#F59E0B",
            },
        )

        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#172033",
            legend_title_text="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with chart_col2:
        fig = px.histogram(
            result_df,
            x=probability_col,
            nbins=20,
            title="결제 확률 분포",
        )

        fig.add_vline(
            x=THRESHOLD,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text=f"분류 기준 {THRESHOLD:.2f}",
        )

        fig.update_traces(
            marker_color="#2563EB"
        )

        fig.update_layout(
            xaxis_title="결제 확률",
            yaxis_title="고객 수",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#172033",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()
    st.subheader("고객별 예측 결과")

    result_view = result_df.sort_values(
        probability_col,
        ascending=False,
    ).reset_index(drop=True)

    if "user_uuid" in result_view.columns:
        result_view["고객"] = (
            result_view["user_uuid"].astype(str)
        )
    else:
        result_view["고객"] = [
            f"고객 {index + 1}"
            for index in range(len(result_view))
        ]

    summary_view = result_view[
        [
            "고객",
            prediction_label_col,
            probability_percent_col,
        ]
    ].copy()

    summary_view.columns = [
        "고객",
        "예측 결과",
        "결제 확률(%)",
    ]

    selection_event = st.dataframe(
        summary_view,
        use_container_width=True,
        hide_index=True,
        height=360,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "결제 확률(%)": (
                st.column_config.ProgressColumn(
                    "결제 확률",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                )
            )
        },
    )

    selected_rows = selection_event.selection.rows

    if selected_rows:
        selected_customer = result_view.iloc[
            selected_rows[0]
        ]

        st.subheader("선택 고객 상세 결과")

        d1, d2, d3 = st.columns(3)
        d1.metric(
            "고객",
            selected_customer["고객"],
        )
        d2.metric(
            "예측 결과",
            selected_customer[prediction_label_col],
        )
        d3.metric(
            "결제 확률",
            (
                f"{selected_customer[probability_col]:.1%}"
            ),
        )

        excluded_cols = {
            probability_col,
            probability_percent_col,
            prediction_value_col,
            prediction_label_col,
            "고객",
        }

        detail_rows = []

        for column in result_view.columns:
            if column in excluded_cols:
                continue

            detail_rows.append(
                {
                    "항목": FEATURE_KR.get(
                        column,
                        column,
                    ),
                    "값": selected_customer[column],
                }
            )

        with st.expander(
            "상세정보 더 보기",
            expanded=True,
        ):
            st.dataframe(
                pd.DataFrame(detail_rows),
                use_container_width=True,
                hide_index=True,
            )

    csv_data = result_df.to_csv(
        index=False,
        encoding="utf-8-sig",
    )

    st.download_button(
        "현재 예측 결과 CSV 다운로드",
        data=csv_data,
        file_name="prediction_result.csv",
        mime="text/csv",
        use_container_width=True,
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
        <div class="hero-section">
            <div class="hero-badge">
                FREE TRIAL CONVERSION ANALYSIS
            </div>
            <div class="hero-title">
                공유오피스 무료체험 고객의<br>
                결제 전환 가능성을 예측합니다
            </div>
            <div class="hero-description">
                3일 무료체험 고객의 방문, 체류, 입실 행동을 분석하고
                XGBoost 모델을 활용해 고객별 결제 가능성을 산출합니다.
                예측 결과는 전환 가능 고객 선별과 영업 우선순위 설정에
                활용할 수 있습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("최종 분석 표본", "5,629명")
    col2.metric("최종 모델", "XGBoost")
    col3.metric("분류 기준", f"{THRESHOLD:.2f}")
    col4.metric("모델 피처", f"{len(feature_cols)}개")

    st.divider()
    st.subheader("분석 배경")

    st.write(
        """
        공유오피스 운영사는 신규 고객 유치를 위해 3일 무료체험
        프로그램을 운영하고 있습니다.

        체험자의 방문일수, 체류시간, 입실횟수, 첫 방문 시점 등의
        이용 행동을 분석해 결제로 이어지는 패턴을 확인하고,
        고객별 결제 여부를 예측하는 모델을 구축했습니다.
        """
    )

    st.subheader("분석 범위")

    scope_df = pd.DataFrame(
        {
            "구분": [
                "전체 신청자",
                "실제 방문자",
                "최종 머신러닝 표본",
            ],
            "인원": [
                9624,
                6132,
                5629,
            ],
            "설명": [
                "무료체험 신청 완료 고객",
                "체험기간 내 1회 이상 방문",
                "미방문, 비정상 기록, 결측치 제외",
            ],
        }
    )

    st.dataframe(
        scope_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("주요 발견점")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.markdown(
            """
            <div class="insight-box">
                <div class="insight-title">
                    방문일수와 결제율
                </div>
                <div class="insight-value">
                    33% → 46%
                </div>
                <div class="insight-text">
                    방문일수가 1일에서 3일로 늘어날수록
                    결제율도 증가했습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="insight-box">
                <div class="insight-title">
                    첫 방문 지연일과 결제율
                </div>
                <div class="insight-value">
                    당일 32.3% · 3일 후 44.1%
                </div>
                <div class="insight-text">
                    신청 직후보다 방문 시점을 계획해 방문한
                    고객의 결제율이 높았습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with insight_col2:
        st.markdown(
            """
            <div class="insight-box">
                <div class="insight-title">
                    평균 체류시간과 결제율
                </div>
                <div class="insight-value">
                    2~3시간 46.2%
                </div>
                <div class="insight-text">
                    2~3시간에서 최고점을 기록한 뒤 체류시간이
                    길어질수록 결제율이 낮아졌습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="insight-box">
                <div class="insight-title">
                    방문일수 × 체류시간
                </div>
                <div class="insight-value">
                    최고 결제율 57.5%
                </div>
                <div class="insight-text">
                    방문일수가 많고 체류시간이 짧은 고객군의
                    결제율이 가장 높았습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("분석 절차")

    process_cols = st.columns(5)
    process_items = [
        ("01", "데이터 정제"),
        ("02", "이용 패턴 분석"),
        ("03", "파생변수 생성"),
        ("04", "모델 학습·튜닝"),
        ("05", "결제 여부 예측"),
    ]

    for column, (number, label) in zip(
        process_cols,
        process_items,
    ):
        with column:
            st.markdown(
                f"""
                <div class="process-box">
                    <div class="process-number">
                        {number}
                    </div>
                    <div class="process-label">
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


elif menu == "모델 성능":
    st.title("모델 성능")

    st.markdown(
        """
        <div class="section-caption">
            결제 가능성이 있는 고객을 최대한 놓치지 않는 방향으로
            최종 모델을 선정했습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)

    metrics = [
        ("Accuracy", "0.44"),
        ("Precision", "0.40"),
        ("Recall", "0.94"),
        ("F1-score", "0.56"),
        ("ROC-AUC", "0.63"),
    ]

    for column, (name, value) in zip(cols, metrics):
        column.metric(name, value)

    st.divider()

    model_cols = st.columns(3)
    model_cols[0].metric("최종 모델", "XGBoost")
    model_cols[1].metric("Threshold", f"{THRESHOLD:.2f}")
    model_cols[2].metric(
        "입력 피처",
        f"{len(feature_cols)}개",
    )

    st.subheader("모델 선정 이유")

    st.write(
        """
        XGBoost는 방문일수, 체류시간, 입실횟수 사이의
        비선형 관계와 변수 간 상호작용을 학습할 수 있습니다.

        하이퍼파라미터 튜닝과 Threshold 조정을 거친 결과,
        결제 고객을 찾아내는 Recall이 0.94로 나타났습니다.
        """
    )

    st.warning(
        "Precision은 0.40입니다. 예측 결과는 결제 여부를 "
        "확정하는 용도보다 영업 우선순위를 정하는 용도로 "
        "활용하는 것이 적절합니다."
    )

    st.subheader("최종 하이퍼파라미터")

    parameter_df = pd.DataFrame(
        {
            "파라미터": [
                "learning_rate",
                "max_depth",
                "n_estimators",
                "gamma",
                "colsample_bytree",
                "Threshold",
            ],
            "설정값": [
                0.02,
                2,
                700,
                0.3,
                0.7,
                THRESHOLD,
            ],
        }
    )

    st.dataframe(
        parameter_df,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("모델 입력 피처 보기"):
        st.dataframe(
            pd.DataFrame(
                {
                    "번호": range(
                        1,
                        len(feature_cols) + 1,
                    ),
                    "피처명": feature_cols,
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


elif menu == "결제 전환 예측":
    st.title("결제 전환 예측")

    st.markdown(
        """
        <div class="section-caption">
            고객 정보를 직접 입력하거나 CSV 파일을 업로드해
            결제 전환 확률을 확인할 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    direct_tab, csv_tab = st.tabs(
        [
            "직접 입력",
            "CSV 업로드",
        ]
    )

    with direct_tab:
        st.subheader("고객 정보 입력")

        with st.form("single_customer_form"):
            input_col1, input_col2, input_col3 = (
                st.columns(3)
            )

            with input_col1:
                visit_days = st.selectbox(
                    "체험기간 방문일수",
                    [1, 2, 3],
                    index=1,
                )

                avg_stay_hour = st.number_input(
                    "평균 체류시간",
                    min_value=0.0,
                    max_value=24.0,
                    value=2.0,
                    step=0.1,
                )

                avg_daily_enter = st.number_input(
                    "하루 평균 입실횟수",
                    min_value=0.0,
                    max_value=30.0,
                    value=1.0,
                    step=0.1,
                )

            with input_col2:
                first_visit_delay = st.selectbox(
                    "신청 후 첫 방문까지 걸린 기간",
                    [0, 1, 2, 3],
                    format_func=lambda value: (
                        "신청 당일"
                        if value == 0
                        else f"{value}일 후"
                    ),
                )

                first_visit_hour = st.slider(
                    "첫 방문 시각",
                    min_value=0,
                    max_value=23,
                    value=10,
                    step=1,
                    format="%d시",
                )

                n_sites_visited = st.number_input(
                    "방문 지점 수",
                    min_value=1,
                    max_value=9,
                    value=1,
                    step=1,
                )

            with input_col3:
                area_pyeong = st.selectbox(
                    "주 이용 지점 면적",
                    [50, 100, 150],
                    index=1,
                    format_func=lambda value: (
                        f"{value}평"
                    ),
                )

                is_post_covid = st.selectbox(
                    "체험 시기",
                    [0, 1],
                    format_func=lambda value: (
                        "코로나 기간"
                        if value == 0
                        else "엔데믹 이후"
                    ),
                )

                consecutive_group = st.selectbox(
                    "최대 연속 방문일",
                    [
                        "1일 이하",
                        "2일 연속",
                        "3일 연속",
                    ],
                )

            trial_date = st.date_input(
                "무료체험 신청일"
            )

            submitted = st.form_submit_button(
                "결제 확률 예측",
                use_container_width=True,
            )

        if submitted:
            input_row = pd.DataFrame(
                [
                    {
                        "avg_stay_hour": (
                            avg_stay_hour
                        ),
                        "avg_daily_enter": (
                            avg_daily_enter
                        ),
                        "visit_days": visit_days,
                        "first_visit_delay": (
                            first_visit_delay
                        ),
                        "consecutive_group_2일": int(
                            consecutive_group
                            in {"2일 연속", "3일 연속"}
                        ),
                        "consecutive_group_3일": int(
                            consecutive_group
                            == "3일 연속"
                        ),
                        "first_visit_hour": (
                            first_visit_hour
                        ),
                        "n_sites_visited": (
                            n_sites_visited
                        ),
                        "area_pyeong": area_pyeong,
                        "is_post_covid": (
                            is_post_covid
                        ),
                        "trial_date": pd.Timestamp(
                            trial_date
                        ),
                    }
                ]
            )

            result, missing_cols = predict_payment(
                input_row
            )

            if missing_cols:
                st.error(
                    "입력값으로 피처를 생성하지 "
                    "못했습니다."
                )

            else:
                probability = float(
                    result.loc[
                        0,
                        "payment_probability",
                    ]
                )

                prediction_label = str(
                    result.loc[
                        0,
                        "prediction",
                    ]
                )

                st.divider()
                st.subheader("예측 결과")

                result_col1, result_col2, result_col3 = (
                    st.columns(3)
                )

                result_col1.metric(
                    "결제 확률",
                    f"{probability:.1%}",
                )

                result_col2.metric(
                    "분류 기준",
                    f"{THRESHOLD:.2f}",
                )

                result_col3.metric(
                    "예측 결과",
                    prediction_label,
                )

                st.progress(
                    min(
                        max(probability, 0.0),
                        1.0,
                    )
                )

                if probability >= THRESHOLD:
                    st.markdown(
                        """
                        <div class="prediction-success">
                            결제 예상 고객으로 분류되었습니다.
                            우선 안내 또는 결제 촉진 대상으로
                            활용할 수 있습니다.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:
                    st.markdown(
                        """
                        <div class="prediction-warning">
                            미결제 예상 고객으로 분류되었습니다.
                            추가 체험 안내나 고객 의견 확인이
                            필요할 수 있습니다.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.subheader("입력 정보")

                display_values = {
                    "체험기간 방문일수": (
                        f"{visit_days}일"
                    ),
                    "평균 체류시간": (
                        f"{avg_stay_hour:.1f}시간"
                    ),
                    "하루 평균 입실횟수": (
                        f"{avg_daily_enter:.1f}회"
                    ),
                    "첫 방문 시점": (
                        "신청 당일"
                        if first_visit_delay == 0
                        else (
                            f"신청 "
                            f"{first_visit_delay}일 후"
                        )
                    ),
                    "첫 방문 시각": (
                        f"{first_visit_hour}시"
                    ),
                    "방문 지점 수": (
                        f"{n_sites_visited}개"
                    ),
                    "주 이용 지점 면적": (
                        f"{area_pyeong}평"
                    ),
                    "체험 시기": (
                        "코로나 기간"
                        if is_post_covid == 0
                        else "엔데믹 이후"
                    ),
                    "최대 연속 방문일": (
                        consecutive_group
                    ),
                    "무료체험 신청일": (
                        str(trial_date)
                    ),
                }

                st.dataframe(
                    pd.DataFrame(
                        {
                            "항목": list(
                                display_values.keys()
                            ),
                            "입력값": list(
                                display_values.values()
                            ),
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    with csv_tab:
        st.subheader("고객 CSV 업로드")

        with st.expander("필수 컬럼 확인"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "컬럼명": BASE_FEATURES,
                        "설명": [
                            FEATURE_KR[column]
                            for column in (
                                BASE_FEATURES
                            )
                        ],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        uploaded_file = st.file_uploader(
            "CSV 파일 선택",
            type=["csv"],
            key="customer_csv_uploader",
        )

        if uploaded_file is None:
            st.info(
                "CSV 파일을 업로드하면 각 행을 고객 "
                "1명으로 보고 결제 확률을 예측합니다."
            )

        else:
            try:
                uploaded_df = pd.read_csv(
                    uploaded_file
                )

            except Exception as error:
                st.error(
                    "CSV 파일을 읽지 못했습니다."
                )
                st.code(str(error))
                st.stop()

            result, missing_cols = predict_payment(
                uploaded_df
            )

            if missing_cols:
                st.error(
                    "예측에 필요한 컬럼이 부족합니다."
                )

                st.dataframe(
                    pd.DataFrame(
                        {
                            "누락 컬럼": missing_cols,
                            "설명": [
                                FEATURE_KR.get(
                                    column,
                                    column,
                                )
                                for column in (
                                    missing_cols
                                )
                            ],
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                show_result_dashboard(
                    result,
                    "직접 업로드한 CSV",
                )


elif menu == "자동 예측 결과":
    st.title("자동 예측 결과")

    st.markdown(
        """
        <div class="section-caption">
            Google Drive에 입력 CSV가 업로드되면
            n8n과 FastAPI가 자동으로 예측을 수행합니다.
            이 페이지는 가장 최근의 자동 예측 결과를 표시합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "최신 결과 새로고침",
        use_container_width=True,
    ):
        st.rerun()

    try:
        latest_result_df, updated_at, error_message, result_source = (
            fetch_latest_result()
        )

        if error_message:
            st.info(error_message)

        else:
            st.caption(f"결과 불러온 위치: {result_source}")
            show_result_dashboard(
                latest_result_df,
                updated_at,
            )

    except requests.ConnectionError:
        st.error(
            "FastAPI 서버에 연결할 수 없습니다. "
            "api/main.py가 실행 중인지 확인하세요."
        )

        st.code(
            "uvicorn api.main:app "
            "--host 0.0.0.0 --port 8000 --reload"
        )

    except requests.HTTPError as error:
        st.error(
            "자동 예측 결과를 불러오지 못했습니다."
        )
        st.code(str(error))

    except Exception as error:
        st.error(
            "예측 결과 화면 처리 중 오류가 발생했습니다."
        )
        st.code(str(error))
