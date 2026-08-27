from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. 프로젝트 경로
# ============================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_DIR = PROJECT_ROOT / "models"
RUNTIME_DIR = PROJECT_ROOT / "runtime"

MODEL_PATH = MODEL_DIR / "best_tuned_model.pkl"
THRESHOLD_PATH = MODEL_DIR / "best_tuned_threshold.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"
FEATURE_MEDIANS_PATH = MODEL_DIR / "feature_medians.json"

LATEST_RESULT_PATH = RUNTIME_DIR / "latest_prediction_result.csv"

DEFAULT_THRESHOLD = 0.35
EPS = 1e-6

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 입력 컬럼
# ============================================================
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

COLUMN_ALIASES = {
    # 시연용 한글 CSV
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

    # 추가 호환 표기
    "평균 체류시간": "avg_stay_hour",
    "하루 평균 입실횟수": "avg_daily_enter",
    "방문일수(일)": "visit_days",
    "첫 방문 지연일": "first_visit_delay",
    "첫 방문 시각": "first_visit_hour",
    "방문 지점 수": "n_sites_visited",
    "지점 면적": "area_pyeong",
    "엔데믹 이후 여부": "is_post_covid",
}


# ============================================================
# 3. API 요청 스키마
# ============================================================
class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="고객별 입력 데이터를 JSON 배열로 전달합니다.",
    )


# ============================================================
# 4. 모델 자산 로드
# ============================================================
def load_prediction_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    threshold = (
        float(joblib.load(THRESHOLD_PATH))
        if THRESHOLD_PATH.exists()
        else DEFAULT_THRESHOLD
    )

    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold는 0~1 범위여야 합니다: {threshold}")

    model_feature_cols = model.get_booster().feature_names
    if not model_feature_cols:
        raise ValueError("모델 내부 피처명을 불러오지 못했습니다.")

    model_feature_cols = [str(c) for c in model_feature_cols]

    if FEATURE_COLS_PATH.exists():
        with FEATURE_COLS_PATH.open("r", encoding="utf-8") as f:
            saved_feature_cols = [str(c) for c in json.load(f)]

        if saved_feature_cols != model_feature_cols:
            raise ValueError(
                "feature_cols.json의 피처명/순서가 모델 내부 피처와 일치하지 않습니다."
            )
        feature_cols = saved_feature_cols
    else:
        feature_cols = model_feature_cols

    feature_medians: dict[str, float] = {}
    if FEATURE_MEDIANS_PATH.exists():
        with FEATURE_MEDIANS_PATH.open("r", encoding="utf-8") as f:
            feature_medians = {
                str(k): float(v)
                for k, v in json.load(f).items()
            }

    return model, threshold, feature_cols, feature_medians


model, threshold, feature_cols, feature_medians = load_prediction_assets()


# ============================================================
# 5. 입력 정규화 / 검증
# ============================================================
def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(
        columns={
            c: COLUMN_ALIASES.get(str(c).strip(), str(c).strip())
            for c in df.columns
        }
    )
    return renamed


def validate_input_data(df: pd.DataFrame) -> None:
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError("예측에 필요한 기본 컬럼이 없습니다: " + ", ".join(missing))

    all_missing = [c for c in BASE_FEATURES if df[c].isna().all()]
    if all_missing:
        raise ValueError("모든 값이 결측인 컬럼이 있습니다: " + ", ".join(all_missing))

    if (~df["visit_days"].between(0, 3, inclusive="both")).any():
        raise ValueError("visit_days는 0~3 범위여야 합니다.")

    if (~df["first_visit_hour"].between(0, 23, inclusive="both")).any():
        raise ValueError("first_visit_hour는 0~23 범위여야 합니다.")

    for col in [
        "consecutive_group_2일",
        "consecutive_group_3일",
        "is_post_covid",
    ]:
        if (~df[col].isin([0, 1])).any():
            raise ValueError(f"{col}은 0 또는 1이어야 합니다.")

    invalid_consecutive = (
        (df["consecutive_group_3일"] == 1)
        & (df["consecutive_group_2일"] == 0)
    )
    if invalid_consecutive.any():
        raise ValueError(
            "consecutive_group_3일이 1이면 consecutive_group_2일도 1이어야 합니다."
        )


# ============================================================
# 6. 최종 노트북 기준 42개 피처 생성
# ============================================================
def make_features(input_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_input_columns(input_df.copy())

    for col in BASE_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    validate_input_data(df)

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

    # 최종 모델 학습 노트북과 동일한 시간대 정의
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

    for col in [
        "avg_stay_hour",
        "avg_daily_enter",
        "first_visit_delay",
        "area_pyeong",
        "n_sites_visited",
        "stay_x_visit",
    ]:
        df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))

    X = df.reindex(columns=feature_cols).copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # 학습 데이터 중앙값으로 결측 보정
    for col in feature_cols:
        if col in feature_medians:
            X[col] = X[col].fillna(feature_medians[col])

    remaining_missing = [
        c for c in feature_cols
        if X[c].isna().any()
    ]
    if remaining_missing:
        raise ValueError(
            "결측치를 처리할 수 없는 피처가 있습니다: "
            + ", ".join(remaining_missing)
        )

    return X


# ============================================================
# 7. 예측
# ============================================================
def predict_payment(input_df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_input_columns(input_df.copy())
    X = make_features(normalized)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result = normalized.copy()
    result["payment_probability"] = probabilities.astype(float)
    result["payment_probability_percent"] = (probabilities * 100).round(1)
    result["prediction_value"] = predictions.astype(int)
    result["prediction"] = np.where(
        predictions == 1,
        "결제예상",
        "미결제예상",
    )

    return result


def save_latest_result(result_df: pd.DataFrame) -> None:
    result_df.to_csv(
        LATEST_RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    response_df = df.replace([np.inf, -np.inf], np.nan)
    response_df = response_df.astype(object).where(
        pd.notna(response_df),
        None,
    )
    return response_df.to_dict(orient="records")


# ============================================================
# 8. FastAPI
# ============================================================
app = FastAPI(
    title="공유오피스 결제전환 예측 API",
    description="3일 무료체험 고객의 이용 행동 기반 결제 전환 예측 API",
    version="2.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Prediction API is running",
        "predict_endpoint": "/predict",
        "latest_endpoint": "/latest",
        "health_endpoint": "/health",
        "features_endpoint": "/features",
        "threshold": threshold,
        "feature_count": len(feature_cols),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "threshold": threshold,
        "feature_count": len(feature_cols),
        "feature_medians_loaded": bool(feature_medians),
        "latest_result_exists": LATEST_RESULT_PATH.exists(),
    }


@app.get("/features")
def features():
    return {
        "feature_count": len(feature_cols),
        "features": feature_cols,
    }


@app.get("/assets")
def assets():
    return {
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "threshold_path": str(THRESHOLD_PATH),
        "threshold_exists": THRESHOLD_PATH.exists(),
        "feature_cols_path": str(FEATURE_COLS_PATH),
        "feature_cols_exists": FEATURE_COLS_PATH.exists(),
        "feature_medians_path": str(FEATURE_MEDIANS_PATH),
        "feature_medians_exists": FEATURE_MEDIANS_PATH.exists(),
    }


@app.get("/latest")
def latest():
    if not LATEST_RESULT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="아직 저장된 예측 결과가 없습니다.",
        )

    latest_df = pd.read_csv(LATEST_RESULT_PATH)
    return {
        "count": len(latest_df),
        "results": dataframe_to_records(latest_df),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_df = pd.DataFrame(request.records)

        if input_df.empty:
            raise HTTPException(
                status_code=400,
                detail="입력 데이터가 비어 있습니다.",
            )

        result_df = predict_payment(input_df)
        save_latest_result(result_df)

        return {
            "count": len(result_df),
            "threshold": threshold,
            "payment_count": int(
                (result_df["prediction_value"] == 1).sum()
            ),
            "non_payment_count": int(
                (result_df["prediction_value"] == 0).sum()
            ),
            "results": dataframe_to_records(result_df),
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"예측 처리 중 오류가 발생했습니다: {error}",
        ) from error
