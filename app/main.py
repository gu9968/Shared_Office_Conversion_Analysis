from __future__ import annotations

import json
import logging
import os
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. 기본 경로 설정
# ============================================================

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

LATEST_RESULT_PATH = (
    RUNTIME_DIR / "latest_prediction_result.csv"
)

DEFAULT_THRESHOLD = 0.35
EPS = 1e-6
MAX_BATCH_SIZE = int(os.getenv("PREDICTION_MAX_BATCH_SIZE", "10000"))

logger = logging.getLogger(__name__)

RUNTIME_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. API 입력 데이터 형식
# ============================================================

class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )


# ============================================================
# 3. 모델·Threshold·피처 목록 불러오기
# ============================================================

def load_prediction_assets():
    # --------------------------------------------------------
    # 모델 파일 확인
    # --------------------------------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}"
        )

    loaded_model = joblib.load(MODEL_PATH)

    # --------------------------------------------------------
    # Threshold 불러오기
    # --------------------------------------------------------
    if THRESHOLD_PATH.exists():
        loaded_threshold = float(
            joblib.load(THRESHOLD_PATH)
        )
    else:
        loaded_threshold = DEFAULT_THRESHOLD

    if not 0 <= loaded_threshold <= 1:
        raise ValueError(
            "Threshold는 0 이상 1 이하이어야 합니다. "
            f"현재 값: {loaded_threshold}"
        )

    # --------------------------------------------------------
    # 모델 내부 피처 목록 불러오기
    # --------------------------------------------------------
    try:
        model_feature_cols = (
            loaded_model
            .get_booster()
            .feature_names
        )

    except Exception as error:
        raise ValueError(
            "XGBoost 모델 내부에서 피처명을 "
            "불러오지 못했습니다."
        ) from error

    if not model_feature_cols:
        raise ValueError(
            "모델 내부에 학습 피처명이 저장되어 있지 않습니다."
        )

    model_feature_cols = [
        str(column)
        for column in model_feature_cols
    ]

    # --------------------------------------------------------
    # feature_cols.json 검사
    # --------------------------------------------------------
    if FEATURE_COLS_PATH.exists():
        with open(
            FEATURE_COLS_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            saved_feature_cols = json.load(file)

        saved_feature_cols = [
            str(column)
            for column in saved_feature_cols
        ]

        if saved_feature_cols != model_feature_cols:
            raise ValueError(
                "feature_cols.json의 피처명 또는 순서가 "
                "모델 내부 피처와 일치하지 않습니다."
            )

        loaded_feature_cols = saved_feature_cols

    else:
        loaded_feature_cols = model_feature_cols

    # --------------------------------------------------------
    # 학습 데이터 중앙값 불러오기
    #
    # 파일이 없으면 빈 딕셔너리 사용
    # 입력 CSV에 결측값이 없다면 예측 가능
    # --------------------------------------------------------
    feature_medians: dict[str, float] = {}

    if FEATURE_MEDIANS_PATH.exists():
        with open(
            FEATURE_MEDIANS_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            raw_medians = json.load(file)

        feature_medians = {
            str(column): float(value)
            for column, value in raw_medians.items()
            if value is not None
        }

    return (
        loaded_model,
        loaded_threshold,
        loaded_feature_cols,
        feature_medians,
    )


(
    model,
    threshold,
    feature_cols,
    feature_medians,
) = load_prediction_assets()


# ============================================================
# 4. CSV에 반드시 포함되어야 하는 기본 입력 피처
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


# ============================================================
# 4-1. n8n/CSV 입력 컬럼명 표준화
# ============================================================

COLUMN_ALIASES = {
    "고객 ID": "customer_id",
    "고객ID": "customer_id",
    "체험 날짜": "trial_date",
    "체험날짜": "trial_date",
    "재판 날짜": "trial_date",
    "재판날짜": "trial_date",
    "평균 체류 시간": "avg_stay_hour",
    "평균체류시간": "avg_stay_hour",
    "평균 일일 입실": "avg_daily_enter",
    "평균일일입실": "avg_daily_enter",
    "방문일수": "visit_days",
    "방문 일수": "visit_days",
    "첫 방문 지연": "first_visit_delay",
    "첫방문지연": "first_visit_delay",
    "연속_그룹_2일": "consecutive_group_2일",
    "연속 그룹 2일": "consecutive_group_2일",
    "연속_그룹_3일": "consecutive_group_3일",
    "연속 그룹 3일": "consecutive_group_3일",
    "첫 방문 시간": "first_visit_hour",
    "첫방문시간": "first_visit_hour",
    "방문한 사이트 수": "n_sites_visited",
    "방문 사이트 수": "n_sites_visited",
    "평 지역": "area_pyeong",
    "평지역": "area_pyeong",
    "면적(평)": "area_pyeong",
    "코로나19 이후": "is_post_covid",
    "코로나 이후": "is_post_covid",
}


def normalize_input_columns(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    df = input_df.copy()

    df.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in df.columns
    ]

    rename_map = {
        column: COLUMN_ALIASES[column]
        for column in df.columns
        if column in COLUMN_ALIASES
    }

    df = df.rename(columns=rename_map)

    if df.columns.duplicated().any():
        merged_columns = {}

        for column in dict.fromkeys(df.columns):
            same_name = df.loc[:, df.columns == column]

            if same_name.shape[1] == 1:
                merged_columns[column] = same_name.iloc[:, 0]
            else:
                merged_columns[column] = (
                    same_name.bfill(axis=1).iloc[:, 0]
                )

        df = pd.DataFrame(merged_columns)

    if "trial_date" in df.columns:
        trial_text = (
            df["trial_date"]
            .astype(str)
            .str.strip()
            .str.replace("년", "-", regex=False)
            .str.replace("월", "-", regex=False)
            .str.replace("일", "", regex=False)
            .str.replace(r"\s+", "", regex=True)
        )

        parsed_trial_date = pd.to_datetime(
            trial_text,
            errors="coerce",
        )

        fallback_trial_date = pd.to_datetime(
            df["trial_date"],
            errors="coerce",
        )

        df["trial_date"] = parsed_trial_date.fillna(
            fallback_trial_date
        )

    return df


# ============================================================
# 5. 기본 입력 데이터 검사
# ============================================================

def validate_input_data(
    df: pd.DataFrame,
) -> None:
    # --------------------------------------------------------
    # 필수 컬럼 검사
    # --------------------------------------------------------
    missing_columns = [
        column
        for column in BASE_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "예측에 필요한 기본 컬럼이 없습니다: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # 전체 행이 결측인 컬럼 검사
    # --------------------------------------------------------
    all_missing_columns = [
        column
        for column in BASE_FEATURES
        if df[column].isna().all()
    ]

    if all_missing_columns:
        raise ValueError(
            "모든 값이 결측인 컬럼이 있습니다: "
            + ", ".join(all_missing_columns)
        )

    # --------------------------------------------------------
    # 논리적인 입력 범위 검사
    # --------------------------------------------------------
    invalid_visit_days = ~df["visit_days"].between(
        0,
        3,
        inclusive="both",
    )

    if invalid_visit_days.any():
        invalid_values = (
            df.loc[
                invalid_visit_days,
                "visit_days",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "visit_days는 0~3 범위여야 합니다. "
            f"확인된 값: {invalid_values}"
        )

    invalid_hour = ~df["first_visit_hour"].between(
        0,
        23,
        inclusive="both",
    )

    if invalid_hour.any():
        invalid_values = (
            df.loc[
                invalid_hour,
                "first_visit_hour",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "first_visit_hour는 0~23 범위여야 합니다. "
            f"확인된 값: {invalid_values}"
        )

    for binary_column in [
        "consecutive_group_2일",
        "consecutive_group_3일",
        "is_post_covid",
    ]:
        invalid_binary = ~df[binary_column].isin(
            [0, 1]
        )

        if invalid_binary.any():
            invalid_values = (
                df.loc[
                    invalid_binary,
                    binary_column,
                ]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                f"{binary_column}은 0 또는 1이어야 합니다. "
                f"확인된 값: {invalid_values}"
            )

    # 3일 연속 방문이면 2일 연속 방문도 성립
    invalid_consecutive = (
        (df["consecutive_group_3일"] == 1)
        & (df["consecutive_group_2일"] == 0)
    )

    if invalid_consecutive.any():
        raise ValueError(
            "consecutive_group_3일이 1인 경우 "
            "consecutive_group_2일도 1이어야 합니다."
        )


# ============================================================
# 6. 최종 노트북 기준 피처 엔지니어링
# ============================================================

def make_features(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    df = input_df.copy()

    # --------------------------------------------------------
    # 기본 피처 숫자형 변환
    # --------------------------------------------------------
    for column in BASE_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    validate_input_data(df)

    # --------------------------------------------------------
    # 체험 신청일 관련 피처
    # --------------------------------------------------------
    if "trial_date" in df.columns:
        trial_date = pd.to_datetime(
            df["trial_date"],
            errors="coerce",
        )

        df["trial_month"] = (
            trial_date
            .dt
            .month
            .fillna(1)
            .astype(int)
        )

        df["trial_dayofweek"] = (
            trial_date
            .dt
            .dayofweek
            .fillna(0)
            .astype(int)
        )

    else:
        # trial_date가 없을 때 기존 API 기본값 유지
        df["trial_month"] = 1
        df["trial_dayofweek"] = 0

    df["is_weekend_trial"] = (
        df["trial_dayofweek"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # 기본 상호작용 피처
    # --------------------------------------------------------
    df["stay_x_visit"] = (
        df["avg_stay_hour"]
        * df["visit_days"]
    )

    df["enter_x_visit"] = (
        df["avg_daily_enter"]
        * df["visit_days"]
    )

    df["stay_x_enter"] = (
        df["avg_stay_hour"]
        * df["avg_daily_enter"]
    )

    # --------------------------------------------------------
    # 비율 피처
    #
    # 최종 분석 노트북과 동일하게 EPS 사용
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 방문일수 × 방문지연 상호작용
    # --------------------------------------------------------
    df["visit_delay_interaction"] = (
        df["visit_days"]
        * df["first_visit_delay"]
    )

    # --------------------------------------------------------
    # 행동 특성 피처
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 첫 방문 시간대 피처
    #
    # 최종 노트북 기준
    # 오전: 06:00~11:59
    # 오후: 12:00~17:59
    # 저녁: 18:00~23:59
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 첫 방문 시간 순환형 변환
    # --------------------------------------------------------
    df["first_visit_hour_sin"] = np.sin(
        2
        * np.pi
        * df["first_visit_hour"]
        / 24
    )

    df["first_visit_hour_cos"] = np.cos(
        2
        * np.pi
        * df["first_visit_hour"]
        / 24
    )

    # --------------------------------------------------------
    # 체험 신청 월 순환형 변환
    # --------------------------------------------------------
    df["trial_month_sin"] = np.sin(
        2
        * np.pi
        * df["trial_month"]
        / 12
    )

    df["trial_month_cos"] = np.cos(
        2
        * np.pi
        * df["trial_month"]
        / 12
    )

    # --------------------------------------------------------
    # 엔데믹 이후 상호작용 피처
    # --------------------------------------------------------
    df["post_covid_x_visit_days"] = (
        df["is_post_covid"]
        * df["visit_days"]
    )

    df["post_covid_x_avg_stay"] = (
        df["is_post_covid"]
        * df["avg_stay_hour"]
    )

    df["post_covid_x_daily_enter"] = (
        df["is_post_covid"]
        * df["avg_daily_enter"]
    )

    # --------------------------------------------------------
    # 로그 변환 피처
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 모델 피처 생성 여부 검사
    #
    # 없는 피처를 임의로 0으로 만들지 않음
    # --------------------------------------------------------
    missing_model_features = [
        column
        for column in feature_cols
        if column not in df.columns
    ]

    if missing_model_features:
        raise ValueError(
            "API에서 생성하지 못한 모델 피처가 있습니다: "
            + ", ".join(missing_model_features)
        )

    # --------------------------------------------------------
    # 학습 당시 피처 순서 적용
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 학습 데이터 중앙값 파일이 있으면 동일하게 대체
    # --------------------------------------------------------
    for column in feature_cols:
        if column in feature_medians:
            X[column] = X[column].fillna(
                feature_medians[column]
            )

    # --------------------------------------------------------
    # 중앙값 파일 적용 후 남은 결측값 검사
    # --------------------------------------------------------
    remaining_missing = [
        column
        for column in feature_cols
        if X[column].isna().any()
    ]

    if remaining_missing:
        missing_detail = {
            column: int(X[column].isna().sum())
            for column in remaining_missing
        }

        raise ValueError(
            "예측 피처에 결측값이 남아 있습니다. "
            "feature_medians.json을 저장하거나 입력값을 확인하세요. "
            f"결측 현황: {missing_detail}"
        )

    return X


# ============================================================
# 7. 결제전환 예측
# ============================================================

def predict_payment(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    X = make_features(input_df)

    if X.empty:
        raise ValueError(
            "예측에 사용할 데이터가 비어 있습니다."
        )

    # --------------------------------------------------------
    # XGBoost Booster + DMatrix로 예측
    #
    # 저장 당시와 현재 XGBoost 버전 차이로 sklearn API에서
    # 피처명이 누락되는 문제를 방지하기 위해 명시적으로 전달
    # --------------------------------------------------------
    booster = model.get_booster()

    dmatrix = xgb.DMatrix(
        X,
        feature_names=feature_cols,
    )

    probabilities = booster.predict(
        dmatrix,
        validate_features=True,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    logger.info(
        "prediction_completed records=%d features=%d "
        "threshold=%.4f payment_count=%d",
        len(X),
        X.shape[1],
        threshold,
        int((predictions == 1).sum()),
    )

    # --------------------------------------------------------
    # 결과 데이터 생성
    # --------------------------------------------------------
    result = input_df.copy()

    result["payment_probability"] = (
        probabilities.astype(float)
    )

    result["payment_probability_percent"] = (
        probabilities * 100
    ).round(1)

    result["prediction_value"] = (
        predictions.astype(int)
    )

    result["prediction"] = np.where(
        predictions == 1,
        "결제예상",
        "미결제예상",
    )

    return result


# ============================================================
# 8. 최신 예측 결과 저장
# ============================================================

def save_latest_result(
    result_df: pd.DataFrame,
) -> None:
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=RUNTIME_DIR,
            prefix="latest_prediction_",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            result_df.to_csv(
                temporary_file,
                index=False,
            )

        temporary_path.replace(LATEST_RESULT_PATH)

    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


# ============================================================
# 9. JSON 응답용 데이터 정리
# ============================================================

def dataframe_to_records(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    response_df = df.copy()

    response_df = response_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    response_df = response_df.astype(object).where(
        pd.notna(response_df),
        None,
    )

    return response_df.to_dict(
        orient="records"
    )


# ============================================================
# 10. FastAPI 애플리케이션
# ============================================================

app = FastAPI(
    title="공유오피스 결제전환 예측 API",
    description=(
        "3일 무료체험 고객의 이용 행동을 바탕으로 "
        "유료 결제전환 가능성을 예측합니다."
    ),
    version="2.0.0",
)


# ============================================================
# 11. 기본 페이지
# ============================================================

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
        "feature_medians_loaded": bool(
            feature_medians
        ),
    }


# ============================================================
# 12. API 상태 확인
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "threshold": threshold,
        "feature_count": len(feature_cols),
        "feature_medians_loaded": bool(
            feature_medians
        ),
        "latest_result_exists": (
            LATEST_RESULT_PATH.exists()
        ),
    }


# ============================================================
# 13. 모델 피처 목록 확인
# ============================================================

@app.get("/features")
def features():
    return {
        "feature_count": len(feature_cols),
        "features": feature_cols,
    }


# ============================================================
# 14. 모델 및 파일 경로 확인
# ============================================================

@app.get("/assets")
def assets():
    return {
        "model_file": MODEL_PATH.name,
        "model_exists": MODEL_PATH.exists(),
        "threshold_file": THRESHOLD_PATH.name,
        "threshold_exists": (
            THRESHOLD_PATH.exists()
        ),
        "feature_cols_file": FEATURE_COLS_PATH.name,
        "feature_cols_exists": (
            FEATURE_COLS_PATH.exists()
        ),
        "feature_medians_file": FEATURE_MEDIANS_PATH.name,
        "feature_medians_exists": (
            FEATURE_MEDIANS_PATH.exists()
        ),
        "latest_result_file": LATEST_RESULT_PATH.name,
        "latest_result_exists": (
            LATEST_RESULT_PATH.exists()
        ),
    }


# ============================================================
# 15. 결제전환 예측
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest,
):
    try:
        input_df = pd.DataFrame(
            request.records
        )

        input_df = normalize_input_columns(
            input_df
        )

        if input_df.empty:
            raise HTTPException(
                status_code=400,
                detail="입력 데이터가 비어 있습니다.",
            )

        result_df = predict_payment(
            input_df
        )

        save_latest_result(
            result_df
        )

        payment_count = int(
            (
                result_df["prediction_value"] == 1
            ).sum()
        )

        non_payment_count = int(
            (
                result_df["prediction_value"] == 0
            ).sum()
        )

        return {
            "count": len(result_df),
            "threshold": threshold,
            "payment_count": payment_count,
            "non_payment_count": non_payment_count,
            "payment_rate_percent": round(
                payment_count
                / len(result_df)
                * 100,
                1,
            ),
            "results": dataframe_to_records(
                result_df
            ),
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception("prediction_failed")
        raise HTTPException(
            status_code=500,
            detail="예측 처리 중 서버 오류가 발생했습니다.",
        ) from error


# ============================================================
# 16. 최신 예측 결과 조회
# ============================================================

@app.get("/latest")
def latest():
    if not LATEST_RESULT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "아직 저장된 예측 결과가 없습니다."
            ),
        )

    try:
        result_df = pd.read_csv(
            LATEST_RESULT_PATH
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "최신 결과 파일을 읽지 못했습니다: "
                f"{error}"
            ),
        ) from error

    if result_df.empty:
        raise HTTPException(
            status_code=404,
            detail="최신 예측 결과가 비어 있습니다.",
        )

    if "prediction_value" in result_df.columns:
        prediction_values = pd.to_numeric(
            result_df["prediction_value"],
            errors="coerce",
        )

        payment_count = int(
            (prediction_values == 1).sum()
        )

        non_payment_count = int(
            (prediction_values == 0).sum()
        )

    else:
        payment_count = None
        non_payment_count = None

    modified_time = pd.Timestamp(
        LATEST_RESULT_PATH
        .stat()
        .st_mtime,
        unit="s",
        tz="Asia/Seoul",
    )

    return {
        "count": len(result_df),
        "threshold": threshold,
        "payment_count": payment_count,
        "non_payment_count": non_payment_count,
        "updated_at": modified_time.isoformat(),
        "results": dataframe_to_records(
            result_df
        ),
    }
