import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .leaks import EXCLUDED_COLUMNS

NUMBER_COLUMNS = [
    "lead_time",
    "arrival_date_week_number",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
    "adr",
    "total_of_special_requests",
]
# country, agent and company are left out because they have too many values for a first model
CATEGORY_COLUMNS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "deposit_type",
    "customer_type",
]
DEFAULT_CUTOFF = "2017-03-01"


def split_by_arrival(df: pd.DataFrame, cutoff: str = DEFAULT_CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_train = (df["arrival_date"] < pd.Timestamp(cutoff)).to_numpy()
    return df.loc[is_train].copy(), df.loc[~is_train].copy()


def feature_columns() -> list[str]:
    columns = NUMBER_COLUMNS + CATEGORY_COLUMNS
    assert not set(columns) & set(EXCLUDED_COLUMNS)
    return columns


def features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[feature_columns()].copy()
    x = x.fillna({"children": 0})  # 4 rows have no value
    return pd.DataFrame(x)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numbers", StandardScaler(), NUMBER_COLUMNS),
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORY_COLUMNS),
        ]
    )


def fit_logistic(train: pd.DataFrame) -> Pipeline:
    model = Pipeline([("prepare", make_preprocessor()), ("model", LogisticRegression(C=1.0, max_iter=1000))])
    return model.fit(features(train), train["is_canceled"])


def fit_boosting(train: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        [
            ("prepare", make_preprocessor()),
            ("model", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)),
        ]
    )
    return model.fit(features(train), train["is_canceled"])


def predict_cancel(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features(df))[:, 1]


def calibration_table(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "bookings": int(mask.sum()),
                "mean_predicted": float(prob[mask].mean()),
                "actual_rate": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def score(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {"auc": float(roc_auc_score(y_true, prob)), "brier": float(brier_score_loss(y_true, prob))}


def evaluate(df: pd.DataFrame, cutoff: str = DEFAULT_CUTOFF) -> dict[str, object]:
    train, test = split_by_arrival(df, cutoff)
    y_test = test["is_canceled"].to_numpy()
    base_rate = float(np.mean(train["is_canceled"].to_numpy()))
    probs = {
        "base rate": np.full(len(test), base_rate),
        "logistic regression": predict_cancel(fit_logistic(train), test),
        "gradient boosting": predict_cancel(fit_boosting(train), test),
    }
    scores = {name: score(y_test, p) for name, p in probs.items()}
    calibration = {name: calibration_table(y_test, p) for name, p in probs.items() if name != "base rate"}
    return {
        "train_bookings": len(train),
        "test_bookings": len(test),
        "train_cancel_rate": base_rate,
        "test_cancel_rate": float(y_test.mean()),
        "scores": scores,
        "calibration": calibration,
    }
