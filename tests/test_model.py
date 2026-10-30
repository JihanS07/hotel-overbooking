import numpy as np
import pandas as pd

from hotel_overbooking.leaks import EXCLUDED_COLUMNS
from hotel_overbooking.model import (
    CATEGORY_COLUMNS,
    NUMBER_COLUMNS,
    calibration_table,
    evaluate,
    feature_columns,
    score,
    split_by_arrival,
)


def fake_bookings(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    lead = rng.integers(0, 300, rows)
    data = {name: rng.integers(0, 3, rows) for name in NUMBER_COLUMNS}
    data["lead_time"] = lead
    data["adr"] = rng.uniform(50, 150, rows)
    for name in CATEGORY_COLUMNS:
        data[name] = rng.choice(["a", "b", "c"], rows)
    data["is_canceled"] = (rng.uniform(size=rows) < lead / 300).astype(int)
    data["arrival_date"] = pd.date_range("2016-01-01", periods=rows, freq="D")
    return pd.DataFrame(data)


def test_features_do_not_use_excluded_columns():
    assert not set(feature_columns()) & set(EXCLUDED_COLUMNS)


def test_split_puts_earlier_arrivals_in_train():
    train, test = split_by_arrival(fake_bookings(), "2016-09-01")
    assert train["arrival_date"].to_numpy().max() < test["arrival_date"].to_numpy().min()
    assert len(train) + len(test) == 400


def test_calibration_table_counts_every_booking():
    prob = np.array([0.05, 0.15, 0.95, 0.99])
    table = calibration_table(np.array([0, 0, 1, 1]), prob)
    assert int(table["bookings"].to_numpy().sum()) == 4
    assert table["actual_rate"].to_numpy()[-1] == 1.0


def test_score_of_perfect_predictions():
    result = score(np.array([0, 1, 0, 1]), np.array([0.0, 1.0, 0.0, 1.0]))
    assert result["auc"] == 1.0
    assert result["brier"] == 0.0


def test_models_beat_base_rate_on_fake_data():
    result = evaluate(fake_bookings(), "2016-09-01")
    scores = result["scores"]
    assert isinstance(scores, dict)
    assert scores["base rate"]["auc"] == 0.5
    assert scores["logistic regression"]["auc"] > 0.7
    assert scores["logistic regression"]["brier"] < scores["base rate"]["brier"]
