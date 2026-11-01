import numpy as np
from sklearn.linear_model import LogisticRegression

from hotel_overbooking.model import features, make_preprocessor
from hotel_overbooking.scratch import fit_logistic_scratch, predict_scratch, prepare_matrix
from tests.test_model import fake_bookings

WEIGHT_TOLERANCE = 1e-3
PROBABILITY_TOLERANCE = 1e-3


def test_scratch_logistic_matches_sklearn():
    df = fake_bookings(300)
    preprocessor = make_preprocessor()
    x = prepare_matrix(preprocessor, features(df), fit=True)
    y = df["is_canceled"].to_numpy()

    reference = LogisticRegression(C=1.0, max_iter=5000, tol=1e-10).fit(x, y)
    weights, intercept = fit_logistic_scratch(x, y, c=1.0, steps=30000)

    assert np.abs(weights - reference.coef_[0]).max() < WEIGHT_TOLERANCE
    assert abs(intercept - reference.intercept_[0]) < WEIGHT_TOLERANCE
    gap = np.abs(predict_scratch(x, weights, intercept) - reference.predict_proba(x)[:, 1])
    assert gap.max() < PROBABILITY_TOLERANCE
