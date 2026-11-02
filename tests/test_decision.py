import numpy as np
import pandas as pd
import pytest

from hotel_overbooking.decision import (
    active_before_arrival,
    backtest,
    choose_accepted,
    estimate_capacity,
    expected_cost,
    show_distribution,
)


def test_show_distribution_matches_two_coin_flips():
    pmf = show_distribution(np.array([0.5, 0.5]))
    assert pmf == pytest.approx([0.25, 0.5, 0.25])


def test_show_distribution_sums_to_one_and_has_right_mean():
    probs = np.array([0.9, 0.2, 0.7, 0.4, 0.55])
    pmf = show_distribution(probs)
    assert pmf.sum() == pytest.approx(1.0)
    assert float(np.arange(len(pmf)) @ pmf) == pytest.approx(probs.sum())


def test_expected_cost_counts_walks_and_empty_rooms():
    pmf = np.array([0.0, 0.5, 0.5])
    assert expected_cost(pmf, capacity=1, walk_cost=4.0) == pytest.approx(0.5 * 4.0)


def test_sure_shows_are_never_overbooked():
    assert choose_accepted(np.ones(10), capacity=5, walk_cost=3.0) == 5


def test_likely_cancellations_lead_to_extra_bookings():
    assert choose_accepted(np.full(10, 0.5), capacity=5, walk_cost=1.0) > 5


def test_higher_walk_cost_accepts_fewer_extra():
    probs = np.full(30, 0.7)
    cheap = choose_accepted(probs, capacity=10, walk_cost=1.0)
    costly = choose_accepted(probs, capacity=10, walk_cost=10.0)
    assert costly <= cheap


def test_fewer_bookings_than_rooms_accepts_all():
    assert choose_accepted(np.full(3, 0.5), capacity=5, walk_cost=3.0) == 3


def small_bookings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hotel": ["City", "City", "City", "Resort"],
            "arrival_date": pd.to_datetime(["2017-05-10", "2017-05-10", "2017-05-11", "2017-05-10"]),
            "lead_time": [60, 10, 100, 45],
            "is_canceled": [0, 0, 1, 0],
            "reservation_status_date": ["2017-05-12", "2017-05-12", "2017-03-01", "2017-05-12"],
        }
    )


def test_capacity_is_busiest_night_of_kept_bookings():
    assert estimate_capacity(small_bookings()) == {"City": 2, "Resort": 1}


def test_active_drops_late_bookings_and_early_cancellations():
    active = active_before_arrival(small_bookings())
    assert active["lead_time"].tolist() == [60, 45]


def test_backtest_accepts_extra_when_guests_cancel():
    rows = 8
    test = pd.DataFrame(
        {
            "hotel": ["City"] * rows,
            "arrival_date": pd.to_datetime(["2017-06-01"] * rows),
            "lead_time": list(range(100, 100 - rows, -1)),
            "is_canceled": [0, 1, 0, 1, 0, 1, 0, 0],
        }
    )
    nights = backtest(test, np.full(rows, 0.5), {"City": 4}, walk_cost=1.0)
    assert len(nights) == 1
    assert nights.loc[0, "extra_accepted"] > 0
