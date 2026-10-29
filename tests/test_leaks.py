import pandas as pd

from hotel_overbooking.leaks import EXCLUDED_COLUMNS, checkin_signals, outcome_leaks, usable_columns


def sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "is_canceled": [1, 1, 0, 0],
            "reservation_status": ["Canceled", "No-Show", "Check-Out", "Check-Out"],
            "reservation_status_date": ["2016-01-01", "2016-02-10", "2016-03-05", "2016-03-07"],
            "arrival_date": pd.to_datetime(["2016-03-01", "2016-02-10", "2016-03-05", "2016-03-07"]),
            "reserved_room_type": ["A", "A", "A", "A"],
            "assigned_room_type": ["A", "A", "D", "A"],
            "booking_changes": [0, 0, 2, 0],
            "required_car_parking_spaces": [0, 0, 1, 0],
            "total_of_special_requests": [0, 1, 2, 0],
            "adr": [90.0, 0.0, 100.0, 80.0],
            "country": ["PRT", None, "GBR", "FRA"],
            "lead_time": [100, 30, 10, 5],
        }
    )


def test_status_gives_the_outcome_away():
    result = outcome_leaks(sample())
    assert result["canceled_with_status_canceled_or_no_show"] == 1.0
    assert result["kept_with_status_check_out"] == 1.0


def test_status_date_is_the_arrival_day_for_kept_bookings():
    result = outcome_leaks(sample())
    assert result["kept_status_date_before_arrival"] == 0.0
    assert result["canceled_status_date_before_arrival"] == 0.5


def test_checkin_signals_rates():
    table = checkin_signals(sample())
    assert table.loc["assigned_differs_from_reserved", "kept"] == 0.5
    assert table.loc["parking_spaces_above_0", "canceled"] == 0.0
    assert table.loc["country_missing", "canceled"] == 0.5


def test_usable_columns_drop_the_excluded_list_and_the_label():
    columns = usable_columns(sample())
    assert "is_canceled" not in columns
    assert all(name not in columns for name in EXCLUDED_COLUMNS)
    assert "lead_time" in columns
