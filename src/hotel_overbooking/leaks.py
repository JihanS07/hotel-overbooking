import pandas as pd

# Known only after the outcome, or recorded at check-in. Never use these as model inputs.
EXCLUDED_COLUMNS = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "required_car_parking_spaces",
    "booking_changes",
]

# Looked at in the audit but kept for now. See the README for the numbers.
WATCHED_COLUMNS = ["adr", "total_of_special_requests", "country"]


def outcome_leaks(df: pd.DataFrame) -> dict[str, float]:
    """Share of canceled and kept bookings where each outcome-coded column gives the answer away."""
    canceled = df["is_canceled"] == 1
    status_date = pd.to_datetime(df["reservation_status_date"])
    return {
        "canceled_with_status_canceled_or_no_show": float(
            df.loc[canceled, "reservation_status"].isin(["Canceled", "No-Show"]).mean()
        ),
        "kept_with_status_check_out": float((df.loc[~canceled, "reservation_status"] == "Check-Out").mean()),
        "kept_status_date_before_arrival": float(
            (status_date[~canceled] < df.loc[~canceled, "arrival_date"]).mean()
        ),
        "canceled_status_date_before_arrival": float(
            (status_date[canceled] < df.loc[canceled, "arrival_date"]).mean()
        ),
    }


def checkin_signals(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column with simple rates for canceled versus kept bookings."""
    canceled = df["is_canceled"] == 1
    rows = {
        "assigned_differs_from_reserved": df["assigned_room_type"] != df["reserved_room_type"],
        "booking_changes_above_0": df["booking_changes"] > 0,
        "parking_spaces_above_0": df["required_car_parking_spaces"] > 0,
        "special_requests_above_0": df["total_of_special_requests"] > 0,
        "adr_is_0": df["adr"] == 0,
        "country_missing": df["country"].isna(),
    }
    table = {
        name: {"canceled": float(flag[canceled].mean()), "kept": float(flag[~canceled].mean())}
        for name, flag in rows.items()
    }
    return pd.DataFrame(table).T


def usable_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDED_COLUMNS and c != "is_canceled"]
