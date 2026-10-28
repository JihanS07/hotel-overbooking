from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/hotels.csv")
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip


def load_bookings(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    month_number = {name: i + 1 for i, name in enumerate(MONTHS)}
    dates = [
        f"{year}-{month_number[month]:02d}-{day:02d}"
        for year, month, day in zip(
            df["arrival_date_year"].tolist(),
            df["arrival_date_month"].tolist(),
            df["arrival_date_day_of_month"].tolist(),
            strict=True,
        )
    ]
    df["arrival_date"] = pd.to_datetime(pd.Series(dates, index=df.index))
    return df


def summary(df: pd.DataFrame) -> dict[str, object]:
    canceled = df["is_canceled"].to_numpy()
    return {
        "bookings": len(df),
        "canceled": int(canceled.sum()),
        "cancel_rate": round(float(canceled.mean()), 4),
        "first_arrival": df["arrival_date"].min().date().isoformat(),
        "last_arrival": df["arrival_date"].max().date().isoformat(),
        "by_hotel": df["hotel"].value_counts().to_dict(),
    }
