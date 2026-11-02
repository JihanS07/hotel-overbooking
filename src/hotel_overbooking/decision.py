import numpy as np
import pandas as pd

from .model import fit_logistic, predict_cancel, split_by_arrival

DEFAULT_CUTOFF_DAYS = 30
MAX_EXTRA = 20
COST_RATIOS = [1, 2, 3, 5, 10]


def active_before_arrival(df: pd.DataFrame, days: int = DEFAULT_CUTOFF_DAYS) -> pd.DataFrame:
    """Bookings that exist and are not yet canceled `days` before arrival."""
    # reservation_status_date only builds this test setup, it is never a model input
    cutoff = df["arrival_date"] - pd.Timedelta(days=days)
    status_date = pd.to_datetime(df["reservation_status_date"])
    canceled_already = (df["is_canceled"] == 1) & (status_date <= cutoff)
    keep = (df["lead_time"] >= days) & ~canceled_already
    return df.loc[keep].copy()


def estimate_capacity(df: pd.DataFrame) -> dict[str, int]:
    """Busiest observed night of arriving guests who did not cancel, per hotel."""
    capacity = {}
    for hotel in sorted(set(df["hotel"].tolist())):
        kept = df[(df["hotel"] == hotel) & (df["is_canceled"] == 0)]
        capacity[str(hotel)] = int(pd.Series(kept["arrival_date"]).value_counts().max())
    return capacity


def show_distribution(show_probs: np.ndarray) -> np.ndarray:
    """Exact Poisson binomial distribution of the number of guests who show up."""
    pmf = np.array([1.0])
    for p in show_probs:
        pmf = np.append(pmf * (1 - p), 0.0) + np.append(0.0, pmf * p)
    return pmf


def expected_cost(pmf: np.ndarray, capacity: int, walk_cost: float, empty_cost: float = 1.0) -> float:
    shows = np.arange(len(pmf))
    walked = np.maximum(shows - capacity, 0)
    empty = np.maximum(capacity - shows, 0)
    return float(np.sum(pmf * (walk_cost * walked + empty_cost * empty)))


def choose_accepted(
    show_probs: np.ndarray, capacity: int, walk_cost: float, max_extra: int = MAX_EXTRA
) -> int:
    """How many of the bookings (in the order they were made) to accept."""
    limit = min(len(show_probs), capacity + max_extra)
    pmf = np.array([1.0])
    best_count = 0
    best_cost = float("inf")
    for count in range(limit + 1):
        if count > 0:
            p = show_probs[count - 1]
            pmf = np.append(pmf * (1 - p), 0.0) + np.append(0.0, pmf * p)
        if count < min(len(show_probs), capacity):
            continue
        cost = expected_cost(pmf, capacity, walk_cost)
        if cost < best_cost:
            best_cost = cost
            best_count = count
    return best_count


def walked_and_empty(shows: int, capacity: int) -> tuple[int, int]:
    return max(shows - capacity, 0), max(capacity - shows, 0)


def backtest(
    test: pd.DataFrame, show_prob: np.ndarray, capacity: dict[str, int], walk_cost: float
) -> pd.DataFrame:
    hotels = test["hotel"].to_numpy()
    nights = test["arrival_date"].to_numpy()
    booked_on = (test["arrival_date"] - pd.to_timedelta(test["lead_time"].to_numpy(), unit="D")).to_numpy()
    shows = (test["is_canceled"].to_numpy() == 0).astype(int)
    rows = []
    for hotel in sorted(set(hotels.tolist())):
        for night in np.unique(nights[hotels == hotel]):
            rows_for_night = np.flatnonzero((hotels == hotel) & (nights == night))
            order = rows_for_night[np.argsort(booked_on[rows_for_night], kind="stable")]
            rooms = capacity[str(hotel)]
            plain_count = min(len(order), rooms)
            extra_count = choose_accepted(show_prob[order], rooms, walk_cost)
            walked_plain, empty_plain = walked_and_empty(int(shows[order[:plain_count]].sum()), rooms)
            walked_extra, empty_extra = walked_and_empty(int(shows[order[:extra_count]].sum()), rooms)
            rows.append(
                {
                    "hotel": hotel,
                    "night": night,
                    "active_bookings": len(order),
                    "extra_accepted": extra_count - plain_count,
                    "walked_plain": walked_plain,
                    "empty_plain": empty_plain,
                    "walked_extra": walked_extra,
                    "empty_extra": empty_extra,
                }
            )
    return pd.DataFrame(rows)


def summarize(nights: pd.DataFrame, walk_cost: float) -> dict[str, float]:
    total = {
        name: float(nights[name].to_numpy().sum())
        for name in nights.columns
        if name.startswith(("walked", "empty", "extra"))
    }
    plain = walk_cost * total["walked_plain"] + total["empty_plain"]
    extra = walk_cost * total["walked_extra"] + total["empty_extra"]
    return {
        "nights_with_extra": float((nights["extra_accepted"].to_numpy() > 0).sum()),
        "extra_accepted": total["extra_accepted"],
        "walked_plain": total["walked_plain"],
        "walked_extra": total["walked_extra"],
        "empty_plain": total["empty_plain"],
        "empty_extra": total["empty_extra"],
        "cost_plain": plain,
        "cost_extra": extra,
        "cost_change_pct": 100 * (extra - plain) / plain if plain else 0.0,
    }


def run_backtest(
    df: pd.DataFrame, cutoff: str, active_days: int | None, ratios: list[int] = COST_RATIOS
) -> pd.DataFrame:
    """active_days=None uses every booking request, otherwise only bookings active that many days out."""
    capacity = estimate_capacity(df)
    train, test = split_by_arrival(df, cutoff)
    if active_days is not None:
        train = active_before_arrival(train, active_days)
        test = active_before_arrival(test, active_days)
    show_prob = 1.0 - predict_cancel(fit_logistic(train), test)
    rows = {}
    for ratio in ratios:
        rows[ratio] = summarize(backtest(test, show_prob, capacity, float(ratio)), float(ratio))
    return pd.DataFrame(rows).T.rename_axis("walk_cost_ratio")
