from pathlib import Path

from hotel_overbooking.data import load_bookings, summary

HEADER = "hotel,is_canceled,lead_time,arrival_date_year,arrival_date_month,arrival_date_day_of_month"
ROWS = [
    "Resort Hotel,0,10,2015,July,1",
    "Resort Hotel,1,200,2015,July,3",
    "City Hotel,1,45,2017,August,31",
    "City Hotel,0,5,2016,February,29",
]


def write_sample(tmp_path: Path) -> Path:
    path = tmp_path / "hotels.csv"
    path.write_text("\n".join([HEADER, *ROWS]) + "\n")
    return path


def test_arrival_date_is_built_from_the_three_columns(tmp_path: Path):
    df = load_bookings(write_sample(tmp_path))
    assert df["arrival_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2015-07-01",
        "2015-07-03",
        "2017-08-31",
        "2016-02-29",
    ]


def test_summary_counts(tmp_path: Path):
    s = summary(load_bookings(write_sample(tmp_path)))
    assert s["bookings"] == 4
    assert s["canceled"] == 2
    assert s["cancel_rate"] == 0.5
    assert s["first_arrival"] == "2015-07-01"
    assert s["last_arrival"] == "2017-08-31"
    assert s["by_hotel"] == {"Resort Hotel": 2, "City Hotel": 2}
