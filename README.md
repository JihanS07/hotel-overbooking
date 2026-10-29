# hotel-overbooking

[![CI](https://github.com/JihanS07/hotel-overbooking/actions/workflows/ci.yml/badge.svg)](https://github.com/JihanS07/hotel-overbooking/actions/workflows/ci.yml)

Hotels overbook because some guests cancel or never show up, accept too few bookings and rooms sit
empty or accept too many and a guest gets turned away at the desk. This project predicts which bookings
will cancel, uses those probabilities to decide how many extra bookings to accept for each night,
and writes a recommendation with the reasons behind it that a person has to approve.

Status: project setup, data download and leak audit. The models and decision step come next.

## Data

The hotel booking demand dataset from Antonio, de Almeida and Nunes, "Hotel booking demand
datasets", Data in Brief, 2019 ([doi](https://doi.org/10.1016/j.dib.2018.11.126)). It has 119,390
bookings for a resort hotel and a city hotel in Portugal, with arrivals from July 2015 to August
2017. The authors removed anything that identifies a guest or a hotel.

`make data` downloads the copy published by the TidyTuesday project and checks its SHA-256 hash. The
data is not stored in this repo.

License: not confirmed yet. Neither the paper's repository record nor the TidyTuesday copy states one
that I could find, so check the original article before reusing the data.

## Leak audit

A cancellation model must only use what is known when the forecast is made. `uv run overbook audit`
prints the numbers below. The excluded columns live in one list, `EXCLUDED_COLUMNS` in
`src/hotel_overbooking/leaks.py`.

Excluded because they encode the outcome:

- `reservation_status` matches the label exactly. All 44,224 canceled bookings are Canceled or
  No-Show, and all 75,166 kept bookings are Check-Out.
- `reservation_status_date` is the day the status last changed. For kept bookings it is never before
  the arrival date. For 95.3% of canceled bookings it is before arrival, so the date alone separates
  the two groups.

Excluded because they are recorded at or after check-in:

- `assigned_room_type` differs from the reserved room for 18.8% of kept bookings and 1.8% of canceled
  ones. A room can only be assigned to a guest who shows up.
- `required_car_parking_spaces` is above 0 for 9.9% of kept bookings and for none of the canceled
  ones. A canceled guest never asks for parking.
- `booking_changes` is above 0 for 20.3% of kept bookings and 6.4% of canceled ones. The count keeps
  growing until arrival or cancellation, so the final value is not known at forecast time.

Checked and kept for now:

- `total_of_special_requests` is above 0 for 51.1% of kept and 24.1% of canceled bookings. Requests
  can be added late, so this is a weaker signal than it looks. It is worth testing the model without it.
- `adr` is 0 for 2.3% of kept and 0.5% of canceled bookings. The means are close (100.0 and 105.0),
  so it does not look recorded at check-in.
- `country` is missing for 0.6% of kept and 0.2% of canceled bookings. The cancel rate is 57% for
  Portugal and 17% to 25% for the next four biggest countries, which looks like real behavior.

These checks show patterns, not proof. Nothing in the data says when each field was written.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
make install
make data
uv run overbook info
uv run overbook audit
make check
```
