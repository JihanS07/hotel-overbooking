# hotel-overbooking

[![CI](https://github.com/JihanS07/hotel-overbooking/actions/workflows/ci.yml/badge.svg)](https://github.com/JihanS07/hotel-overbooking/actions/workflows/ci.yml)

Hotels overbook because some guests cancel or never show up, accept too few bookings and rooms sit
empty or accept too many and a guest gets turned away at the desk. This project predicts which bookings
will cancel, uses those probabilities to decide how many extra bookings to accept for each night,
and writes a recommendation with the reasons behind it that a person has to approve.

Status: project setup, data download, leak audit, cancellation models and a logistic regression written from scratch. The decision step comes next.

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

## Cancellation model

`uv run overbook evaluate-models` trains on bookings that arrive before 2017-03-01 (86,561 bookings,
36% canceled) and tests on the later ones (32,829 bookings, 40% canceled). Splitting by arrival date
means the model never sees the future. The inputs are the numeric columns and seven categorical
columns listed in `src/hotel_overbooking/model.py`. Country, agent and company are left out for now.

| Model | AUC | Brier score |
| --- | --- | --- |
| Base rate (train cancel rate for everyone) | 0.500 | 0.2417 |
| Logistic regression | 0.804 | 0.1680 |
| Gradient boosting | 0.815 | 0.1697 |

Gradient boosting ranks bookings slightly better, but logistic regression has the better Brier score,
so its probabilities are closer to the true rates. The calibration tables are printed by the command.
Both models under-predict cancellations in the middle bins, because the test months cancel more than
the training months. The top bin holds about 3,400 bookings predicted at 99% that cancel 99.9% of the
time. These are non-refundable bookings: 99.4% of them are canceled in this dataset. That pattern is
odd for a real hotel and is worth knowing before trusting the result.

## Logistic regression from scratch

`src/hotel_overbooking/scratch.py` fits the same logistic regression with plain NumPy. It uses
full-batch gradient descent on the mean log loss plus an L2 penalty, on the same scaled and one-hot
features as the scikit-learn model. The penalty is set so it matches scikit-learn's `C=1.0`, and the
intercept is not penalized.

`tests/test_scratch.py` checks that the coefficients, the intercept and the predicted probabilities
agree with scikit-learn within 1e-3 on a small fake dataset. On a random sample of 3,000 real training
bookings (30,000 steps) I measured the largest gap: 0.005 for a coefficient, 0.021 for the intercept and
0.0005 for a predicted probability. The one-hot columns add up to one, which makes the loss almost flat
in one direction, so gradient descent settles slower there than scikit-learn's solver. The predictions
still agree.

What it shows: the model is not a black box. Each prediction is a sum of weights times scaled feature
values passed through a sigmoid. The recommendation step will use these weights to explain each booking.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
make install
make data
uv run overbook info
uv run overbook audit
uv run overbook evaluate-models
make check
```
