# hotel-overbooking

[![CI](https://github.com/JihanS07/hotel-overbooking/actions/workflows/ci.yml/badge.svg)](https://github.com/JihanS07/hotel-overbooking/actions/workflows/ci.yml)

Hotels overbook because some guests cancel or never show up. Accept too few bookings and rooms sit
empty. Accept too many and a guest gets turned away at the desk. This project predicts which bookings
will cancel, uses those probabilities to decide how many extra bookings to accept for each night,
and writes a recommendation with the reasons behind it that a person has to approve.

Status: project setup and data download. The leak audit, models and decision step come next.

## Data

The hotel booking demand dataset from Antonio, de Almeida and Nunes, "Hotel booking demand
datasets", Data in Brief, 2019 ([doi](https://doi.org/10.1016/j.dib.2018.11.126)). It has 119,390
bookings for a resort hotel and a city hotel in Portugal, with arrivals from July 2015 to August
2017. The authors removed anything that identifies a guest or a hotel.

`make data` downloads the copy published by the TidyTuesday project and checks its SHA-256 hash. The
data is not stored in this repo.

License: not confirmed yet. Neither the paper's repository record nor the TidyTuesday copy states one
that I could find, so check the original article before reusing the data.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
make install
make data
uv run overbook info
make check
```
