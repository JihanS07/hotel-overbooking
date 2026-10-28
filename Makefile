.PHONY: install data check test

install:            ## install everything, including dev tools and git hooks
	uv sync --all-extras
	uv run pre-commit install

data:               ## download the hotel bookings CSV and check its hash
	bash scripts/download_data.sh

check:              ## what CI runs
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pytest

test:
	uv run pytest
