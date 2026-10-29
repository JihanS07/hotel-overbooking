from pathlib import Path

import typer

from .data import DEFAULT_PATH, load_bookings, summary
from .leaks import checkin_signals, outcome_leaks

app = typer.Typer(help="Hotel cancellation forecasting and overbooking decisions.", no_args_is_help=True)


@app.command(help="Print basic counts for the bookings file.")
def info(path: Path = DEFAULT_PATH) -> None:
    for key, value in summary(load_bookings(path)).items():
        typer.echo(f"{key}: {value}")


@app.command(help="Print the leak audit numbers.")
def audit(path: Path = DEFAULT_PATH) -> None:
    df = load_bookings(path)
    for key, value in outcome_leaks(df).items():
        typer.echo(f"{key}: {value:.4f}")
    typer.echo(checkin_signals(df).round(4).to_string())


@app.command(help="Show the installed version.")
def version() -> None:
    typer.echo("0.1.0")
