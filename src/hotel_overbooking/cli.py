from pathlib import Path

import typer

from .data import DEFAULT_PATH, load_bookings, summary

app = typer.Typer(help="Hotel cancellation forecasting and overbooking decisions.", no_args_is_help=True)


@app.command(help="Print basic counts for the bookings file.")
def info(path: Path = DEFAULT_PATH) -> None:
    for key, value in summary(load_bookings(path)).items():
        typer.echo(f"{key}: {value}")


@app.command(help="Show the installed version.")
def version() -> None:
    typer.echo("0.1.0")
