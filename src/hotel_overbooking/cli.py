from pathlib import Path

import pandas as pd
import typer

from .data import DEFAULT_PATH, load_bookings, summary
from .decision import estimate_capacity, run_backtest
from .leaks import checkin_signals, outcome_leaks
from .model import DEFAULT_CUTOFF, evaluate

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


@app.command(help="Fit the cancellation models and print test scores.")
def evaluate_models(path: Path = DEFAULT_PATH, cutoff: str = DEFAULT_CUTOFF) -> None:
    result = evaluate(load_bookings(path), cutoff)
    typer.echo(f"train bookings: {result['train_bookings']}, test bookings: {result['test_bookings']}")
    typer.echo(f"cancel rate: train {result['train_cancel_rate']:.4f}, test {result['test_cancel_rate']:.4f}")
    typer.echo(pd.DataFrame(result["scores"]).T.round(4).to_string())
    for name, table in result["calibration"].items():  # type: ignore[union-attr]
        typer.echo(f"\ncalibration, {name}")
        typer.echo(table.round(3).to_string(index=False))


@app.command(help="Backtest accepting extra bookings against accepting none.")
def decide(path: Path = DEFAULT_PATH, cutoff: str = DEFAULT_CUTOFF) -> None:
    df = load_bookings(path)
    typer.echo(f"estimated capacity (busiest night): {estimate_capacity(df)}")
    typer.echo("\nonly bookings still active 30 days before arrival")
    typer.echo(run_backtest(df, cutoff, 30).round(2).to_string())
    typer.echo("\nevery booking request, in the order it was made")
    typer.echo(run_backtest(df, cutoff, None).round(2).to_string())
