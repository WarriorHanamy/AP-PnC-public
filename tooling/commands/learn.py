"""Collect accumulated radar views, train CIRI-conditioned RL, and train YOPO."""

from pathlib import Path
from typing import Annotated

import typer

from tooling.learn.dataset import collect_dataset, pack_dataset
from tooling.learn.pipeline import evaluate, smoke as run_smoke, train_rl, train_yopo

app = typer.Typer(
    no_args_is_help=True,
    help="Radar accumulation: CIRI corridors for RL, eroded depth for YOPO.",
)


def _dataset(path: Path) -> Path:
    if not path.is_absolute():
        typer.echo("dataset directory must be absolute", err=True)
        raise typer.Exit(2)
    return path


@app.command()
def collect(
    name: Annotated[
        str, typer.Option(help="Dataset folder name under .artifacts/learn/datasets")
    ] = "synthetic-radar",
    envs: Annotated[int, typer.Option(min=1)] = 2,
    samples: Annotated[int, typer.Option(min=2)] = 16,
    seed: int = 0,
) -> None:
    """Simulate synthetic radar sweeps and store CIRI planes and eroded depth."""
    path = collect_dataset(name, envs=envs, samples_per_env=samples, seed=seed)
    typer.echo(path)


@app.command()
def pack(
    dataset: Annotated[Path, typer.Option("--dataset", help="Absolute dataset directory")],
    seed: int = 0,
) -> None:
    """Shuffle recorded views into packed/train.npz and packed/val.npz."""
    path = pack_dataset(_dataset(dataset), seed=seed)
    typer.echo(path)


@app.command("train-yopo")
def train_yopo_command(
    dataset: Annotated[Path, typer.Option("--dataset", help="Absolute packed dataset directory")],
    updates: Annotated[int, typer.Option(min=1)] = 20,
    seed: int = 0,
) -> None:
    """Train YOPO on depth made by accumulating, eroding, and projecting radar points."""
    path = train_yopo(_dataset(dataset), updates, seed)
    typer.echo(path)


@app.command("train-rl")
def train_rl_command(
    dataset: Annotated[Path, typer.Option("--dataset", help="Absolute dataset directory")],
    updates: Annotated[int, typer.Option(min=1)] = 20,
    seed: int = 0,
) -> None:
    """Train the quadrotor policy on stored SUPER CIRI corridors."""
    try:
        path = train_rl(_dataset(dataset), updates, seed)
    except (OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    typer.echo(path)


@app.command("eval")
def eval_command(
    dataset: Annotated[Path, typer.Option("--dataset", help="Absolute dataset directory")],
    seed: int = 0,
) -> None:
    """Score YOPO and the CIRI policy on the validation split."""
    try:
        directory = evaluate(_dataset(dataset), seed)
    except (OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    typer.echo(directory / "comparison.md")


@app.command("smoke")
def smoke_command(seed: int = 0) -> None:
    """Collect, pack, and take one update of RL and YOPO on synthetic radar."""
    directory = run_smoke(seed)
    typer.echo(directory / "comparison.md")
