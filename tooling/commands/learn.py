"""Radar-to-depth collection and YOPO training.

CIRI is not part of this command group. ``train-rl`` stops because that policy
was supposed to read CIRI, and the CIRI interface has not been specified.
"""

from pathlib import Path
from typing import Annotated

import typer

from tooling.learn.ciri import CiriUnavailable
from tooling.learn.dataset import collect_dataset, pack_dataset
from tooling.learn.pipeline import evaluate, smoke as run_smoke, train_rl, train_yopo

app = typer.Typer(
    no_args_is_help=True,
    help="Collect radar depth, pack a train/val set, and train YOPO. CIRI is unfinished.",
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
    """Simulate radar at random poses and store each cloud with its depth raster."""
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
    """Train YOPO on stored radar-depth views. Depth is not rendered again."""
    path = train_yopo(_dataset(dataset), updates, seed)
    typer.echo(path)


@app.command("train-rl")
def train_rl_command(
    dataset: Annotated[Path, typer.Option("--dataset", help="Absolute dataset directory")],
    updates: Annotated[int, typer.Option(min=1)] = 20,
    seed: int = 0,
) -> None:
    """Refuse to train RL until a CIRI observation interface is specified."""
    try:
        train_rl(_dataset(dataset), updates, seed)
    except CiriUnavailable as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error


@app.command("eval")
def eval_command(
    dataset: Annotated[Path, typer.Option("--dataset", help="Absolute dataset directory")],
    seed: int = 0,
) -> None:
    """Score the saved YOPO weights on the validation split."""
    try:
        directory = evaluate(_dataset(dataset), seed)
    except (OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    typer.echo(directory / "comparison.md")


@app.command("smoke")
def smoke_command(seed: int = 0) -> None:
    """Collect, pack, and train YOPO on a tiny synthetic radar set."""
    directory = run_smoke(seed)
    typer.echo(directory / "comparison.md")
