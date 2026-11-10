"""CLI for post-CIRI quadrotor RL and radar-depth YOPO."""

from pathlib import Path
from typing import Annotated

import typer

from tooling.learn.pipeline import evaluate, smoke as run_smoke, train_rl, train_yopo
from tooling.learn.scenes import load_post_ciri_scene

app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Train a quadrotor RL policy and a radar-depth YOPO baseline on a "
        "post-CIRI occupancy scene, then compare them."
    ),
)


def _scene(path: Path):
    if not path.is_absolute():
        typer.echo("scene directory must be absolute", err=True)
        raise typer.Exit(2)
    try:
        return load_post_ciri_scene(path)
    except (OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error


@app.command("train-rl")
def train_rl_command(
    scene: Annotated[Path, typer.Option("--scene", help="Absolute post-CIRI scene directory")],
    updates: Annotated[int, typer.Option(min=1)] = 20,
    seed: int = 0,
) -> None:
    """Train the on-policy quadrotor policy. Weights go under .artifacts/learn/."""
    loaded = _scene(scene)
    path = train_rl(loaded, updates, seed)
    typer.echo(path)


@app.command("train-yopo")
def train_yopo_command(
    scene: Annotated[Path, typer.Option("--scene", help="Absolute post-CIRI scene directory")],
    updates: Annotated[int, typer.Option(min=1)] = 20,
    seed: int = 0,
) -> None:
    """Train YOPO on depth rasterized from the forward radar point cloud."""
    loaded = _scene(scene)
    path = train_yopo(loaded, updates, seed)
    typer.echo(path)


@app.command("eval")
def eval_command(
    scene: Annotated[Path, typer.Option("--scene", help="Absolute post-CIRI scene directory")],
    episodes: Annotated[int, typer.Option(min=1)] = 5,
    seed: int = 0,
) -> None:
    """Roll out the saved RL and YOPO weights and write comparison.md."""
    loaded = _scene(scene)
    try:
        directory = evaluate(loaded, episodes, seed)
    except FileNotFoundError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    typer.echo(directory / "comparison.md")


@app.command("smoke")
def smoke_command(seed: int = 0) -> None:
    """Exercise RL and YOPO on a synthetic room. The report is not a CIRI result."""
    directory = run_smoke(seed)
    typer.echo(directory / "comparison.md")
