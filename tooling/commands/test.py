"""ROS tests run in Docker; uv only dispatches the commands."""

import typer

from tooling import container_tools

app = typer.Typer()
ROOT = container_tools.CONTAINER_ROOT


def _pytest(path: str, *, arch: str = "native") -> None:
    outputs = ROOT / ".artifacts/tests"
    container_tools.run(
        [
            "env",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
            "python3",
            "-m",
            "pytest",
            str(ROOT / path),
            "--basetemp",
            str(outputs / "pytest_tmp"),
            "-o",
            f"cache_dir={outputs / 'pytest_cache'}",
        ],
        arch=arch,
    )


@app.command()
def pytest(path: str = "test/cli", arch: str = "native") -> None:
    """Run one pytest path inside the Linux tools container."""
    _pytest(path, arch=arch)


@app.command(name="colcon")
def colcon(arch: str = "native") -> None:
    """Run tests in the architecture-specific Docker ROS workspace."""
    base = ROOT / ".artifacts/colcon"
    container_tools.run(
        ["colcon", "--log-base", str(base / "log"), "test", "--merge-install",
         "--build-base", str(base / "build"), "--install-base", str(base / "install")],
        arch=arch,
    )


@app.command(name="colcon-result")
def colcon_result(arch: str = "native") -> None:
    """Read test results from the architecture-specific Docker workspace."""
    base = ROOT / ".artifacts/colcon"
    container_tools.run(
        ["colcon", "--log-base", str(base / "log"), "test-result",
         "--test-result-base", str(base / "build")],
        arch=arch,
    )
