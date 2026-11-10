"""Quadrotor stack sweep: SE3 geometric control versus NMPC.

Runs speed caps, actuator profiles and the two controllers through
:mod:`tooling.simple_stack` and prints one table. Controller gains come from
``core/bringup/config/simple_sim.yaml``. The plant is a quadrotor: there is no
aerodynamic-model axis and no wing-force feedforward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tooling import simple_stack
from tooling.env import artifact_path

VARIANTS = ("se3", "nmpc")


@dataclass(frozen=True)
class Row:
    profile: str
    v_max: float
    variant: str
    status: str
    steps: int
    position_rmse: float
    position_max: float
    velocity_rmse: float
    max_sideslip_deg: float
    run_dir: Path

    def as_dict(self) -> dict:
        return {
            "v_max": self.v_max,
            "variant": self.variant,
            "status": self.status,
            "steps": self.steps,
            "position_rmse_m": self.position_rmse,
            "position_max_m": self.position_max,
            "velocity_rmse_mps": self.velocity_rmse,
            "max_sideslip_deg": self.max_sideslip_deg,
            "run_dir": str(self.run_dir),
        }


def _variant_overrides(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    return {"simple_sim.controller": variant}


def _run_once(
    *,
    v_max: float,
    duration_s: float,
    profile: str,
    hold_s: float,
    mass_kg: float,
    overrides: dict,
    log_dir: Path,
    tag: str,
) -> Path:
    job = simple_stack.run_job(
        simple_stack.JobRequest(
            profile=profile,
            v_max=v_max,
            hold_s=hold_s,
            mass_kg=mass_kg,
            duration_s=duration_s,
            overrides=overrides,
        )
    )
    (log_dir / f"{tag}.log").write_text(str(job / "stack.log"))
    (log_dir / f"{tag}.job").write_text(str(job))
    return job


def run_matrix(
    *,
    v_maxs: tuple[float, ...] = (8.0, 10.0, 12.0),
    variants: tuple[str, ...] = VARIANTS,
    duration_s: float = 0.0,
    profiles: tuple[str, ...] = ("ideal",),
    hold_s: float = 2.0,
    mass_kg: float = 2.0,
    output_dir: Path | None = None,
) -> list[Row]:
    """Run the quadrotor matrix and return one Row per run."""
    out = output_dir or artifact_path(".artifacts/benchmark/stack-sweep")
    out.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    for variant in variants:
        overrides = _variant_overrides(variant)
        for profile in profiles:
            for v_max in v_maxs:
                tag = f"{profile}-v{v_max:g}-{variant}"
                try:
                    job = _run_once(
                        v_max=v_max,
                        duration_s=duration_s,
                        profile=profile,
                        hold_s=hold_s,
                        mass_kg=mass_kg,
                        overrides=overrides,
                        log_dir=out,
                        tag=tag,
                    )
                    summary = simple_stack.run_summary(job)
                    run_dir = summary["run_dir"]
                    manifest = summary["manifest"]
                    metrics = summary["metrics"]
                    row = Row(
                        profile=profile,
                        v_max=v_max,
                        variant=variant,
                        status=manifest["status"],
                        steps=int(manifest["committed_steps"]),
                        position_rmse=float(metrics["position_rmse_m"]),
                        position_max=float(metrics["position_max_m"]),
                        velocity_rmse=float(metrics["velocity_rmse_mps"]),
                        max_sideslip_deg=float(metrics.get("max_sideslip_deg", float("nan"))),
                        run_dir=run_dir,
                    )
                except Exception as error:  # - keep sweeping
                    print(f"{tag}: {error}")
                    row = Row(
                        profile=profile,
                        v_max=v_max,
                        variant=variant,
                        status="error",
                        steps=0,
                        position_rmse=float("nan"),
                        position_max=float("nan"),
                        velocity_rmse=float("nan"),
                        max_sideslip_deg=float("nan"),
                        run_dir=out,
                    )
                rows.append(row)
                print(
                    f"{profile:9s} v{v_max:<3g} {variant:6s} {row.status:9s} {row.position_rmse:.4f}",
                    flush=True,
                )

    (out / "sweep.json").write_text(json.dumps([r.as_dict() for r in rows], indent=2))
    (out / "table.md").write_text(render_table(rows))
    return rows


def _cell(row: Row | None) -> str:
    if row is None:
        return "-"
    if row.status == "error" or row.position_rmse != row.position_rmse:  # NaN
        return "error"
    mark = "" if row.status == "finished" else " (failed)"
    return f"{row.position_rmse:.4f}{mark}"


def render_table(rows: list[Row]) -> str:
    """Markdown tables: one per profile, one row per speed cap."""
    present = [v for v in VARIANTS if any(r.variant == v for r in rows)]
    lines: list[str] = []
    for profile in sorted({r.profile for r in rows}):
        selected = [r for r in rows if r.profile == profile]
        lines.append(f"### actuator profile: {profile}\n")
        lines.append("| v_max | " + " | ".join(f"{v} posRMSE" for v in present) + " |")
        lines.append("|---:|" + "---:|" * len(present))
        for v_max in sorted({r.v_max for r in selected}):
            picked = {r.variant: r for r in selected if r.v_max == v_max}
            cells = [_cell(picked.get(v)) for v in present]
            lines.append(f"| {v_max:g} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
