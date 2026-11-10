# Synthetic radar: CIRI for RL, eroded depth for YOPO

The quadrotor plant has no wing-force model. Learning starts from synthetic
radar sweeps. One accumulated cloud feeds two pipelines. The manifest records
`synthetic: true`.

```text
radar sweeps
    └─ accumulate in the world frame
         ├─ CIRI  →  h-polytope corridor  →  RL
         └─ erosion  →  projection  →  depth  →  YOPO
```

## CIRI

CIRI is Configuration-space Iterative Regional Inflation from SUPER
(hku-mars/SUPER, Ren et al., Science Robotics 2025). `tooling/learn/ciri.py`
follows that call: obstacle points, a boundary box, and a seed segment in;
halfspace rows `[nx, ny, nz, d]` out, with the free set `n·x + d <= 0`.
Obstacles are inflated by the robot radius, so the polytope is a corridor in
configuration space. This is the observation the RL policy reads. It is not an
occupancy cloud and it is not a depth image.

## Depth

Depth is not a raster of one sparse scan. `tooling/learn/radar_depth.py`:

1. Accumulate every sweep so far into one cloud.
2. Erode it: an occupied voxel is dropped unless a face-adjacent voxel is also
   occupied, which removes isolated returns and keeps an accumulated surface.
3. Project the survivors into a 32×24 forward depth image (90° by 40°). The
   nearest forward range wins. Empty pixels stay at maximum range.

## What is stored

```text
$AP_PNC_DIR/.artifacts/learn/datasets/<name>/
  manifest.json          schema ap-pnc/radar-accumulated/v1, synthetic: true
  raw/<env>/label.npz    depth, CIRI planes, accumulated points, poses
  raw/<env>/environment.npy
  packed/train.npz
  packed/val.npz
  yopo.npz               after train-yopo
  rl.npz                 after train-rl
  comparison.md          after eval
```

## Commands

```bash
export AP_PNC_DIR=/absolute/path/to/AP-PnC
uv sync --project "$AP_PNC_DIR"

uv run --project "$AP_PNC_DIR" ap-pnc learn collect \
  --name synthetic-radar --envs 2 --samples 16 --seed 0

DATA="$AP_PNC_DIR/.artifacts/learn/datasets/synthetic-radar"
uv run --project "$AP_PNC_DIR" ap-pnc learn pack --dataset "$DATA" --seed 0
uv run --project "$AP_PNC_DIR" ap-pnc learn train-rl --dataset "$DATA" --updates 20
uv run --project "$AP_PNC_DIR" ap-pnc learn train-yopo --dataset "$DATA" --updates 20
uv run --project "$AP_PNC_DIR" ap-pnc learn eval --dataset "$DATA"
```

`collect` and `pack` run once. Training reads the stored planes and the stored
depth; it does not rebuild them. Each loaded item samples a body velocity,
acceleration, and forward goal direction.

`ap-pnc learn smoke` runs the chain on four synthetic views with one update
each.

Author YOPO weights trained on a depth camera are not loaded.
