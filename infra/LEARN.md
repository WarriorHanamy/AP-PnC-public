# Post-CIRI learning: quadrotor RL and radar-depth YOPO

The vehicle is the quadrotor plant (thrust along body +Z). This page is the
training and comparison entry point. It does not restore wing-force models.

YOPO's input is a depth image **rasterized from a sparse forward radar point
cloud**. A depth camera is not the baseline. The published YOPO weights, which
were trained on depth-camera images, are not used.

## Where post-CIRI scenes come from

This checkout does not contain CIRI reconstructions. A search of the tracked
tree for `CIRI` / `ciri` finds no scene files, configs, or loaders. The shipped
perception scenes are different:

- `infra/sim_infra/lidar_sim` pillars, and an optional XYZ PLY for the lidar node
- the MatrixCity / CityGaussian road used by the 3DGS demo
  ([start guide](CITY_DEMO_START.md))

Those are not post-CIRI scenes. Do not rename them to CIRI.

Prepare one directory per reconstruction, anywhere you keep data, and pass its
**absolute** path:

```text
<scene-dir>/scene.yaml
<scene-dir>/occupancy.ply
```

`scene.yaml`:

```yaml
source: post-ciri
name: block-a
occupancy: occupancy.ply
voxel_m: 0.4
ground_z: 0.0
start: [0.0, 0.0, 1.5]
goal: [12.0, 0.0, 1.5]
```

`source` must be the string `post-ciri`. `occupancy.ply` is an ENU XYZ cloud in
metres produced by your CIRI pipeline (ASCII or binary little-endian). The
occupancy path has to stay inside the scene directory. `start` must not lie in
occupied space.

If that directory is missing, `train-rl`, `train-yopo`, and `eval` stop. They
will not synthesize a CIRI scene.

## Commands

From the checkout, with `AP_PNC_DIR` set to the absolute root:

```bash
export AP_PNC_DIR=/absolute/path/to/AP-PnC
uv sync --project "$AP_PNC_DIR"

SCENE=/absolute/path/to/block-a
uv run --project "$AP_PNC_DIR" ap-pnc learn train-rl --scene "$SCENE" --updates 20
uv run --project "$AP_PNC_DIR" ap-pnc learn train-yopo --scene "$SCENE" --updates 20
uv run --project "$AP_PNC_DIR" ap-pnc learn eval --scene "$SCENE" --episodes 5
```

Weights and the comparison are written under
`$AP_PNC_DIR/.artifacts/learn/<scene name>/`:

- `rl.npz` — on-policy quadrotor actor-critic
- `yopo.npz` — radar-depth YOPO
- `comparison.md` and `comparison.json`

Both policies observe the same radar raster (32×24 forward range, 90°×40°).
The RL action is collective specific thrust and body rates on a reduced
quadrotor (body +Z thrust). YOPO predicts a scored set of quintic primitives
and the quadrotor tracks the selected primitive. This learning quadrotor is
not the acados NMPC; regenerate that solver separately when you need it.

`ap-pnc learn smoke` runs a few updates on a synthetic corridor and writes
`$AP_PNC_DIR/.artifacts/learn/smoke-<seed>/`. The report says
`scene_source: synthetic-smoke` and is not a CIRI result. Use it to check that
the install can train and compare. Use `train-rl`, `train-yopo`, and `eval`
for a real post-CIRI scene.

## What is not included

- CIRI point clouds, labels, or a downloader
- The authors' YOPO ResNet weights (they expect a depth camera)
- A claim that a short numpy training run matches a paper result
