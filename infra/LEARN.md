# Radar-depth data for YOPO

The quadrotor plant has no wing-force model. This page is the recorded
radar-to-depth path used to train YOPO. It follows the same stages as the
public YOPO collector (`data_collection_simulation.py`): sample poses, save the
sensor image and the pose, then train from that recording with velocity and
goal sampled when each item is loaded.

CIRI is **not** part of this pipeline. Its representation has not been
specified, so nothing here is a CIRI tensor, a CIRI scene, or a CIRI-to-depth
conversion. `ap-pnc learn train-rl` stops for that reason.

## What is stored

Each sample is one forward radar sweep and the depth image made directly from
those points:

- sparse body-frame radar points (16 by 4 beams, 90° by 40°)
- a 32×24 depth raster (forward range in metres; empty pixels stay at max range)
- the pose of the sweep
- the synthetic obstacle points that cast the rays, kept only as the cost map
  for YOPO's guidance step (the same role as YOPO's saved map)

The obstacle field is a synthetic simulator world. It is not a reconstruction
and it is not CIRI.

```text
$AP_PNC_DIR/.artifacts/learn/datasets/<name>/
  manifest.json          schema ap-pnc/radar-depth/v1, ciri: unspecified
  raw/<env>/label.npz    depth, valid mask, poses, radar points
  raw/<env>/environment.npy
  packed/train.npz
  packed/val.npz
  yopo.npz               after training
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
uv run --project "$AP_PNC_DIR" ap-pnc learn train-yopo --dataset "$DATA" --updates 20
uv run --project "$AP_PNC_DIR" ap-pnc learn eval --dataset "$DATA"
```

`collect` and `pack` only need to be run once for a dataset. Training reads the
packed depth; it does not raycast again. At each training item the loader
samples a body velocity, acceleration, and forward goal direction, which is the
augmentation YOPO applies in `YopoDataset.__getitem__`.

`ap-pnc learn smoke` runs collect, pack, one YOPO update, and eval on four
synthetic views.

`ap-pnc learn train-rl` exits with an error. RL was going to consume CIRI, and
that interface is not defined.

Author YOPO weights trained on a depth camera are not loaded.
