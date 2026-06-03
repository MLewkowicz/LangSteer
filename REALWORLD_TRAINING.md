# Real-world training pipeline (Franka + ZED) — end-to-end

Canonical recipe for porting fresh teleop data through preprocessing, bounds,
and training of a 3D Diffuser Actor on the real-world wine-glass / cabinet task.

This document is the source of truth for the v7-on-`data_v2` run. For the wider
context behind these choices (rate matching, sliding-window dataset, relative
bounds, mode collapse, low-t loss, steering) see `DIFFUSER_ACTOR_DEPLOY_DEBUG.md`
and the `CLAUDE.md` overview.

## 0. Branch + prerequisites

```bash
git checkout isaac           # this branch has the v6 recipe + session work
git status                   # confirm clean before starting
cd /home/clear/Documents/michal/LangSteer
```

All the steering / probe / autoprobe / low-t-loss / wide-bounds work lives on
`isaac`; `realworld/data-support` is an older sibling that does not.

## 1. Raw data layout

Raw data lives in `/media/clear/Backup/training_v2/` with three files per
replayed execution:

```
training_v2/
  <task>_<demo>_<execution>.h5                   # state stream @ ~200 Hz
  <task>_<demo>_<execution>_hand_video.hdf5      # wrist ZED rgb+depth+timestamps
  <task>_<demo>_<execution>_third_person_video.hdf5
```

`<task>` is currently `cabinet` or `wine_rack`. The directory also contains
*source-demo* h5s with single-number stems (e.g. `cabinet_0.h5`, no `_<execution>`)
that have no paired video files — those are the unreplayed originals and are
**auto-skipped** by `_discover_state_files`. As of writing the dir has 73
paired executions (cabinet + wine_rack) out of 112 total h5s.

State h5 carries `ee_pos`, `ee_rot`, `gripper_open`, `timestamps`,
`camera_timestamps/{hand,third_person}`. **Camera extrinsics are embedded as
ROOT attrs** (`extrinsics_hand` / `extrinsics_third_person` as JSON strings) —
the converter reads them automatically, no separate JSON files needed. File
naming is agnostic (legacy `episode_<ts>` and new `<task>_<demo>_<execution>`
both work).

## 2. The whole pipeline (overwrite & retrain)

Set up paths and the object name once:

```bash
cd /home/clear/Documents/michal/LangSteer
RAW=/media/clear/Backup/training_v2
PKG=/home/clear/Documents/michal/realworld_3da_v2
OBJECT=glass                       # pooled single-object label for ALL tasks
                                   # (cabinet + wine_rack live in one conditional)
```

**Conditioning split — what splits, what pools:**

| dimension | split? | why |
|---|---|---|
| **primitive** (`grasp` vs `place`) | **YES** — keep `num_primitives: 2`, `primitive_vocab: {grasp: 0, place: 1}` | grasp and place are different physical actions on different scene context. The bimodality lives only *within* place, not across grasp/place. |
| **object** (`cabinet`-task vs `wine_rack`-task) | **NO** — pool under one label, `num_objects: 1`, `object_vocab: {glass: 0}` | the upright-vs-inverted choice is the basin the steering module is built to select. Splitting it into separate object ids tokenizes the choice and destroys the in-model multimodality. |

So the converter emits 77 grasp + 77 place segments, all tagged `object: glass`.
See [`scripts/probe_yaw_diversity.py`](scripts/probe_yaw_diversity.py) and
[`steering/target_rotation.py`](steering/target_rotation.py) for the validation
and steering machinery this conditioning is designed for.

### 2a. Wipe the target so the convert is an overwrite

```bash
rm -rf "$PKG"
```

### 2b. Convert raw → packaged `.dat` shards

Auto-detects state rate per episode (~200 Hz → stride ~20 → effective 10 Hz),
splices each episode into `(grasp, place)` segments at the gripper-close /
gripper-open events (`detect_gripper_segments`), uses embedded extrinsics,
and routes the last `val_fraction` of episodes (alphabetically) to validation.

```bash
uv run python scripts/convert_realworld_for_diffuser_actor.py \
  --raw_dir "$RAW" --save_path "$PKG" \
  --auto_segment_object "$OBJECT" \
  --val_fraction 0.15
```

Outputs:

```
$PKG/training/D+0/ann_*.dat                                    # one per segment
$PKG/training/lang_annotations/primitive_object_lang_ann.npy   # primitive+object labels
$PKG/validation/D+0/ann_*.dat
$PKG/validation/lang_annotations/primitive_object_lang_ann.npy
```

Watch the converter logs for `[auto-stride] raising stride 2 -> ~20` lines —
that confirms it picked up the high state rate correctly.

### 2c. Compute relative bounds (action deltas excluded; v6 methodology)

```bash
uv run python scripts/compute_relative_gripper_loc_bounds.py \
  --dat_dir "$PKG" --nhist 3 --horizon_frames 10 --margin 0.10
```

The script prints a `gripper_loc_bounds:` YAML line. **Symmetrize** it (pad each
axis to its larger magnitude — never shrink, that clips real motion) and paste
the **identical** line into BOTH:

- `conf/training/diffuser_actor_realworld_primitive_object.yaml`
  (under `policy:`, replacing the existing `gripper_loc_bounds:`)
- `conf/policy/diffuser_actor_realworld_primitive_object.yaml`

The two must be byte-for-byte identical (training/deploy normalization must match).

### 2d. Update vocab and paths in the training yaml

Edit `conf/training/diffuser_actor_realworld_primitive_object.yaml`:

```yaml
policy:
  object_vocab:
    glass: 0             # single pooled label — see §1 "no object-token splits"
  num_objects: 1         # bump if more than one object class
dataset:
  train_path: "/home/clear/Documents/michal/realworld_3da_v2/training"
  val_path:   "/home/clear/Documents/michal/realworld_3da_v2/validation"
  primitive_ann_path_train: "/home/clear/Documents/michal/realworld_3da_v2/training/lang_annotations/primitive_object_lang_ann.npy"
  primitive_ann_path_val:   "/home/clear/Documents/michal/realworld_3da_v2/validation/lang_annotations/primitive_object_lang_ann.npy"
```

`primitive_vocab: {grasp: 0, place: 1}` stays as-is.

### 2e. Train

100-step retrain, overnight (~10-12 h on a 5090), v7 output dir, EMA off, low-t
loss on by default in the yaml:

```bash
nohup uv run python scripts/train_diffuser_actor.py \
  training=diffuser_actor_realworld_primitive_object \
  training.policy.diffusion_timesteps=100 \
  training.train_iters=200000 \
  training.use_ema=false \
  training.checkpoint_dir=outputs/checkpoints/diffuser_actor_realworld_primitive_object_v7 \
  > outputs/v7_train.log 2>&1 &
echo "PID=$! → tail -f outputs/v7_train.log"
```

For a 25-step retrain instead, drop `training.policy.diffusion_timesteps=100`
and set `training.train_iters=80000` (the v6 baseline).

## 3. Train without validation

The val set on a 3-episode dataset is already at the floor (one episode → two
segments). To **skip val entirely during training** but keep the dataset built
(so the trainer doesn't error on missing files), pass:

```bash
training.val_freq=999999
```

To **collect fewer episodes into val** at convert time, lower `--val_fraction`
(it floors at 1 episode regardless of fraction). To eliminate val episodes
entirely point `val_path` at the train data — the trainer loads but never
queries when `val_freq` is huge.

## 4. Coupling invariants — get all four to agree

Diverge on any one and the model trains at one scale and unnormalizes at another.

1. **Data rate ≈ 10 Hz** — converter auto-stride targets `TARGET_EFFECTIVE_RATE_HZ=10`.
2. **`dataset.horizon_frames: 10`** (training yaml) — window length per plan.
3. **`--horizon_frames 10`** (bounds script) — bounds computed over the same windows.
4. **Same `gripper_loc_bounds` in both training and deploy yamls.**

`horizon_frames` is purely a data-side parameter. It lives in
`RealworldSlidingWindowDataset` and in the bounds script; once the `.dat` files
are built and the bounds are computed, neither the model nor deploy ever reads
it again.

## 5. Monitoring training

Live progress:

```bash
tail -f outputs/v7_train.log              # wallclock + losses
# wandb run appears under project "langsteer_diffuser_actor"
```

Mode-B (wrist-inversion) initiation curve, on CPU so it doesn't fight the GPU:

```bash
uv run python scripts/autoprobe_checkpoints.py \
  --ckpt_dir outputs/checkpoints/diffuser_actor_realworld_primitive_object_v7 \
  --device cpu --diffusion_timesteps 100 --interval 600
```

Headline metric = frame-0 initiation fraction; target ~43 % (the data prior).

## 6. Deploy after training

In `franka-teleop/conf/deploy_policy.yaml` (and any policy yaml deploy reads):

- Flip `diffusion_timesteps: 100` to match the trained schedule.
- Paste the same symmetric bounds as the training yaml.
- Update the `stages` list in `deploy_diffuser_actor.py` if primitive/object IDs changed.
- For rotational steering toward the inverted place, see
  `conf/steering/target_rotation.yaml` and the deploy-side wiring in
  `franka-teleop/deploy_diffuser_actor.py`.

## 7. Common failure modes

- **`KeyError` on an object label in the trainer** → mismatch between the
  `--auto_segment_object` value used at convert time and `policy.object_vocab`
  in the yaml. Both should be `glass` (single pooled label).
- **Bounds wildly different from v6's `[[-0.4056…], [0.5103…]]`** → you ran the
  bounds script with `--include_action_deltas` (default off). Re-run without it.
- **`StopIteration` from the train loader on a small dataset** → `train_iters`
  is so low that `drop_last=True` empties the loader. Either raise
  `train_iters` or lower `batch_size`.
- **Violent / OOD motion at deploy** → `diffusion_timesteps` differs between
  training and deploy. Same value in both yamls.

## 8. What's on disk after a successful run

```
outputs/checkpoints/diffuser_actor_realworld_primitive_object_v7/
  0000499.pth … 0199999.pth      # periodic
  last.pth
  best.pth                        # (val) — only meaningful if val_freq < train_iters
  autoprobe_yaw.log               # if you ran the autoprobe
```

Use `last.pth` for deploy unless `best.pth` was selected on a metric you trust.
