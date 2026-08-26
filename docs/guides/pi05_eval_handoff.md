# pi0.5 CALVIN Evaluation — Workstation Handoff & Preflight

**Branch:** `feature/vla_finetuning`  ·  **Audience:** an agent on the GPU workstation
that will run the eval.  **Goal of this doc:** (1) explain what we're doing, (2) let you
**confirm** every precondition is in place, (3) give the exact commands to run the eval.

---

## 0. Objective

Add **pi0.5** (OpenPI, fine-tuned on CALVIN to ~200k steps — checkpoint step `199999`) as a
baseline in our CALVIN language-robustness evaluation, in the **same protocol as the 3D Diffuser
Actor (3DDA) runs**:

- **34 CALVIN tasks** (the full set in `conf/evaluation/baseline.yaml`)
- **5 language conditions**: `BASE` (canonical annotation, aka "P0") + four perturbation axes
  `P1, P2, P3, P4`
- **25 trials per task per condition**
- driven by `scripts/run_evaluation.py`, identical runner/flags to the 3DDA baseline, so the
  numbers are directly comparable.

pi0.5 is a JAX/Flax model (~3.3 B params). It runs as a **separate OpenPI websocket policy
server**; the LangSteer eval talks to it through a thin **numpy-only client** policy. Both run on
this one ~16 GB-VRAM workstation, in **two separate venvs** (JAX and PyTorch stacks must not mix).

```
workstation
├── openpi venv   →  serve_pi05.py  →  WebsocketPolicyServer @ 127.0.0.1:8000   (JAX, holds weights)
└── langsteer venv → run_evaluation.py → Pi05Policy (openpi_client) → infer() over localhost
                        └── CALVIN (PyBullet) env, per-episode results JSON
```

pi0.5 emits a chunk of **10 relative** 7-DOF actions; the client returns the first `replan_steps`
rows as a **relative** `Action`; the env executes them as CALVIN flat-`(7,)` relative commands, then
re-infers (receding horizon).

---

## 1. Paths — set these first

```bash
export LANGSTEER=~/LangSteer            # this repo, branch feature/vla_finetuning
export OPENPI=~/openpi                  # cloned Physical-Intelligence/openpi
export CALVIN=~/calvin/dataset/task_D_D # CALVIN dataset dir (validation subset is enough)
export CKPT=$OPENPI/ckpt/199999         # pi0.5 checkpoint (server side)
```

---

## 2. What has already been built on this branch

These files are the "infra"; confirm they exist (§3-A). If any are missing, the branch was not
pushed/pulled with the pi0.5 change — pull again.

| File | Purpose |
|---|---|
| `policies/pi05.py` | `Pi05Policy(BasePolicy)` — numpy-only websocket client; packs obs, infers, returns a relative `Action`. |
| `core/types.py` | `Action.relative: bool = False` — signals relative execution. |
| `envs/calvin.py` | `step()` relative flat-`(7,)` branch; `_process_obs()` exposes native 84×84 gripper as `rgb['gripper_native']`. |
| `scripts/run_experiment.py` | `instantiate_policy` dispatch branch `name == "pi05"`. |
| `conf/policy/pi05.yaml` | client config (host/port, `action_horizon=10`, `replan_steps=5`, `ckpt_path: null`). |
| `conf/evaluation/pi05_baseline.yaml` | eval condition — structural mirror of `baseline.yaml` (34 tasks, `steering: none`, `policy_config: pi05`). |
| `serve_pi05.py` | server entry point (runs in the **openpi** venv). |

---

## 3. Preflight checklist — confirm each, report ✅/❌

### A. LangSteer branch infra (langsteer venv, `cd $LANGSTEER`)
```bash
git rev-parse --abbrev-ref HEAD                  # expect: feature/vla_finetuning
git log --oneline -1                             # expect a commit mentioning pi0.5
ls policies/pi05.py serve_pi05.py conf/policy/pi05.yaml conf/evaluation/pi05_baseline.yaml
grep -n "relative" core/types.py                 # Action.relative field present
grep -n '"pi05"' scripts/run_experiment.py       # dispatch branch present
grep -nE "gripper_native|is_relative" envs/calvin.py
ls perturbed_language_annotations.json           # needed for P1–P4

# static checks
uv run python -m py_compile policies/pi05.py serve_pi05.py core/types.py envs/calvin.py scripts/run_experiment.py
uv run python -c "from policies.pi05 import Pi05Policy; print('abstract:', set(Pi05Policy.__abstractmethods__) or 'concrete OK')"
uv run python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
    cfg = compose(config_name='config', overrides=['policy=pi05','steering=none'])
assert cfg.policy.name == 'pi05'
print('compose OK:', cfg.policy.name, cfg.policy.host, cfg.policy.port, cfg.policy.replan_steps)
PY
```
**✅ CONFIRM:** branch is `feature/vla_finetuning`; all 7 files present; static checks pass;
`Pi05Policy` reports `concrete OK`; compose prints `pi05 127.0.0.1 8000 5`.

### B. Checkpoint in place (server side)
```bash
ls "$CKPT"/params "$CKPT"/assets "$CKPT"/_CHECKPOINT_METADATA
du -sh "$CKPT"/params                             # ~12 GB
NS=$(find "$CKPT"/assets -name norm_stats.json | head -1); echo "$NS"
python3 -c "import json;d=json.load(open('$NS'))['norm_stats'];print('state',len(d['state']['mean']),'actions',len(d['actions']['mean']))"
[ -d "$CKPT/train_state" ] && echo 'train_state present (not needed)' || echo 'train_state absent (good)'
```
**✅ CONFIRM:** `params/` + `assets/` + `_CHECKPOINT_METADATA` present; `params/` ≈ 12 GB;
norm_stats prints **`state 15 actions 7`** (authoritative — verified from the pulled checkpoint);
`train_state` absent. Norm-stats path is `assets/aryannav/calvin_abc_d_train/norm_stats.json`.

> If the checkpoint is **not** here: transfer it (it is NOT in git). Either LAN-rsync from the source
> laptop `checkpoints/pi05_calvin/199999/`, or on this box:
> `aws --profile calvin s3 sync s3://calvin-abcd-dataset-bucket/langsteer/models/pi05_calvin/checkpoints/199999/ "$CKPT"/ --exclude "train_state/*"`

### C. CALVIN environment (langsteer venv)
```bash
uv run python -c "import calvin_env; print('calvin_env package OK')"     # from setup_calvin.sh
ls "$CALVIN"/validation/.hydra/merged_config.yaml
ls "$CALVIN"/validation/lang_annotations/auto_lang_ann.npy
```
The eval reads only the `validation/.hydra/` config + `validation/lang_annotations/` (~1.8 MB total).
The `episode_*.npz` frame files (~28 GB) are **not** needed (we use hardcoded `task_configs.py`
initial conditions, `use_task_initial_condition: true`).

**Path wiring (important):** `conf/env/calvin.yaml` hardcodes `dataset_path` and `lang_ann_path` to
the source laptop's paths. On this box either put CALVIN at the identical path, or edit those two
lines in `conf/env/calvin.yaml` to `$CALVIN` and `$CALVIN/validation/lang_annotations/auto_lang_ann.npy`.
(`run_evaluation.py` does not accept an env-path CLI override, so this must be set in the yaml.)

End-to-end env check (builds PyBullet, resets, no pi0.5 needed):
```bash
uv run python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
    cfg = compose(config_name='config',
                  overrides=['env.task=open_drawer','env.provide_pcd_images=true'])
from scripts.run_experiment import instantiate_env
env = instantiate_env(cfg); obs = env.reset()
print('reset OK | rgb keys:', sorted(obs.rgb), '| proprio:', obs.proprio.shape, '| instr:', repr(obs.instruction))
PY
```
**✅ CONFIRM:** `calvin_env` imports; the two config/annotation paths exist; the reset prints
`rgb keys: ['gripper', 'gripper_native', 'static']` and `proprio: (15,)` and a non-empty instruction.

### D. OpenPI server (openpi venv, `cd $OPENPI`)
```bash
uv run python -c "import jax; print('devices:', jax.devices())"                 # a GPU must appear
uv run python -c "from openpi.policies import policy_config; from openpi.serving import websocket_policy_server; print('openpi serving OK')"
ls serve_pi05.py pi05_calvin_config.py                                          # copied from $LANGSTEER
uv run python -c "from pi05_calvin_config import PI05_CALVIN_CONFIGS as C; print(C[0].name, 'horizon', C[0].model.action_horizon)"
```
Setup if missing: `git clone …/openpi && cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync`, then
`cp $LANGSTEER/serve_pi05.py $LANGSTEER/training/common/pi05_calvin_config.py .`.

**✅ CONFIRM:** `jax.devices()` lists the GPU; openpi serving imports; config prints
`pi05_calvin horizon 10`.

### E. Client dependency (langsteer venv, `cd $LANGSTEER`)
```bash
uv run python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__)"
uv run python -c "from openpi_client import websocket_client_policy; print('openpi_client OK')"
```
Install if missing (WITHOUT its pins — a plain install downgrades numpy<2 and breaks torch):
```bash
uv pip install --no-deps "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git@main#subdirectory=packages/openpi-client"
uv pip install "websockets>=13" "msgpack>=1.0.5"
```
**✅ CONFIRM:** numpy stays 2.x, torch import unaffected, `openpi_client` imports.

### F. End-to-end smoke (start the server first)
```bash
# openpi venv, terminal 1 — leave running:
cd $OPENPI && XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python serve_pi05.py --ckpt-dir "$CKPT" --port 8000

# langsteer venv, terminal 2:
cd $LANGSTEER
# F1 raw contract:
uv run python - <<'PY'
import numpy as np
from openpi_client import websocket_client_policy as w
c = w.WebsocketClientPolicy(host='127.0.0.1', port=8000)
a = c.infer({'observation/image':np.zeros((200,200,3),np.uint8),
             'observation/wrist_image':np.zeros((84,84,3),np.uint8),
             'observation/state':np.zeros(15,np.float32),'prompt':'open the drawer'})['actions']
print('actions', np.asarray(a).shape)            # expect (10, 7)
PY
# F2 single-env rollout (needs env.provide_pcd_images=true for pi0.5 via run_experiment):
uv run python scripts/run_experiment.py policy=pi05 policy.host=127.0.0.1 policy.port=8000 \
  env.task=open_drawer env.provide_pcd_images=true steering=none num_episodes=1 max_steps=360
```
**✅ CONFIRM:** F1 prints `(10, 7)`; F2 completes an episode without error, and the per-step log
shows **small (~1e-2) relative position deltas** (sign that relative execution is wired correctly,
not absolute).

---

## 4. Run the full evaluation (matches the 3DDA protocol)

Server must be running (§3-F terminal 1). From `$LANGSTEER` in the langsteer venv:

```bash
# BASE / P0 (canonical instruction — omit the perturbation flag)
uv run python scripts/run_evaluation.py --evaluation pi05_baseline --num-episodes 25 --max-steps 360

# P1–P4 (perturbed instructions)
for AX in P1 P2 P3 P4; do
  uv run python scripts/run_evaluation.py --evaluation pi05_baseline --num-episodes 25 --max-steps 360 --perturbation-axis $AX
done
```
- Match the 3DDA runs' `--max-steps` and `--seed` exactly (defaults: `max_steps=360`, `seed=42`).
- `run_evaluation.py` already force-sets `env.provide_pcd_images=true`, so pi0.5 gets the native
  200×200 static + `gripper_native` — no extra flag needed here (unlike the `run_experiment` smoke).
- Results are written per-condition under `outputs/evaluation/<timestamp>/` (updated after every
  episode; resumable via `--output-dir`). Aggregate with `scripts/summarize_evaluations.py`.

---

## 5. Risks / things to watch during the first rollout

1. **Relative-action fidelity (top risk):** pi0.5's `rel_actions` are fed as CALVIN flat-`(7,)`
   relative commands (scaled by `max_rel_pos`/`max_rel_orn`). Confirm commanded ≈ achieved dpos on a
   short rollout. If the arm barely moves or diverges, the relative scaling/convention is off.
2. **State / norm-stats:** `observation/state` must be the 15-dim `robot_obs` (= `obs.proprio`).
   Verified: checkpoint norm_stats `state`=15, `actions`=7. A wrong-length state normalizes silently
   wrong.
3. **Native gripper:** pi0.5 must get the native 84×84 gripper (`rgb['gripper_native']`), not the
   200×200 upsample. The client uses `gripper_key: gripper_native`; the env now provides it.
4. **Two venvs:** never install `openpi`/jax into the langsteer venv, or `torch` into the openpi
   venv.
5. **Server first:** `WebsocketClientPolicy` blocks/retries forever until the server answers — start
   §3-F terminal 1 before any client command.
6. **16 GB VRAM is tight:** keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`; if it OOMs, force PyBullet to
   the CPU (TinyRenderer) so JAX owns the GPU.

---

## 6. Report back (fill in)

```
[ ] A  branch feature/vla_finetuning + all 7 files + static checks pass
[ ] B  checkpoint present, params ~12G, norm_stats state=15 actions=7, no train_state
[ ] C  calvin_env imports; validation .hydra + lang_annotations present; reset() -> rgb {static,gripper,gripper_native}, proprio (15,)
[ ] D  openpi venv: jax sees GPU; serving imports; pi05_calvin config horizon=10
[ ] E  langsteer venv: numpy 2.x + torch intact; openpi_client imports
[ ] F  smoke: connect OK; raw infer (10,7); single-env open_drawer rollout completes with ~1e-2 relative deltas
Blockers / anomalies:
```

If A–F all pass, the full §4 sweep (5 conditions × 34 tasks × 25 trials) is cleared to run.
