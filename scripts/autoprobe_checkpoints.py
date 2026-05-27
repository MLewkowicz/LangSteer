"""Watch a checkpoint dir and auto-probe each new checkpoint for the mode-B inversion.

Polls --ckpt_dir for new / updated NNNNNNN.pth files and runs the yaw-rotation
probe on each, logging a compact one-line summary. The tracked headline is the
fraction of draws that INITIATE the wrist inversion from the upright start
(frame 0) — the decision point that is currently broken (0% at 25 steps, ~20% at
100 steps for the v6 80k checkpoint). As training with the low-t loss progresses,
that fraction should climb toward the data's ~43% mode-B prior.

By default only checkpoints modified AFTER the watcher starts are probed (so it
doesn't grind through the ~160 existing v6 files); pass --probe_existing to probe
what's already there.

Usage:
    # watch the run that the current training config writes to (25 = trained regime)
    uv run python scripts/autoprobe_checkpoints.py \
        --ckpt_dir outputs/checkpoints/diffuser_actor_realworld_primitive_object_v6_resume \
        --interval 120

    # one pass over everything already on disk, then exit
    uv run python scripts/autoprobe_checkpoints.py --ckpt_dir <dir> --probe_existing --once
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import blosc
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policies.diffuser_actor import build_diffuser_actor_policy  # noqa: E402
from scripts.probe_yaw_diversity import (  # noqa: E402
    DATA_ROOT, POLICY_CFG, ROT_THRESH,
    _keypose_yaws, _pick_mode_b_dat, _rotation_onset_keypose, probe_checkpoint,
)

STEP_RE = re.compile(r"(\d{7})\.pth$")


def _discover(ckpt_dir: Path):
    """Return {path: mtime} for step-numbered checkpoints, sorted by step."""
    out = {}
    for f in sorted(ckpt_dir.glob("*.pth")):
        if STEP_RE.search(f.name):
            out[f] = f.stat().st_mtime
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--policy_cfg", default=POLICY_CFG)
    ap.add_argument("--weights", choices=["raw", "ema"], default="raw")
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--dat", default=None, help="Mode-B place .dat (default: auto-pick).")
    ap.add_argument("--frames", type=int, nargs="+", default=None,
                    help="Conditioning frames. Default: [0 (decision point), onset-1].")
    ap.add_argument("--n_samples", type=int, default=24)
    ap.add_argument("--diffusion_timesteps", type=int, default=25,
                    help="Inference steps = scheduler num_train_timesteps. Keep at "
                         "the trained 25; higher values run a mismatched schedule.")
    ap.add_argument("--rot_thresh", type=float, default=ROT_THRESH)
    ap.add_argument("--interval", type=float, default=120.0, help="Poll interval (s).")
    ap.add_argument("--logfile", default=None)
    ap.add_argument("--probe_existing", action="store_true",
                    help="Also probe checkpoints already present at startup.")
    ap.add_argument("--once", action="store_true", help="One poll then exit.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    logfile = Path(args.logfile) if args.logfile else ckpt_dir / "autoprobe_yaw.log"

    cfg = OmegaConf.load(args.policy_cfg)
    cfg.device = args.device
    cfg.diffusion_timesteps = args.diffusion_timesteps
    nhist = int(cfg.get("nhist", 3))

    root = Path(args.data_root)
    dat = Path(args.dat) if args.dat else _pick_mode_b_dat(
        root / "lang_annotations" / "primitive_object_lang_ann.npy", root / "D+0")
    ep = pickle.loads(blosc.decompress(dat.read_bytes()))
    n_kp = len(ep[4])
    onset = _rotation_onset_keypose(_keypose_yaws(ep))
    frames = args.frames if args.frames is not None else (
        sorted({0, max(onset - 1, 0)}) if onset >= 0 else [0])
    frames = [f for f in frames if 0 <= f < n_kp]

    # Build the policy once; probe_checkpoint() reloads weights per checkpoint.
    cfg.ckpt_path = str(next(iter(_discover(ckpt_dir)), ckpt_dir / "last.pth"))
    policy = build_diffuser_actor_policy(cfg)

    def log(msg: str):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(logfile, "a") as fh:
            fh.write(line + "\n")

    log(f"autoprobe watching {ckpt_dir}  episode={dat.name} frames={frames} "
        f"steps={args.diffusion_timesteps} n={args.n_samples} weights={args.weights}")
    log(f"headline = frac of draws initiating inversion from upright start "
        f"(frame {frames[0]}); target ~43% (data mode-B prior)")

    seen = {} if args.probe_existing else dict(_discover(ckpt_dir))

    while True:
        current = _discover(ckpt_dir)
        new = [p for p, mt in current.items() if seen.get(p) != mt]
        for p in sorted(new, key=lambda x: x.name):
            step = STEP_RE.search(p.name).group(1)
            try:
                r = probe_checkpoint(policy, str(p), args.weights, ep, frames,
                                     primitive=1, object_id=0, n_samples=args.n_samples,
                                     base_seed=0, device=args.device, nhist=nhist,
                                     rot_thresh=args.rot_thresh, verbose=False)
            except Exception as e:  # noqa: BLE001 — keep the watcher alive
                log(f"step {step} {args.weights} | ERROR {type(e).__name__}: {e}")
                seen[p] = current[p]
                continue
            f0 = r["per_frame"][0]
            parts = " ".join(
                f"f{m['frame']}:{m['frac_rotating']:.0%}rot/|Δ|{m['mean_abs_dyaw']:.2f}"
                for m in r["per_frame"])
            log(f"step {step} {args.weights} | start(f{f0['frame']}) "
                f"{f0['frac_rotating']:.0%} initiate | best {r['best_frac_rotating']:.0%} "
                f"@f{r['best_frame']} | {parts}")
            seen[p] = current[p]

        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
