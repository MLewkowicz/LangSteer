"""Sampling-diversity probe: does the trained policy produce the mode-B inversion?

The v6 wine-glass place task is bimodal. Mode A places upright (yaw stays ≈−1.5
the whole place, total |Δyaw| ≈ 0.2 rad). Mode B starts upright too, then INVERTS
the wrist partway through — a ~π rotation (total |Δyaw| ≈ 3.3 rad) ending at
yaw ≈ +1.4. Both modes share conditioning (primitive=place, object=glass) and an
upright start, so the inversion is a free choice the model must represent.

Because the model predicts only a short (~1 s) window per forward, you cannot see
the inversion by conditioning at the upright START and reading the window's final
yaw — the rotation happens later. The right signal is the IN-WINDOW rotation
(unwrapped Δyaw across the predicted trajectory), conditioned at frames AT/AFTER
the rotation onset of a mode-B episode. A model that learned B produces a large
|Δyaw| there; a collapsed model predicts ≈0 (stays upright) or pulls back to A.

Detector calibration (GT, full place): mode A total |Δyaw| mean 0.18 (max 0.51),
mode B mean 3.29 (min 2.80). A per-window threshold of ~1.0 rad cleanly flags
"executing the inversion".

Usage:
    # default: auto-pick a mode-B episode, probe at [start, onset, onset+1]
    uv run python scripts/probe_yaw_diversity.py

    # raw vs EMA at the training-native 100 steps, more draws
    uv run python scripts/probe_yaw_diversity.py --weights both --diffusion_timesteps 100 --n_samples 50

    # explicit frames / episode
    uv run python scripts/probe_yaw_diversity.py --dat .../D+0/ann_101.dat --frames 0 6 7

    # sweep checkpoints
    uv run python scripts/probe_yaw_diversity.py --diffusion_timesteps 100 \
        --ckpt outputs/checkpoints/.../00{20,40,60,79}*.pth
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import blosc
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policies.diffuser_actor import build_diffuser_actor_policy  # noqa: E402
from scripts.diagnose_inference_conditioning import _episode_obs  # noqa: E402

CKPT_DIR = "outputs/checkpoints/diffuser_actor_realworld_primitive_object_v6"
DEFAULT_CKPT = f"{CKPT_DIR}/last.pth"
POLICY_CFG = "conf/policy/diffuser_actor_realworld_primitive_object.yaml"
DATA_ROOT = "/home/clear/Documents/michal/realworld_3da_v6/training"
ROT_THRESH = 1.0  # rad of in-window |Δyaw| that counts as "executing the inversion"


# --------------------------------------------------------------------------
# Episode helpers
# --------------------------------------------------------------------------

def _keypose_yaws(ep) -> np.ndarray:
    """Per-keypose absolute yaw (ep[4] = list of (1,7) [pos3,eul3,grip])."""
    return np.array([g.numpy().reshape(-1)[5] for g in ep[4]])


def _rotation_onset_keypose(kp_yaws: np.ndarray) -> int:
    """Keypose index of the largest wrap-aware yaw step, or -1 if none > 0.5 rad."""
    d = np.abs((np.diff(kp_yaws) + np.pi) % (2 * np.pi) - np.pi)
    return int(np.argmax(d)) if len(d) and d.max() > 0.5 else -1


def _episode_final_mode(ep) -> str:
    """Mode label from the episode's final keypose yaw (>0 = B inversion)."""
    return "B" if _keypose_yaws(ep)[-1] > 0 else "A"


def _pick_mode_b_dat(ann_path: Path, dat_dir: Path) -> Path:
    """First place .dat whose final keypose yaw is positive (mode B)."""
    ann = np.load(ann_path, allow_pickle=True).item()
    prims = ann["info"]["primitive"]
    for i, p in enumerate(prims):
        f = dat_dir / f"ann_{i}.dat"
        if p == "place" and f.exists():
            ep = pickle.loads(blosc.decompress(f.read_bytes()))
            if _keypose_yaws(ep)[-1] > 0:
                return f
    raise FileNotFoundError(f"No mode-B place .dat under {dat_dir}")


# --------------------------------------------------------------------------
# Checkpoint weight loading (raw `weight` vs `ema_weight`)
# --------------------------------------------------------------------------

def _load_weights(policy, ckpt_path: str, which: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    key = "ema_weight" if which == "ema" else "weight"
    if key not in ckpt:
        raise KeyError(f"{ckpt_path} has no '{key}' (keys: {list(ckpt)})")
    state = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt[key].items()}
    missing, unexpected = policy._model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys, e.g. {missing[:3]}")
    policy._model.eval()


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def _sample_window(policy, ep, frame_idx, primitive, object_id, seed, device, nhist):
    """One stochastic rollout from a primed observation. Returns (start_yaw, end_yaw, dyaw)."""
    policy.reset()
    # Prime gripper history with the preceding frames so a mid-rotation state
    # carries "rotation in progress" context, matching how deploy fills history.
    for pf in range(max(0, frame_idx - nhist + 1), frame_idx):
        policy._prepare_gripper(_episode_obs(ep, pf))
    obs = _episode_obs(ep, frame_idx)
    policy.set_primitive(primitive)
    policy.set_object(object_id)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    traj = policy.forward(obs).trajectory  # (H, 7): pos(3) euler(3) grip(1)
    yaw = np.unwrap(traj[:, 5])
    return float(traj[0, 5]), float(traj[-1, 5]), float(yaw[-1] - yaw[0])


def _ascii_hist(vals, lo, hi, bins=24, mark0=True) -> str:
    counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    peak = max(1, counts.max())
    out = []
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        mid = 0.5 * (e0 + e1)
        marker = " <-0" if (mark0 and e0 <= 0.0 < e1) else ""
        out.append(f"  {mid:+5.2f} | {'#' * int(round(36 * c / peak)):<36} {c}{marker}")
    return "\n".join(out)


def probe_frame(policy, ep, frame_idx, primitive, object_id, n_samples, base_seed,
                device, nhist, rot_thresh, verbose=True) -> dict:
    """Sample N windows from one frame; return rotation-diversity metrics."""
    rows = [_sample_window(policy, ep, frame_idx, primitive, object_id,
                           base_seed + i, device, nhist)
            for i in range(n_samples)]
    start_yaw = np.array([r[0] for r in rows])
    end_yaw = np.array([r[1] for r in rows])
    dyaw = np.array([r[2] for r in rows])
    rotating = np.abs(dyaw) >= rot_thresh
    cond_yaw = float(_keypose_yaws(ep)[frame_idx])
    m = {
        "frame": frame_idx,
        "cond_yaw": cond_yaw,
        "mean_abs_dyaw": float(np.abs(dyaw).mean()),
        "max_abs_dyaw": float(np.abs(dyaw).max()),
        "frac_rotating": float(rotating.mean()),
        "n_rotating": int(rotating.sum()),
        "n": n_samples,
        "end_yaw_pos": int((end_yaw > 0).sum()),
    }
    if verbose:
        print(f"\n--- frame {frame_idx} (conditioning yaw {cond_yaw:+.2f}) ---")
        print(f"  in-window |Δyaw|: mean {m['mean_abs_dyaw']:.2f}  max {m['max_abs_dyaw']:.2f} rad")
        print(f"  executing inversion (|Δyaw|>={rot_thresh}): "
              f"{m['n_rotating']}/{n_samples} ({m['frac_rotating']:.0%})")
        print(f"  window end-yaw>0: {m['end_yaw_pos']}/{n_samples}")
        print("  Δyaw distribution:")
        print(_ascii_hist(dyaw, -np.pi, np.pi))
    return m


def probe_checkpoint(policy, ckpt_path, which, ep, frames, primitive, object_id,
                     n_samples, base_seed, device, nhist, rot_thresh,
                     verbose=True) -> dict:
    """Load weights and probe every frame. Returns the max frac_rotating over frames."""
    _load_weights(policy, ckpt_path, which, device)
    per_frame = [probe_frame(policy, ep, f, primitive, object_id, n_samples,
                             base_seed, device, nhist, rot_thresh, verbose)
                 for f in frames]
    best = max(per_frame, key=lambda d: d["frac_rotating"])
    return {"ckpt": Path(ckpt_path).name, "which": which,
            "best_frac_rotating": best["frac_rotating"],
            "best_frame": best["frame"], "per_frame": per_frame}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", nargs="+", default=[DEFAULT_CKPT])
    ap.add_argument("--policy_cfg", default=POLICY_CFG)
    ap.add_argument("--weights", choices=["raw", "ema", "both"], default="raw")
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--dat", default=None, help="Mode-B place .dat to condition on (default: auto-pick).")
    ap.add_argument("--frames", type=int, nargs="+", default=None,
                    help="Frame (keypose) indices to condition at. Default: [0, onset, onset+1].")
    ap.add_argument("--primitive", type=int, default=1, help="place=1")
    ap.add_argument("--object", type=int, default=0, help="glass=0")
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rot_thresh", type=float, default=ROT_THRESH)
    ap.add_argument("--diffusion_timesteps", type=int, default=None)
    ap.add_argument("--corrector_steps", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.policy_cfg)
    cfg.device = args.device
    if args.diffusion_timesteps is not None:
        cfg.diffusion_timesteps = args.diffusion_timesteps
    if args.corrector_steps is not None:
        cfg.corrector_steps = args.corrector_steps
    nhist = int(cfg.get("nhist", 3))

    root = Path(args.data_root)
    dat = Path(args.dat) if args.dat else _pick_mode_b_dat(
        root / "lang_annotations" / "primitive_object_lang_ann.npy", root / "D+0")
    ep = pickle.loads(blosc.decompress(dat.read_bytes()))
    n_kp = len(ep[4])
    onset = _rotation_onset_keypose(_keypose_yaws(ep))
    frames = args.frames if args.frames is not None else (
        sorted({0, onset, min(onset + 1, n_kp - 1)}) if onset >= 0 else [0, n_kp // 2, n_kp - 1])
    frames = [f for f in frames if 0 <= f < n_kp]

    print(f"Mode-B episode: {dat.name}  ({n_kp} keyposes, final mode "
          f"{_episode_final_mode(ep)}, rotation onset keypose {onset})")
    print(f"Conditioning frames: {frames}  |  inference steps "
          f"{cfg.diffusion_timesteps}, corrector {cfg.get('corrector_steps', 0)}, "
          f"rot_thresh {args.rot_thresh} rad")

    cfg.ckpt_path = args.ckpt[0]
    policy = build_diffuser_actor_policy(cfg)
    which_list = ["raw", "ema"] if args.weights == "both" else [args.weights]
    for ckpt in args.ckpt:
        for which in which_list:
            print(f"\n===== {Path(ckpt).name} [{which}] =====")
            r = probe_checkpoint(policy, ckpt, which, ep, frames, args.primitive,
                                 args.object, args.n_samples, args.seed, args.device,
                                 nhist, args.rot_thresh)
            verdict = ("COLLAPSED (no inversion sampled)" if r["best_frac_rotating"] == 0
                       else f"INVERSION PRESENT: up to {r['best_frac_rotating']:.0%} "
                            f"of draws rotate (frame {r['best_frame']})")
            print(f"  >>> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
