"""Real-world dataset variant with fixed-horizon sliding-window trajectories.

Background
----------
The packaged real-world `.dat` shards (output of
`convert_realworld_for_diffuser_actor.py`) anchor observations at keyposes and
store, per keypose, the *dense* proprio path from that keypose to the next
(`episode[5][m]`, sharing the boundary frame with its neighbour). The default
`CalvinDataset` trains on one keypose interval per sample, which yields
short, high-variance trajectory targets (median ~few cm, fat tail) — the model
then collapses to predicting near-stationary plans.

This subclass keeps everything about `CalvinDataset` the same EXCEPT how the
low-level trajectory target is built: it reconstructs the full dense proprio
path for the episode and, for each keypose anchor, takes a fixed number of
dense frames forward (`horizon_frames`) before interpolating to
`interpolation_length` waypoints. Every sample therefore covers the SAME
temporal extent regardless of keypose spacing, giving a uniform-scale target
that's much easier to fit and produces longer plans at deploy.

Only `_build_traj_items` is overridden; observations, gripper history,
relative-action handling, and the collate contract are inherited unchanged.
The Isaac path uses a different dataset class and is unaffected.
"""

from __future__ import annotations

import torch

from training.policies.diffuser_actor.dataset import CalvinDataset


class RealworldSlidingWindowDataset(CalvinDataset):
    """CalvinDataset with fixed-time-horizon sliding-window trajectories."""

    def __init__(self, *args, horizon_frames: int | None = None, **kwargs):
        """`horizon_frames` is measured in *effective* (post-stride) state
        frames; at the converter's ~10 Hz target, N frames ≈ N/10 s per plan.

        FIDELITY NOTE: the window of real recorded frames is resampled to
        `interpolation_length` waypoints by CubicSpline. When
        horizon_frames == interpolation_length the spline is evaluated exactly
        at its input knots, so every output waypoint is an actual recorded
        robot pose (no synthesis). horizon_frames < interpolation_length
        upsamples (synthesises intermediate points); > downsamples. Default
        (None) sets horizon_frames = interpolation_length for the faithful
        1:1 case. Raise it only if you deliberately want longer (down-sampled)
        plans and accept the spline approximation between recorded frames.
        """
        super().__init__(*args, **kwargs)
        interp_len = int(kwargs.get("interpolation_length", 100))
        self._horizon_frames = (
            int(horizon_frames) if horizon_frames is not None else interp_len
        )

    @staticmethod
    def _reconstruct_dense(trajs) -> tuple[torch.Tensor, list[int]]:
        """Stitch per-keypose trajectories into one dense (D, F) proprio path.

        Consecutive entries share a boundary frame (trajs[m][0] == trajs[m-1][-1]),
        so we drop the duplicate when concatenating. Returns the dense path and
        `anchor_start[m]` = the dense-path index where keypose m's pose sits.
        """
        if len(trajs) == 0:
            return torch.zeros((0, 0)), []
        parts = [trajs[0]]
        anchor_start = [0]
        cum = trajs[0].shape[0] - 1  # index of the shared boundary frame
        for m in range(1, len(trajs)):
            anchor_start.append(cum)
            parts.append(trajs[m][1:])
            cum += trajs[m].shape[0] - 1
        dense = torch.cat(parts, dim=0)
        return dense, anchor_start

    def _build_traj_items(self, episode, frame_ids):
        # Fall back to the parent (keypose-interval) behaviour if the shard has
        # no stored trajectories (len(episode) <= 5).
        if len(episode) <= 5:
            return super()._build_traj_items(episode, frame_ids)

        trajs = episode[5]
        dense, anchor_start = self._reconstruct_dense(trajs)
        n = len(trajs)
        H = self._horizon_frames

        items = []
        for i in frame_ids:
            # frame_ids index into the .dat's kept (keypose) frames, same as trajs.
            if not (0 <= i < n):
                # defensive; shouldn't happen given chunk slicing
                items.append(self._interpolate_traj(trajs[max(0, min(i, n - 1))]))
                continue
            start = anchor_start[i]
            window = dense[start:start + H]
            # Near the episode end the window can be shorter than H, and right
            # at the final frame it may be a single pose — interpolation needs
            # >= 2 points, so fall back to that keypose's own interval there.
            if window.shape[0] < 2:
                window = trajs[i]
            items.append(self._interpolate_traj(window))
        return items
