"""tkinter + matplotlib GUI for annotating semantic bounding boxes in Isaac Sim.

Replaces tmp/app.py for the Isaac Sim workflow.  Displays the static camera
RGB from IsaacSimScene and overlays 3D bounding boxes projected to pixel space.
Sliders allow live adjustment of each fixture's position, size, and orientation.

Usage::

    # From repo root, with Isaac Sim environment active:
    python -c "
    from envs.isaac_sim_utils.scene import IsaacSimScene
    from envs.isaac_sim_utils.annotate_scene import capture_snapshot
    from envs.isaac_sim_utils.annotate_app import AnnotatorApp

    scene = IsaacSimScene({'use_gui': False, 'task': 'pick_up_red_block'})
    scene.spawn_objects(...)
    scene.step_physics(20)
    snap = capture_snapshot(scene)
    app = AnnotatorApp(scene, snap)
    app.run()
    scene.close()
    "

Exports JSON + a ready-to-paste Python snippet for updating
INITIAL_OVERRIDES in annotate_scene.py.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from envs.isaac_sim_utils.annotate_scene import (
    COLORS,
    INITIAL_DERIVED,
    INITIAL_OVERRIDES,
    SceneSnapshot,
    compute_bboxes,
)

logger = logging.getLogger(__name__)

IMAGE_SIZE = 512
SLIDER_RANGE = 0.35
SLIDER_STEP = 0.001
EULER_RANGE_DEG = 180.0
EULER_STEP_DEG = 0.5

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# 3D OBB projection (pure numpy, ported from tmp/projection.py)
# ---------------------------------------------------------------------------

_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

_UNIT_CUBE = np.array([
    [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5],
    [-0.5, +0.5, -0.5], [+0.5, +0.5, -0.5],
    [-0.5, -0.5, +0.5], [+0.5, -0.5, +0.5],
    [-0.5, +0.5, +0.5], [+0.5, +0.5, +0.5],
], dtype=np.float64)


def _obb_corners(center: np.ndarray, size: np.ndarray, R: np.ndarray) -> np.ndarray:
    local = _UNIT_CUBE * np.asarray(size, dtype=np.float64)
    return (np.asarray(R, dtype=np.float64) @ local.T).T + np.asarray(center, dtype=np.float64)


def _build_mvp(extrinsic: np.ndarray, intrinsic: Dict[str, float],
               w: int, h: int) -> np.ndarray:
    """Build camera → NDC matrix from Isaac Sim extrinsic + intrinsic."""
    # Extrinsic is cam-to-world; we need world-to-cam
    V = np.linalg.inv(extrinsic)
    # Build projection from intrinsics
    fx, fy = intrinsic["fx"], intrinsic["fy"]
    cx, cy = intrinsic["cx"], intrinsic["cy"]
    near, far = 0.01, 5.0
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2 * fx / w
    P[1, 1] = 2 * fy / h
    P[0, 2] = 1.0 - 2 * cx / w
    P[1, 2] = 2 * cy / h - 1.0
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] = -1.0
    return P @ V


def _project_points(pts: np.ndarray, mvp: np.ndarray,
                    img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    clip = h @ mvp.T
    w_clip = clip[:, 3]
    visible = w_clip > 1e-6
    safe_w = np.where(visible, w_clip, 1.0)
    ndc = clip[:, :3] / safe_w[:, None]
    u = (ndc[:, 0] + 1.0) * 0.5 * img_w
    v = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * img_h
    return np.stack([u, v], axis=1), visible


def _edges_2d(corners: np.ndarray, mvp: np.ndarray,
              img_w: int, img_h: int) -> Tuple[List[float], List[float]]:
    pixels, vis = _project_points(corners, mvp, img_w, img_h)
    xs, ys = [], []
    for i, j in _EDGES:
        if not (vis[i] and vis[j]):
            continue
        xs.extend([pixels[i, 0], pixels[j, 0], float("nan")])
        ys.extend([pixels[i, 1], pixels[j, 1], float("nan")])
    return xs, ys


def _center_pixel(center: np.ndarray, mvp: np.ndarray,
                  img_w: int, img_h: int) -> Tuple[float, float] | None:
    pix, vis = _project_points(center[None, :], mvp, img_w, img_h)
    if not vis[0]:
        return None
    return float(pix[0, 0]), float(pix[0, 1])


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _fmt_arr(v, fmt=".4f") -> str:
    return f"np.array([{', '.join(f'{x:{fmt}}' for x in v)}])"


def _render_python_snippet(overrides: Dict, derived: Dict) -> str:
    lines = ["    _FIXTURE_AABB_OVERRIDES = {"]
    for name, spec in overrides.items():
        lines.append(f"        '{name}': {{")
        lines.append(f"            'rest_position': {_fmt_arr(spec['rest_position'])},")
        lines.append(f"            'size':          {_fmt_arr(spec['size'])},")
        lines.append(f"            'euler_xyz_deg': {_fmt_arr(spec.get('euler_xyz_deg', [0, 0, 0]), '.2f')},")
        lines.append("        },")
    lines.append("    }")
    lines.append("    _DERIVED_OFFSETS = {")
    for name, spec in derived.items():
        lines.append(f"        '{name}': {{")
        lines.append(f"            'parent':        '{spec['parent']}',")
        lines.append(f"            'offset':        {_fmt_arr(spec['offset'])},")
        sz = spec.get("size")
        if sz is None:
            lines.append("            'size':          None,")
        else:
            lines.append(f"            'size':          {_fmt_arr(sz)},")
        lines.append(f"            'euler_xyz_deg': {_fmt_arr(spec.get('euler_xyz_deg', [0, 0, 0]), '.2f')},")
        lines.append("        },")
    lines.append("    }")
    return "\n".join(lines)


def export_tuned(overrides: Dict, derived: Dict) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _REPO_ROOT / "envs" / "isaac_sim_utils"
    fname = out_dir / f"overrides_{stamp}.json"

    def _clean(d: dict) -> dict:
        out = {}
        for name, spec in d.items():
            out[name] = {
                k: (list(v) if isinstance(v, (np.ndarray, list, tuple)) else v)
                for k, v in spec.items()
                if not k.startswith("_")
            }
        return out

    clean = {"overrides": _clean(overrides), "derived": _clean(derived)}
    fname.write_text(json.dumps(clean, indent=2))

    snippet = _render_python_snippet(_clean(overrides), _clean(derived))
    sep = "=" * 70
    print(f"\n{sep}")
    print("BBOX EXPORT — paste into envs/isaac_sim_utils/annotate_scene.py")
    print(sep)
    print(snippet)
    print(sep)
    print(f"JSON: {fname}")
    print(f"{sep}\n")
    return f"saved {fname.name}"


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------

class AnnotatorApp:
    """Single-window GUI: matplotlib image canvas + scrollable slider panel."""

    def __init__(self, scene, snapshot: SceneSnapshot) -> None:
        self._scene = scene
        self._snapshot = snapshot
        self._overrides = copy.deepcopy(INITIAL_OVERRIDES)
        self._derived = copy.deepcopy(INITIAL_DERIVED)

        # Capture initial camera data for the background image
        rgb, _, intrinsics, extrinsics = scene.get_camera_data()
        self._rgb = rgb.get("static", np.zeros((200, 200, 3), dtype=np.uint8))
        self._intrinsics = intrinsics.get("static", {"fx": 200, "fy": 200, "cx": 100, "cy": 100})
        self._extrinsics = extrinsics.get("static", np.eye(4))
        self._mvp = _build_mvp(self._extrinsics, self._intrinsics, IMAGE_SIZE, IMAGE_SIZE)

        self._overlay_artists: list = []
        self._image_artist = None
        self._build_window()
        self._draw_initial_image()
        self._build_sliders()
        self._redraw_overlay()

    def _build_window(self) -> None:
        self._root = tk.Tk()
        self._root.title("Isaac Sim BBox Annotator")
        self._root.configure(bg="#111")
        self._root.geometry("1280x820")
        self._root.columnconfigure(0, weight=1)
        self._root.columnconfigure(1, weight=0)
        self._root.rowconfigure(0, weight=1)

        self._fig = Figure(figsize=(7, 7), facecolor="#000")
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#000")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_visible(False)
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self._root)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        side = tk.Frame(self._root, bg="#111", width=480)
        side.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        side.pack_propagate(False)

        top_bar = tk.Frame(side, bg="#111")
        top_bar.pack(fill="x", pady=(0, 4))

        tk.Button(
            top_bar, text="Export", command=self._on_export,
            bg="#2ca02c", fg="white", relief="flat",
            font=("monospace", 10, "bold"), padx=14, pady=6, cursor="hand2",
        ).pack(side="left")

        self._status_var = tk.StringVar(value="")
        tk.Label(
            top_bar, textvariable=self._status_var, bg="#111", fg="#aaa",
            font=("monospace", 9), anchor="w",
        ).pack(side="left", padx=10)

        self._scroll_canvas = tk.Canvas(side, bg="#111", highlightthickness=0, width=460)
        sb = tk.Scrollbar(side, orient="vertical", command=self._scroll_canvas.yview)
        self._sframe = tk.Frame(self._scroll_canvas, bg="#111")
        self._sframe.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all")
            ),
        )
        self._scroll_canvas.create_window((0, 0), window=self._sframe, anchor="nw")
        self._scroll_canvas.configure(yscrollcommand=sb.set)
        self._scroll_canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        def _wheel(event):
            delta = -1 if event.num == 5 or event.delta < 0 else 1
            self._scroll_canvas.yview_scroll(-delta, "units")

        for w in (self._scroll_canvas, self._sframe):
            w.bind("<MouseWheel>", _wheel)
            w.bind("<Button-4>", _wheel)
            w.bind("<Button-5>", _wheel)

    def _draw_initial_image(self) -> None:
        rgb_disp = np.array(
            plt.cm.gray(np.zeros((IMAGE_SIZE, IMAGE_SIZE)))[:, :, :3] * 255,
            dtype=np.uint8,
        )
        if self._rgb is not None:
            from PIL import Image
            try:
                pil = Image.fromarray(self._rgb).resize((IMAGE_SIZE, IMAGE_SIZE))
                rgb_disp = np.array(pil)
            except Exception:
                pass
        self._image_artist = self._ax.imshow(
            rgb_disp, origin="upper", extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0)
        )
        self._ax.set_xlim(0, IMAGE_SIZE)
        self._ax.set_ylim(IMAGE_SIZE, 0)
        self._canvas.draw()

    def _build_sliders(self) -> None:
        def section(title: str) -> None:
            tk.Label(
                self._sframe, text=title, bg="#111", fg="#888",
                font=("monospace", 9), anchor="w",
            ).pack(fill="x", padx=6, pady=(10, 2))

        section("Fixtures")
        for name in self._overrides:
            self._add_object_block(name, "fixture", self._overrides[name])
        if self._derived:
            section("Derived (offset from parent)")
            for name in self._derived:
                self._add_object_block(name, "derived", self._derived[name])

    def _add_object_block(self, name: str, group: str, values: Dict) -> None:
        pos_key = "rest_position" if group == "fixture" else "offset"
        pos = values[pos_key]
        size = values.get("size") or [0.1, 0.1, 0.1]
        euler = values.get("euler_xyz_deg", [0.0, 0.0, 0.0])
        color = COLORS.get(name, "#ccc")

        header = tk.Frame(self._sframe, bg="#111")
        header.pack(fill="x", padx=6, pady=(8, 2))
        tk.Label(header, text=name, bg="#111", fg=color,
                 font=("monospace", 10, "bold"), anchor="w").pack(side="left")
        tk.Frame(header, bg=color, height=2).pack(
            side="left", fill="x", expand=True, padx=6, pady=8
        )

        store = self._overrides if group == "fixture" else self._derived

        def cb_factory(field_name: str, axis: int):
            def _cb(val_str):
                try:
                    v = float(val_str)
                except (TypeError, ValueError):
                    return
                target = store[name].get(field_name)
                if target is None or axis >= len(target):
                    return
                target[axis] = v
                self._redraw_overlay()
            return _cb

        for i, axis in enumerate(["x", "y", "z"]):
            self._make_slider(
                label=f"{pos_key[0]}{axis}",
                initial=float(pos[i]),
                minv=float(pos[i]) - SLIDER_RANGE,
                maxv=float(pos[i]) + SLIDER_RANGE,
                step=SLIDER_STEP,
                on_change=cb_factory(pos_key, i),
            )
        for i, axis in enumerate(["x", "y", "z"]):
            self._make_slider(
                label=f"s{axis}",
                initial=float(size[i]),
                minv=max(0.001, float(size[i]) - SLIDER_RANGE),
                maxv=float(size[i]) + SLIDER_RANGE,
                step=SLIDER_STEP,
                on_change=cb_factory("size", i),
            )
        for i, axis in enumerate(["x", "y", "z"]):
            self._make_slider(
                label=f"r{axis}°",
                initial=float(euler[i]),
                minv=-EULER_RANGE_DEG,
                maxv=EULER_RANGE_DEG,
                step=EULER_STEP_DEG,
                on_change=cb_factory("euler_xyz_deg", i),
            )

    def _make_slider(self, label: str, initial: float, minv: float, maxv: float,
                     step: float, on_change) -> tk.Scale:
        row = tk.Frame(self._sframe, bg="#111")
        row.pack(fill="x", padx=6, pady=0)
        tk.Label(row, text=label, bg="#111", fg="#ccc",
                 font=("monospace", 9), width=5, anchor="w").pack(side="left")
        scale = tk.Scale(
            row, from_=minv, to=maxv, resolution=step, orient="horizontal",
            bg="#222", fg="#ddd", troughcolor="#333",
            highlightthickness=0, sliderlength=14, length=380,
            showvalue=True, font=("monospace", 7), command=on_change,
        )
        scale.set(initial)
        scale.pack(side="left", fill="x", expand=True)
        return scale

    def _redraw_overlay(self) -> None:
        for a in self._overlay_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._overlay_artists.clear()

        bboxes = compute_bboxes(self._snapshot, self._overrides, self._derived)
        for name, bb in bboxes.items():
            corners = _obb_corners(bb["center"], bb["size"], bb["rotation"])
            xs, ys = _edges_2d(corners, self._mvp, IMAGE_SIZE, IMAGE_SIZE)
            if not xs:
                continue
            color = COLORS.get(name, "#cccccc")
            width = 1.8 if bb["editable"] else 0.7
            ls = "-" if bb["editable"] else ":"
            (line,) = self._ax.plot(xs, ys, color=color, linewidth=width,
                                    linestyle=ls, zorder=5)
            self._overlay_artists.append(line)
            c_px = _center_pixel(bb["center"], self._mvp, IMAGE_SIZE, IMAGE_SIZE)
            if c_px is not None:
                txt = self._ax.text(c_px[0], c_px[1] - 6, name, color=color,
                                    fontsize=8, ha="center", va="bottom", zorder=6)
                self._overlay_artists.append(txt)

        self._ax.set_xlim(0, IMAGE_SIZE)
        self._ax.set_ylim(IMAGE_SIZE, 0)
        self._canvas.draw_idle()

    def _on_export(self) -> None:
        msg = export_tuned(self._overrides, self._derived)
        self._status_var.set(msg)

    def run(self) -> None:
        self._root.mainloop()
