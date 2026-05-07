"""Live tkinter + matplotlib 3D costmap window for VoxPoser steering.

Replaces the old browser-based Dash server. The window is driven directly
from the experiment loop's `step_callback`: each tick pumps Tk events and
re-binds matplotlib artists to the current `ValueMap` state.

The window stays responsive without `mainloop()` — `tick()` calls
`root.update()` so mouse-drag rotation, scrolling, and close events fire
between env steps. The single-threaded design means what's drawn is
exactly what `VoxPoserSteering` is using to steer.
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Any, List, Optional

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

from voxposer.calvin_interface import obb_world_corners, voxel2pc

logger = logging.getLogger(__name__)


_OBB_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
    (4, 5), (5, 7), (7, 6), (6, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]

_OBJ_COLORS = {
    'red': '#ff5555', 'blue': '#5599ff', 'pink': '#ff77cc',
    'drawer': '#ffaa33', 'slider': '#ffaa33', 'lightbulb': '#ffd633',
    'light_switch': '#ffd633', 'switch': '#ffd633', 'led': '#ffd633',
    'button': '#ffaa33', 'table': '#888888',
}

# Color the primitive label by what kind of motion it represents.
_PRIMITIVE_COLORS = {
    'grasp': '#ffd633',  # yellow — closure
    'push':  '#7ce28a',  # green  — forward contact
    'pull':  '#ff8866',  # orange — retract
    'place': '#9ad7ff',  # blue   — release
}


def _color_for_object(name: str) -> str:
    lname = name.lower()
    for key, col in _OBJ_COLORS.items():
        if key in lname:
            return col
    return '#ffaa33'


class LiveCostmapWindow:
    """Single-window live 3D costmap viewer driven from the rollout loop."""

    def __init__(self, refresh_interval: int = 1, downsample: int = 4,
                 point_threshold: float = 0.05):
        self.refresh_interval = max(1, int(refresh_interval))
        self.downsample = max(1, int(downsample))
        self.point_threshold = float(point_threshold)

        self._state: dict = {}
        self._tick_counter = 0
        self._last_stage_idx: Optional[int] = None
        self._axes_initialized = False

        # Artist handles (created lazily, mutated in place across ticks).
        self._aff_scatter = None
        self._avoid_scatter = None
        self._target_scatter = None
        self._gripper_scatter = None
        self._obb_collection: Optional[Line3DCollection] = None
        self._obb_text_artists: list = []

        self._build_window()
        logger.info("LiveCostmapWindow opened (live tk costmap viewer)")

    # ------------------------------------------------------------------
    # Window scaffolding
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        self.root = tk.Tk()
        self.root.title("LangSteer — Live Costmap")
        self.root.configure(bg='#111')
        self.root.geometry('1100x780')
        self.root.protocol('WM_DELETE_WINDOW', self._on_close_request)
        self._closed = False

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(7, 7), facecolor='#111')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#111')
        self.ax.tick_params(colors='#aaa', labelsize=7)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.label.set_color('#aaa')
            axis.line.set_color('#444')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew',
                                         padx=4, pady=4)

        side = tk.Frame(self.root, bg='#111', width=260)
        side.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        side.pack_propagate(False)

        self.stage_var = tk.StringVar(value='waiting for costmap…')
        self.step_var = tk.StringVar(value='step —')
        self.instruction_var = tk.StringVar(value='')
        self.primitive_var = tk.StringVar(value='—')

        tk.Label(side, text='PRIMITIVE', bg='#111', fg='#888',
                 font=('monospace', 9)).pack(fill='x', pady=(8, 0))
        # Foreground color is mutated per-tick to match the active primitive.
        self._primitive_label = tk.Label(
            side, textvariable=self.primitive_var, bg='#111', fg='#fff',
            font=('monospace', 18, 'bold'),
        )
        self._primitive_label.pack(fill='x', pady=(0, 8))

        tk.Label(side, text='STAGE', bg='#111', fg='#888',
                 font=('monospace', 9)).pack(fill='x')
        tk.Label(side, textvariable=self.stage_var, bg='#111', fg='#fff',
                 font=('monospace', 14, 'bold')).pack(fill='x', pady=(0, 8))

        tk.Label(side, text='STEP', bg='#111', fg='#888',
                 font=('monospace', 9)).pack(fill='x')
        tk.Label(side, textvariable=self.step_var, bg='#111', fg='#9ad',
                 font=('monospace', 12)).pack(fill='x', pady=(0, 8))

        tk.Label(side, text='INSTRUCTION', bg='#111', fg='#888',
                 font=('monospace', 9)).pack(fill='x')
        tk.Label(side, textvariable=self.instruction_var, bg='#111',
                 fg='#cdf', font=('monospace', 9), wraplength=240,
                 justify='left', anchor='w').pack(fill='x', pady=(0, 8))

        tk.Label(side, text='drag to rotate · scroll to zoom',
                 bg='#111', fg='#555',
                 font=('monospace', 8)).pack(side='bottom', pady=8)

    def _on_close_request(self) -> None:
        # Don't actually destroy on user click — just hide. Otherwise the
        # next tick() blows up. Leave teardown to close().
        self.root.withdraw()
        self._closed = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_state(self, value_map: Any, ee_pos: Any,
                     target: Any, objects: Optional[List],
                     step: int, stage_idx: int, num_stages: int,
                     instruction: str = '',
                     primitive: Optional[str] = None) -> None:
        """Stash the latest state. Cheap — actual draw happens in tick()."""
        self._state = {
            'value_map': value_map,
            'ee_pos': ee_pos,
            'target': target,
            'objects': objects,
            'step': step,
            'stage_idx': stage_idx,
            'num_stages': num_stages,
            'instruction': instruction,
            'primitive': primitive,
        }

    def tick(self) -> None:
        """Re-render the figure and pump Tk events. Throttled by refresh_interval."""
        if self._closed:
            return
        self._tick_counter += 1
        if self._tick_counter % self.refresh_interval != 0:
            return
        if not self._state or self._state.get('value_map') is None:
            try:
                self.root.update()
            except tk.TclError:
                self._closed = True
            return

        try:
            self._render()
            self.canvas.draw_idle()
            self.root.update()
        except tk.TclError:
            # Window destroyed externally; stop trying.
            self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        state = self._state
        vm = state['value_map']
        stage_idx = state['stage_idx']
        num_stages = max(1, state['num_stages'])

        # One-time axis bounds based on the value map's workspace.
        if not self._axes_initialized:
            self._setup_axes(vm)
            self._axes_initialized = True

        stage_changed = stage_idx != self._last_stage_idx
        self._last_stage_idx = stage_idx

        # Affordance / avoidance / target rebuild only on stage change
        # (voxel set is fixed per stage).
        if stage_changed:
            self._rebuild_voxel_artists(vm)

        self._update_gripper(state.get('ee_pos'))
        self._update_obbs(vm, state.get('objects'))

        self.stage_var.set(f"{stage_idx + 1} / {num_stages}")
        self.step_var.set(f"step {state.get('step', 0)}")
        self.instruction_var.set(state.get('instruction') or '')

        primitive = state.get('primitive')
        if primitive:
            self.primitive_var.set(primitive.upper())
            self._primitive_label.configure(
                fg=_PRIMITIVE_COLORS.get(primitive, '#fff')
            )
        else:
            self.primitive_var.set('—')
            self._primitive_label.configure(fg='#666')

    def _setup_axes(self, vm: Any) -> None:
        bmin = np.asarray(vm.workspace_bounds_min, dtype=np.float32)
        bmax = np.asarray(vm.workspace_bounds_max, dtype=np.float32)
        pad = 0.10 * (bmax - bmin)
        self.ax.set_xlim(bmin[0] - pad[0], bmax[0] + pad[0])
        self.ax.set_ylim(bmin[1] - pad[1], bmax[1] + pad[1])
        self.ax.set_zlim(bmin[2] - pad[2], bmax[2] + pad[2])
        # Equal aspect-ish: scale by extent.
        try:
            self.ax.set_box_aspect(tuple((bmax - bmin).tolist()))
        except Exception:
            pass

    def _voxel_world_coords(self, mask: np.ndarray, vm: Any,
                            ds: int = 1) -> np.ndarray:
        """Turn a (M,M,M) bool mask into (N, 3) world coords."""
        if ds > 1:
            mask = mask[::ds, ::ds, ::ds]
        idx = np.argwhere(mask)
        if idx.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        idx_full = idx.astype(np.float32) * ds
        return voxel2pc(idx_full, vm.workspace_bounds_min,
                        vm.workspace_bounds_max, vm.map_size)

    def _rebuild_voxel_artists(self, vm: Any) -> None:
        # Clear previous voxel artists.
        for handle_attr in ('_aff_scatter', '_avoid_scatter', '_target_scatter'):
            artist = getattr(self, handle_attr)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, handle_attr, None)

        ds = self.downsample
        thresh = self.point_threshold

        # Affordance — high-value voxels (greens).
        aff = vm.affordance
        if aff is not None and aff.max() > 0:
            aff_ds = aff[::ds, ::ds, ::ds]
            mask = aff_ds > thresh
            if mask.any():
                idx = np.argwhere(mask).astype(np.float32) * ds
                pts = voxel2pc(idx, vm.workspace_bounds_min,
                               vm.workspace_bounds_max, vm.map_size)
                vals = aff_ds[mask]
                self._aff_scatter = self.ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    c=vals, cmap='Greens', vmin=0.0, vmax=float(aff_ds.max()),
                    s=8, alpha=0.35, depthshade=False, edgecolors='none',
                )

        # Avoidance — soft penalty regions (reds).
        avoid = getattr(vm, 'avoidance', None)
        if avoid is not None and avoid.max() > 0:
            avoid_ds = avoid[::ds, ::ds, ::ds]
            mask = avoid_ds > thresh
            if mask.any():
                idx = np.argwhere(mask).astype(np.float32) * ds
                pts = voxel2pc(idx, vm.workspace_bounds_min,
                               vm.workspace_bounds_max, vm.map_size)
                vals = avoid_ds[mask]
                self._avoid_scatter = self.ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    c=vals, cmap='Reds', vmin=0.0, vmax=float(avoid_ds.max()),
                    s=8, alpha=0.30, depthshade=False, edgecolors='none',
                )

        # Original LLM-set target voxels (sparse, pre-smoothing).
        raw = getattr(vm, '_raw_affordance', None)
        if raw is not None and raw.max() > 0:
            target_pts = self._voxel_world_coords(raw > 0, vm, ds=1)
        elif aff is not None and aff.max() > 0:
            target_pts = self._voxel_world_coords(aff >= aff.max() * 0.95, vm, ds=1)
        else:
            target_pts = np.empty((0, 3), dtype=np.float32)
        if target_pts.shape[0] > 0:
            self._target_scatter = self.ax.scatter(
                target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
                c='#33ff66', s=18, alpha=0.9, depthshade=False,
                edgecolors='black', linewidths=0.3,
            )

    def _update_gripper(self, ee_pos: Any) -> None:
        if ee_pos is None:
            return
        ee = np.asarray(ee_pos, dtype=np.float32).reshape(-1)[:3]
        if self._gripper_scatter is None:
            self._gripper_scatter = self.ax.scatter(
                [ee[0]], [ee[1]], [ee[2]],
                c='#3399ff', s=80, alpha=1.0, depthshade=False,
                edgecolors='white', linewidths=1.0,
            )
        else:
            # mpl 3D scatter exposes positions through _offsets3d.
            self._gripper_scatter._offsets3d = (
                np.array([ee[0]]), np.array([ee[1]]), np.array([ee[2]]),
            )

    def _update_obbs(self, vm: Any, objects: Optional[List]) -> None:
        # Tear down previous OBB artists each tick — object set + poses
        # change as the scene moves, and counts are small.
        if self._obb_collection is not None:
            try:
                self._obb_collection.remove()
            except Exception:
                pass
            self._obb_collection = None
        for txt in self._obb_text_artists:
            try:
                txt.remove()
            except Exception:
                pass
        self._obb_text_artists = []

        if not objects:
            return

        all_segments: list = []
        all_colors: list = []
        for obj in objects:
            name = obj.get('name', '?')
            if name == 'switch':
                continue  # hidden; light_switch overlays it
            obb_center = obj.get('obb_center_world')
            obb_size = obj.get('obb_size')
            obb_rot = obj.get('obb_rotation')
            pos_world = obj.get('_position_world')

            if obb_center is not None and obb_size is not None and obb_rot is not None:
                corners = obb_world_corners(
                    np.asarray(obb_center, dtype=np.float32),
                    np.asarray(obb_size, dtype=np.float32),
                    np.asarray(obb_rot, dtype=np.float32),
                )
            elif obj.get('aabb') is not None:
                aabb_world = voxel2pc(
                    np.array(obj['aabb'], dtype=np.float32),
                    vm.workspace_bounds_min, vm.workspace_bounds_max,
                    vm.map_size,
                )
                bmin, bmax = aabb_world[0], aabb_world[1]
                corners = obb_world_corners(
                    (bmin + bmax) / 2.0, (bmax - bmin), np.eye(3),
                )
            else:
                continue

            color = _color_for_object(name)
            for i, j in _OBB_EDGES:
                all_segments.append([corners[i].tolist(), corners[j].tolist()])
                all_colors.append(color)

            if pos_world is not None:
                pw = np.asarray(pos_world, dtype=np.float32).reshape(-1)[:3]
                self._obb_text_artists.append(self.ax.text(
                    pw[0], pw[1], pw[2], name, color=color, fontsize=7,
                ))

        if all_segments:
            self._obb_collection = Line3DCollection(
                all_segments, colors=all_colors, linewidths=1.4,
            )
            self.ax.add_collection3d(self._obb_collection)
