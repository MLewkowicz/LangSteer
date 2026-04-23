"""Live 3D costmap visualization via a background Dash server.

Serves the current ValueMap as an interactive Plotly 3D figure at
http://127.0.0.1:<port>. The browser polls the server at a configurable
interval and updates the figure in place — the user's camera rotation
and zoom are preserved across updates via the Plotly `uirevision` property.

The experiment loop pushes the latest state via `update_state()`; the
Dash callback reads it on the interval tick and renders a fresh figure.
"""

import logging
import threading
from typing import Any, List, Optional

import plotly.graph_objects as go

from voxposer.visualizer import ValueMapVisualizer

logger = logging.getLogger(__name__)


class CostmapDashServer:
    """Background Dash server for live interactive 3D costmap visualization."""

    def __init__(self, port: int = 8050, refresh_s: float = 2.0,
                 viz_config: Optional[dict] = None):
        self._port = port
        self._refresh_ms = max(200, int(refresh_s * 1000))

        self._lock = threading.Lock()
        self._state: dict = {}
        self._thread: Optional[threading.Thread] = None
        self._started = False

        # Dedicated visualizer with no save_dir (in-memory figures only).
        vz_cfg = {
            'visualization_save_dir': None,
            'visualization_quality': (viz_config or {}).get(
                'visualization_quality', 'low'
            ),
        }
        self._visualizer = ValueMapVisualizer(vz_cfg)

        self._app = self._build_app()

    def _build_app(self):
        import dash
        from dash import Input, Output, dcc, html

        app = dash.Dash(__name__, title='LangSteer Costmap')
        app.layout = html.Div(
            style={'fontFamily': 'monospace', 'padding': '8px'},
            children=[
                html.Div(id='status', style={'fontSize': '14px'}),
                dcc.Graph(
                    id='costmap-3d',
                    style={'height': '88vh'},
                    config={'displayModeBar': True, 'scrollZoom': True},
                ),
                dcc.Interval(id='tick', interval=self._refresh_ms),
            ],
        )

        @app.callback(
            [Output('costmap-3d', 'figure'), Output('status', 'children')],
            [Input('tick', 'n_intervals')],
        )
        def _update(_n):
            with self._lock:
                state = dict(self._state)

            if not state or state.get('value_map') is None:
                fig = go.Figure()
                fig.update_layout(uirevision='costmap')
                return fig, 'waiting for costmap state...'

            fig = self._visualizer.visualize(
                state['value_map'],
                ee_pos_world=state.get('ee_pos'),
                objects=state.get('objects'),
                save=False, show=False,
            )
            # Preserve camera rotation/zoom across updates.
            fig.update_layout(uirevision='costmap')
            status = (
                f"step {state.get('step', 0)}  |  "
                f"stage {state.get('stage_idx', 0) + 1}/"
                f"{state.get('num_stages', 1)}  |  "
                f"refresh {self._refresh_ms}ms"
            )
            return fig, status

        return app

    def start(self):
        """Launch the Dash server in a daemon thread."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run, name='CostmapDashServer', daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Costmap Dash server started → http://127.0.0.1:{self._port}"
        )

    def _run(self):
        # Silence Flask's per-request logging (otherwise each tick logs a GET).
        werkzeug_log = logging.getLogger('werkzeug')
        werkzeug_log.setLevel(logging.ERROR)
        try:
            self._app.run(
                host='127.0.0.1', port=self._port,
                debug=False, use_reloader=False,
            )
        except Exception as e:
            logger.error(f"Dash server crashed: {e}")

    def update_state(self, value_map: Any, ee_pos: Any,
                     target: Any, objects: Optional[List],
                     step: int, stage_idx: int, num_stages: int):
        """Push the latest costmap state to the server.

        Thread-safe; called from the main experiment loop.
        """
        with self._lock:
            self._state = {
                'value_map': value_map,
                'ee_pos': ee_pos,
                'target': target,
                'objects': objects,
                'step': step,
                'stage_idx': stage_idx,
                'num_stages': num_stages,
            }
