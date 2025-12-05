import os
import sys
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import numpy as np

# When executed as a script, relative imports can fail because the package
# context is not set. Ensure the parent directory of `app_divided` is on
# sys.path so absolute imports below succeed when running the file directly.
_HERE = os.path.abspath(os.path.dirname(__file__))
_APP_DIV = os.path.abspath(os.path.join(_HERE, '..'))
_PKG_BASE = os.path.abspath(os.path.join(_APP_DIV, '..'))
if _PKG_BASE not in sys.path:
    sys.path.insert(0, _PKG_BASE)

# Import the modular pieces using absolute imports from the local `app_divided` package
from app_divided.data_loader import load_data
from flask import jsonify, request
import json
"""Simple Dash app that composes the modular data/dendro/viz pieces.

This module intentionally keeps coordinate computation (dendro) and
rendering (viz) separate. For a D3 migration, call the dendro functions
to obtain a JSON-friendly `model` (icoord/dcoord/nodes) and serve that
from an API endpoint; the client can render with D3 independently.
"""

from app_divided.dendro.linkage import get_dendrogram_segments2, compute_dendrogram_coords
from app_divided.viz.plotly_viz import get_dr_figure, plot_dendrogram_plotly


def create_app(base_dir=None):
    """Create a Dash app bound to data discovered under `base_dir`.

    The returned app wires a DR scatter plot to a dendrogram plot. The
    dendrogram callback uses `dendro` to compute coordinates and then
    `viz` to render segments — the two phases are separated so the
    coordinate model can be serialized for a D3 client.
    """

    data = load_data(base_dir=base_dir)
    DR_DATA = data.get('DR_DUMMY_DATA', [])
    POINT_ID_MAP = data.get('POINT_ID_MAP', {})
    Z = data.get('Z', None)
    NODE_ID_MAP = data.get('NODE_ID_MAP', {})

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Full layout (control panel + DR + dendrogram + detail) adapted from original app
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Control Panel", className="text-center mb-3"),
                        dbc.Row([
                            dbc.Label("Dataset:", html_for="dataset-selector", className="fw-bold"),
                            dcc.Dropdown(
                                id='dataset-selector',
                                options=[
                                    {'label': 'Iris', 'value': 'iris'},
                                    {'label': 'Digits', 'value': 'digits'},
                                    {'label': 'Wine', 'value': 'wine'}
                                ],
                                value='iris'
                            )
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("DR Method:", className="fw-bold"),
                            dbc.RadioItems(
                                id='dr-method-selector',
                                options=[
                                    {'label': 'UMAP', 'value': 'UMAP'},
                                    {'label': 'TSNE', 'value': 'TSNE'},
                                    {'label': 'PCA', 'value': 'PCA'}
                                ],
                                value='UMAP'
                            )
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("Parameters:", className="fw-bold"),
                            html.Div(id='parameter-settings', className="flex-grow-1 overflow-auto")
                        ], className="mb-3 d-flex flex-column"),
                        dbc.Button('Run Analysis', id='execute-button', n_clicks=0, color="primary", size="lg", className="w-100 mt-auto")
                    ])
                ], className="h-100 d-flex flex-column"),
            ], width=2, className="p-2 h-100"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("DR Visualization", className="text-center mb-1"),
                        dbc.RadioItems(
                            id='dr-interaction-mode-toggle',
                            options=[
                                {'label': 'Brush Selection', 'value': 'brush'},
                                {'label': 'Zoom/Pan', 'value': 'zoom'}
                            ],
                            value='zoom',
                            inline=True,
                            className="mb-2 text-center"
                        ),
                        dcc.Graph(id='dr-visualization-plot', className="flex-grow-1")
                    ], className="d-flex flex-column p-3 h-100")
                ], style={'height': 'calc(100vh - 40px)'}, className="h-100"),
            ], width=4, className="p-2 h-100"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Cluster Dendrogram", className="text-center mb-1"),
                        dbc.Row([
                            dbc.Col([
                                dbc.RadioItems(
                                    id='dendro-interaction-mode-toggle',
                                    options=[
                                        {'label': 'Node Selection', 'value': 'node'},
                                        {'label': 'Zoom/Pan', 'value': 'zoom'}
                                    ],
                                    value='node',
                                    inline=True
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Checklist(
                                    id='dendro-width-option-toggle',
                                    options=[{'label': 'Proportional Width', 'value': 'prop_width'}],
                                    value=[],
                                )
                            ], width=6)
                        ], className="mb-2"),
                        dcc.Graph(id='dendrogram-plot', className="flex-grow-1")
                    ], className="d-flex flex-column p-3 h-100")
                ], style={'height': 'calc(100vh - 40px)'}, className="h-100"),
            ], width=4, className="p-2 h-100"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Detail & Info", className="text-center mb-3"),
                        dbc.Tabs(id='detail-info-tabs', active_tab='tab-point-details', children=[
                            dbc.Tab(label='Point Details', tab_id='tab-point-details'),
                            dbc.Tab(label='Selection Stats', tab_id='tab-selection-stats'),
                            dbc.Tab(label='System Log', tab_id='tab-system-log')
                        ]),
                        html.Div(id='detail-panel-content', className="mt-3 flex-grow-1 overflow-auto")
                    ], className="d-flex flex-column p-3 h-100")
                ], style={'height': 'calc(100vh - 40px)'}, className="h-100"),
            ], width=2, className="p-2 h-100")
        ], className="g-0 h-100")
    ], fluid=True, className="h-100")


    @app.callback(Output('dr-visualization-plot', 'figure'), Input('dr-visualization-plot', 'id'))
    def update_dr_plot(_):
        return get_dr_figure(DR_DATA, interaction_mode='zoom')


    @app.callback(Output('dendrogram-plot', 'figure'), Input('dr-visualization-plot', 'selectedData'))
    def update_dendrogram(selectedData):
        try:
            if Z is None:
                raise ValueError('No linkage matrix loaded')
            Z_arr = np.array(Z)
            # Build a coordinate model first (icoord,dcoord,leaf_order,nodes) if you
            # want to expose JSON to a D3 client. Here we use the compatibility
            # wrapper to get segments for Plotly rendering:
            segments = get_dendrogram_segments2(Z_arr[:, [0, 1, 2, 3]])
            is_selecteds = [False] * len(segments)
            if selectedData and 'points' in selectedData and selectedData['points']:
                sel_point_indices = []
                for p in selectedData['points']:
                    if 'customdata' in p and p['customdata']:
                        try:
                            sel_point_indices.append(int(p['customdata'][0]))
                        except Exception:
                            if 'pointIndex' in p:
                                sel_point_indices.append(int(p['pointIndex']))
                    elif 'pointIndex' in p:
                        sel_point_indices.append(int(p['pointIndex']))

                selected_cluster_ids = []
                for pid in sel_point_indices:
                    if pid in POINT_ID_MAP:
                        selected_cluster_ids.append(POINT_ID_MAP[pid])

                node_map = NODE_ID_MAP or {}
                n_points = Z_arr.shape[0] + 1
                for cid in selected_cluster_ids:
                    if cid in node_map and node_map[cid] is not None:
                        mapped = node_map[cid]
                        try:
                            if isinstance(mapped, int) and 0 <= mapped < (len(segments) // 3):
                                seg_idx = 3 * int(mapped)
                                is_selecteds[seg_idx:seg_idx+3] = [True, True, True]
                                continue
                        except Exception:
                            pass
                        try:
                            i = int(mapped) - n_points
                            if 0 <= i < (len(segments) // 3):
                                seg_idx = 3 * i
                                is_selecteds[seg_idx:seg_idx+3] = [True, True, True]
                                continue
                        except Exception:
                            pass
                        try:
                            z_parents = Z_arr[:, 1]
                            matches = np.where(z_parents == cid)[0]
                            for j in matches:
                                seg_idx = 3 * int(j)
                                if seg_idx + 2 < len(is_selecteds):
                                    is_selecteds[seg_idx:seg_idx+3] = [True, True, True]
                        except Exception:
                            pass

            # Render using the viz module. Note: for a D3 server, you would
            # instead call `compute_dendrogram_coords(...)` and return the
            # resulting model via an API endpoint (JSON), leaving rendering
            # to the browser-side D3 code.
            fig = plot_dendrogram_plotly(segments, is_selecteds=is_selecteds)
        except Exception:
            # very small fallback
            fig = plot_dendrogram_plotly([[(0, 0), (1, 1)], [(1, 1), (2, 0)]])
        return fig

    return app


if __name__ == '__main__':
    app = create_app()
    # Add JSON API endpoints on the Flask server for D3 clients
    @app.server.route('/api/points')
    def api_points():
        data = load_data()
        dr = data.get('DR_DUMMY_DATA', [])
        return jsonify({'points': dr})

    @app.server.route('/api/tree')
    def api_tree():
        data = load_data()
        Z = data.get('Z')
        if Z is None:
            return jsonify({'error': 'no_linkage'}), 404
        Z_arr = np.array(Z)
        n_points = Z_arr.shape[0] + 1
        icoord, dcoord, leaf_order, nodes = compute_dendrogram_coords(Z_arr[:, [0,1,2,3]], n_points)
        # nodes may contain numpy types; convert to native Python types
        def _clean(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            return obj

        model = {'icoord': _clean(icoord), 'dcoord': _clean(dcoord), 'leaf_order': _clean(leaf_order), 'nodes': _clean(nodes)}
        return app.response_class(json.dumps(model), mimetype='application/json')

    app.run_server(debug=True, port=8055)
