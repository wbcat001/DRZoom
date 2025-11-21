import random
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# 再現性確保のためシード固定
random.seed(42)
np.random.seed(42)

# DBCが利用可能かチェック（Canvas環境では通常利用可能）
DBC_AVAILABLE = True

# ダミーデータ生成
def generate_dr_dummy_data(n=100):
    """DR View用ダミーデータ生成"""
    data = []
    labels = ['A', 'B', 'C']
    for i in range(n):
        data.append({
            'x': random.uniform(-5, 5),
            'y': random.uniform(-5, 5),
            'label': random.choice(labels),
            'id': i
        })
    return data


def generate_dendrogram_dummy_data():
    """Dendrogram用ダミーデータ生成"""
    # 簡単な階層構造データ (葉 + 内部ノード)
    nodes = []
    # リーフノード (データポイント)
    for i in range(10):
        nodes.append({
            'x': i,
            'y': 0,
            'size': 1,
            'id': i,
            'type': 'leaf',
            'parent': None
        })

    # 内部ノード (手動で親情報をセット)
    internal_nodes = [
        {'x': 1, 'y': 1, 'size': 3, 'id': 10, 'type': 'internal', 'children': [0,1,2]},
        {'x': 4, 'y': 1, 'size': 2, 'id': 11, 'type': 'internal', 'children': [3,4]},
        {'x': 7, 'y': 1, 'size': 4, 'id': 12, 'type': 'internal', 'children': [5,6,7,8]},
        {'x': 2.5, 'y': 2, 'size': 5, 'id': 13, 'type': 'internal', 'children': [10,11]},
        {'x': 5.5, 'y': 2, 'size': 6, 'id': 14, 'type': 'internal', 'children': [12,9]},
        {'x': 4, 'y': 3, 'size': 11, 'id': 15, 'type': 'root', 'children': [13,14]}
    ]

    # set parent links for leaves and internal nodes
    id_to_node = {n['id']: n for n in nodes}
    for inode in internal_nodes:
        id_to_node[inode['id']] = inode
    for inode in internal_nodes:
        for c in inode['children']:
            id_to_node[c]['parent'] = inode['id']

    nodes.extend(internal_nodes)
    return nodes

# ダミーデータ生成
DR_DUMMY_DATA = generate_dr_dummy_data(100)
DENDROGRAM_DUMMY_DATA = generate_dendrogram_dummy_data()

# -----------------------------
# Dendrogram helper functions (moved early so they are available during import)
# -----------------------------

def recurse_leaf_dfs(cluster_tree, current_node):
    child1 = cluster_tree[cluster_tree[:,2] == current_node][:,0]
    child2 = cluster_tree[cluster_tree[:,2] == current_node][:,1]
    if len(child1) == 0 and len(child2) == 0:
        return [current_node,]
    else:
        return sum([recurse_leaf_dfs(cluster_tree, child) for child in np.concatenate((child1, child2))], [])


def get_leaves(cluster_tree):
    """
    cluster_tree: (u, v, lambda_val, child_size, parent)
    """
    root = cluster_tree[:, 2].max()
    return recurse_leaf_dfs(cluster_tree, root)


def _get_leaves(condensed_tree):
    """Robust leaf extractor that accepts DataFrame, array, or list-of-dicts."""
    try:
        if isinstance(condensed_tree, pd.DataFrame):
            arr = condensed_tree.values
            return get_leaves(arr)
        else:
            arr = np.array(condensed_tree)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return get_leaves(arr)
    except Exception:
        pass
    # fallback for list-of-dicts representation
    try:
        if isinstance(condensed_tree, (list, tuple)) and len(condensed_tree) > 0 and isinstance(condensed_tree[0], dict):
            children = [int(r.get('child')) for r in condensed_tree if 'child' in r]
            parents = [int(r.get('parent')) for r in condensed_tree if 'parent' in r]
            leaves = [c for c in children if c not in parents]
            return leaves
    except Exception:
        pass
    return []


def get_linkage_matrix_from_hdbscan(condensed_tree):
    """
    (child1, child2, parent, lambda_val, count)
    """
    linkage_matrix = []
    raw_tree = condensed_tree._raw_tree
    condensed_tree = condensed_tree.to_pandas()
    cluster_tree = condensed_tree[condensed_tree['child_size'] > 1]
    sorted_condensed_tree = cluster_tree.sort_values(by=['lambda_val','parent'], ascending=True)
    print(f"len of sorted condensed tree: {len(sorted_condensed_tree)}")

    for i in range(0, len(sorted_condensed_tree), 2):

        # 偶数行（i）と次の奇数行（i+1）をペアとして取得
        if i + 1 < len(sorted_condensed_tree):

            row_a = sorted_condensed_tree.iloc[i]
            row_b = sorted_condensed_tree.iloc[i+1]

            # **前提チェック**: lambda_valが同じであることを確認
            if row_a['lambda_val'] != row_b['lambda_val']:
                # lambda_valが異なる場合は、次の処理に進む（結合の前提が崩れる）
                raise ValueError(f"Lambda value mismatch at rows {i} and {i+1}: {row_a['lambda_val']} vs {row_b['lambda_val']}")

            # Parent IDが同じであることを確認 (同じ結合の結果である可能性が高い)
            if row_a['parent'] != row_b['parent']:
                # Parent IDが異なる場合は、このペアは単一の結合ではない可能性が高い
                raise ValueError(f"Parent ID mismatch at rows {i} and {i+1}: {row_a['parent']} vs {row_b['parent']}")

            child_a = row_a['child']
            child_b = row_b['child']
            lam = row_a['lambda_val']

            total_size = raw_tree[raw_tree['child'] == row_a['parent']]['child_size']
            if len(total_size) == 0:
                total_size = row_a['child_size'] + row_b['child_size']
            else:
                total_size = total_size[0]
            parent_id = row_a['parent']

            linkage_matrix.append([
                int(child_a), 
                int(child_b), 
                int(parent_id),
                lam, 
                total_size,
        ])   
    print(f"len of linkage matrix: {len(linkage_matrix)}")


    # --- 葉ノードをデータフレームから確実に抽出して 0..N-1 にリマップ ---
    node_id_map = {}
    current_id = 0
    # Try to extract leaves from condensed_tree DataFrame (child_size==1)
    try:
        if 'child_size' in condensed_tree.columns:
            leaf_vals = condensed_tree[condensed_tree['child_size'] == 1]['child'].unique().tolist()
        else:
            # fallback to any child values not present in parents
            childs = condensed_tree['child'].unique().tolist()
            parents = condensed_tree['parent'].unique().tolist() if 'parent' in condensed_tree.columns else []
            leaf_vals = [c for c in childs if c not in parents]
    except Exception:
        leaf_vals = _get_leaves(raw_tree)

    print(f"Number of leaves: {len(leaf_vals)}")
    for leaf in leaf_vals:
        node_id_map[int(leaf)] = current_id
        current_id += 1

    print(f"Leaf ID Map Size: {len(node_id_map)}")
    print(f"current id: {current_id}")

    # Ensure every id appearing in linkage_matrix has a mapping (leaves already mapped)
    try:
        all_ids = set()
        for row in linkage_matrix:
            all_ids.add(int(row[0]))
            all_ids.add(int(row[1]))
            all_ids.add(int(row[2]))
        remaining = sorted([a for a in all_ids if a not in node_id_map])
        for aid in remaining:
            node_id_map[aid] = current_id
            current_id += 1
    except Exception:
        # fallback: leave node_id_map as-is
        pass
    print(f"Total Node ID Map Size: {len(node_id_map)}")
    print(f"current_id: {current_id}")

    # linkage matrixを書き換え
    max_lambda = max(row[3] for row in linkage_matrix)
    print(f"Max Lambda: {max_lambda}")
    linkage_matrix_mapped = [ 
        [node_id_map[row[0]], node_id_map[row[1]], node_id_map[row[2]], max_lambda - row[3], row[4]] 
        for row in linkage_matrix.__reversed__()
    ]

    return np.array(linkage_matrix_mapped), node_id_map # linkage matrix, parentid -> newid

# -----------------------------
# Try to load real data (embedding + condensed_tree) if available
# -----------------------------
import os
import glob
import pickle

# search base: try multiple candidate locations for 18_rapids/result
_candidates = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '18_rapids', 'result')),  # src/experiments/19_tree/../18_rapids/result
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '18_rapids', 'result')),  # src/experiments/19_tree/../../18_rapids/result
    os.path.abspath(os.path.join(os.getcwd(), 'src', 'experiments', '18_rapids', 'result')),     # workspace/src/experiments/18_rapids/result
    os.path.abspath(os.path.join(os.getcwd(), 'src', '18_rapids', 'result')),                  # workspace/src/18_rapids/result
]
_base_dir = None
for c in _candidates:
    if os.path.isdir(c):
        _base_dir = c
        break
if _base_dir is None:
    _base_dir = _candidates[0]
    # directory may not exist; loader will handle missing paths
EMBEDDING = None
LABELS = None
POINT_ID_MAP = {}
Z = None

def _find_file(pattern):
    if not os.path.isdir(_base_dir):
        return None
    matches = glob.glob(os.path.join(_base_dir, '**', pattern), recursive=True)
    return matches[0] if matches else None

# try embedding.npz
_emb_path = _find_file('embedding.npz')
if _emb_path:
    try:
        npz = np.load(_emb_path)
        if 'embedding' in npz:
            EMBEDDING = npz['embedding']
        else:
            # fallback: first array
            EMBEDDING = npz[npz.files[0]]
        # try to load labels in same folder
        _dir = os.path.dirname(_emb_path)
        _data_path = os.path.join(_dir, 'data.npz')
        if os.path.exists(_data_path):
            dnp = np.load(_data_path)
            LABELS = dnp[dnp.files[0]] if dnp.files else None
        else:
            LABELS = None
        # build DR data list
        if EMBEDDING is not None:
            n = EMBEDDING.shape[0]
            # ensure at least 2 dims
            if EMBEDDING.shape[1] < 2:
                # pad
                EMBEDDING = np.concatenate([EMBEDDING, np.zeros((n, 2 - EMBEDDING.shape[1]))], axis=1)
            DR_DUMMY_DATA = []
            for i in range(n):
                # Safely convert label to short string; avoid converting large arrays to string
                if LABELS is not None:
                    try:
                        lbl = LABELS[i]
                        # if lbl is scalar-like, use it; otherwise fallback to index
                        if np.shape(lbl) == ():
                            label_str = str(lbl)
                        else:
                            label_str = str(i)
                    except Exception:
                        label_str = str(i)
                else:
                    label_str = str(i)
                DR_DUMMY_DATA.append({'x': float(EMBEDDING[i, 0]), 'y': float(EMBEDDING[i, 1]), 'label': label_str, 'id': int(i)})
            print(f"Loaded embedding from {_emb_path}, n={n}")
    except Exception as e:
        print(f"Failed loading embedding: {e}")

# try condensed_tree pickle
_pkl_path = _find_file('*condensed*tree*.pkl') or _find_file('condensed_tree_object.pkl')
if _pkl_path:
    try:
        with open(_pkl_path, 'rb') as f:
            obj = pickle.load(f)
        # obj may be clusterer or condensed_tree
        condensed = obj.condensed_tree_ if hasattr(obj, 'condensed_tree_') else obj

        # try to get pandas representation
        if hasattr(condensed, 'to_pandas'):
            df = condensed.to_pandas()
        elif hasattr(condensed, '_raw_tree'):
            raw = condensed._raw_tree
            try:
                df = pd.DataFrame(raw)
            except Exception:
                df = None
        else:
            df = None

        if df is not None and 'child' in df.columns and 'parent' in df.columns:
            # leaf rows typically have child_size == 1
            if 'child_size' in df.columns:
                leaf_rows = df[df['child_size'] == 1]
            else:
                leaf_rows = df
            POINT_ID_MAP = {int(r['child']): int(r['parent']) for _, r in leaf_rows.iterrows()}
            print(f"Loaded condensed tree from {_pkl_path}, point_id_map size={len(POINT_ID_MAP)}")

            # try to build a canonical linkage matrix Z using notebook conversion
            try:
                Z_mat, node_id_map = get_linkage_matrix_from_hdbscan(condensed)
                globals()['Z'] = Z_mat
                globals()['NODE_ID_MAP'] = node_id_map
                print(f"Built Z using get_linkage_matrix_from_hdbscan, shape={Z_mat.shape}, node_id_map size={len(node_id_map)}")
            except Exception as e:
                print(f"get_linkage_matrix_from_hdbscan failed: {e}. Falling back to approximate Z builder.")
                # fallback: create linkage list from df rows where 'child_size' in columns
                linkage_rows = []
                if 'lambda_val' in df.columns and 'child_size' in df.columns:
                    merge_df = df[df['child_size'] > 1]
                    node_id_map = {}
                    linkage_rows = []
                    for idx, (_, row) in enumerate(merge_df.iterrows()):
                        parent_orig = int(row['parent'])
                        child = int(row['child'])
                        lam = float(row['lambda_val']) if 'lambda_val' in row else 0.0
                        csize = int(row['child_size']) if 'child_size' in row else 1
                        # store merge index (fallback)
                        node_id_map[parent_orig] = idx
                        linkage_rows.append([child, parent_orig, lam, csize])
                    if linkage_rows:
                        Z = np.array(linkage_rows)
                        globals()['Z'] = Z
                        globals()['NODE_ID_MAP'] = node_id_map  # parent_orig -> merge_index
                        print(f"Built approximate Z from condensed tree, shape={Z.shape}, node_id_map size={len(node_id_map)}")
    except Exception as e:
        print(f"Failed loading condensed tree pickle: {e}")

# helper functions defined earlier (moved up)


def compute_dendrogram_coords(Z, n_points):
    """
    Linkage Matrixからデンドログラム描画用の座標を計算
    Z: (n_merges x 4) array like [c1, c2, dist, count]
    n_points: 葉の数

    Returns: icoord, dcoord, leaf_order
    """
    # --- 1. ノード情報の準備 ---
    n_nodes = 2 * n_points - 1
    nodes = [{'x': None, 'y': 0.0, 'size': 1, 'left': None, 'right': None} for _ in range(n_points)]

    # Z の各行は c1, c2, dist, count
    for i in range(n_points - 1):
        c1, c2, dist, count = Z[i]
        nodes.append({
            'x': None,
            'y': float(dist),
            'size': int(count),
            'left': int(c1),
            'right': int(c2)
        })

    def get_leaf_order_sorted(node_idx):
        node = nodes[node_idx]
        if node_idx < n_points:
            return [node_idx]
        C1_idx, C2_idx = node['left'], node['right']
        size_C1, size_C2 = nodes[C1_idx]['size'], nodes[C2_idx]['size']
        if size_C1 < size_C2:
            C1_idx, C2_idx = C2_idx, C1_idx
        order_left = get_leaf_order_sorted(C1_idx)
        order_right = get_leaf_order_sorted(C2_idx)
        return order_left + order_right

    def calculate_x_coord(node_idx, leaf_to_x):
        node = nodes[node_idx]
        if node_idx < n_points:
            x_coord = leaf_to_x[node_idx]
            node['x'] = x_coord
            return x_coord
        x_left = calculate_x_coord(node['left'], leaf_to_x)
        x_right = calculate_x_coord(node['right'], leaf_to_x)
        x_coord = (x_left + x_right) / 2.0
        node['x'] = x_coord
        return x_coord

    root_node_idx = n_points - 1 + (n_points - 1)
    leaf_order = get_leaf_order_sorted(root_node_idx)
    leaf_to_x = {leaf_idx: 2 * i + 1 for i, leaf_idx in enumerate(leaf_order)}
    calculate_x_coord(root_node_idx, leaf_to_x)

    icoord = []
    dcoord = []
    for i in range(n_points - 1):
        P = n_points + i
        C1 = nodes[P]['left']
        C2 = nodes[P]['right']
        y_P = nodes[P]['y']
        y_C1 = nodes[C1]['y']
        y_C2 = nodes[C2]['y']
        x_P = nodes[P]['x']
        x_C1 = nodes[C1]['x']
        x_C2 = nodes[C2]['x']
        icoord.append([x_C1, x_C1, x_C2, x_C2])
        dcoord.append([y_C1, y_P, y_P, y_C2])

    return icoord, dcoord, leaf_order


def get_dendrogram_segments2(Z: np.ndarray):
    """Linkage Matrixから描画用セグメントを生成。簡易実装。"""
    n_points = Z.shape[0] + 1
    icoord, dcoord, leaf_order = compute_dendrogram_coords(Z, n_points)
    segments = []
    for icoords, dcoords in zip(icoord, dcoord):
        x1, x2, x3, x4 = icoords
        y1, y2, y3, y4 = dcoords
        segments.append([(x1, y1), (x2, y2)])
        segments.append([(x2, y2), (x3, y3)])
        segments.append([(x4, y4), (x3, y3)])
    return segments


def plot_dendrogram_plotly(segments, colors=None, scores=None, is_selecteds=None):
    """Plotlyを使用したデンドログラムの描画（segmentsは線分タプルのリスト）"""
    fig = go.Figure()
    for i, seg in enumerate(segments):
        x_coords = [seg[0][0], seg[1][0]]
        y_coords = [seg[0][1], seg[1][1]]
        color = 'black' if colors is None else colors[i]
        info = 'N/A' if scores is None or i >= len(scores) else f"{scores[i]:.2f}"
        opacity = 1.0
        if is_selecteds is not None:
            if i < len(is_selecteds):
                opacity = 1.0 if is_selecteds[i] else 0.2
            else:
                opacity = 0.2
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color=color, width=1),
            showlegend=False,
            hoverinfo='text' if (is_selecteds is None or (i < len(is_selecteds) and is_selecteds[i])) else 'skip',
            text=f'Segment {i}: ({x_coords[0]:.2f},{y_coords[0]:.2f})→({x_coords[1]:.2f},{y_coords[1]:.2f}) score={info}',
            opacity=opacity
        ))
    fig.update_layout(
        title='Dendrogram (simple)',
        xaxis_title='Observation Index',
        yaxis_title='Distance / Height',
        hovermode='closest',
        height=800,
        width=1000,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig


def get_clusters_from_points(point_ids, point_id_map):
    """選択されたポイントIDから対応するクラスタIDリストを返す（重複除去）"""
    cluster_ids = set()
    for pid in point_ids:
        if pid in point_id_map:
            cluster_ids.add(point_id_map[pid])
    return list(cluster_ids)


def build_point_id_map_from_dummy(nodes):
    """ダミーノード構造から point_id_map を作る（leaf->parent mapping）"""
    mapping = {}
    for n in nodes:
        if n.get('type') == 'leaf':
            # parentがNoneの場合はleaf自身を返す
            mapping[n['id']] = n.get('parent', n['id'])
    return mapping

# -----------------------------
# Dashアプリケーション
# -----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# グラフ無限拡大を防ぐためのHTML/Bodyの高さ設定（省略してDashのデフォルトテンプレートを使う）

# レイアウト定義（DBC版 - グラフ無限拡大対策済み）
if DBC_AVAILABLE:
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
else:
    app.layout = html.Div([html.H3("Dash Bootstrap Components not available. Please install: pip install dash-bootstrap-components")])

# Callbacks
@app.callback(Output('parameter-settings', 'children'), Input('dr-method-selector', 'value'))
def update_parameter_settings(method):
    if method == 'UMAP':
        return html.Div([
            html.Label("n_neighbors:"),
            dcc.Slider(id='umap-n-neighbors', min=5, max=50, step=1, value=15, marks={i: str(i) for i in range(5, 51, 10)}, tooltip={'placement': 'bottom', 'always_visible': True}),
            html.Br(),
            html.Label("min_dist:"),
            dcc.Slider(id='umap-min-dist', min=0.0, max=0.99, step=0.01, value=0.1, marks={i/10: f"{i/10:.1f}" for i in range(0, 10, 2)}, tooltip={'placement': 'bottom', 'always_visible': True})
        ])
    elif method == 'TSNE':
        return html.Div([html.Label("perplexity:"), dcc.Slider(id='tsne-perplexity', min=5, max=50, step=1, value=30, marks={i: str(i) for i in range(5, 51, 10)}, tooltip={'placement': 'bottom', 'always_visible': True})])
    elif method == 'PCA':
        return html.Div([html.Label("n_components:"), dcc.Dropdown(id='pca-n-components', options=[{'label': '2', 'value': 2}, {'label': '3', 'value': 3}], value=2)])
    return html.Div()

@app.callback(Output('dr-visualization-plot', 'figure'), [Input('execute-button', 'n_clicks'), Input('dr-method-selector', 'value'), Input('dr-interaction-mode-toggle', 'value')])
def update_dr_plot(n_clicks, method, interaction_mode):
    df = pd.DataFrame(DR_DUMMY_DATA)
    fig = go.Figure()
    for label in df['label'].unique():
        df_group = df[df['label'] == label]
        custom_ids = df_group['id'].apply(lambda x: [x]).tolist()
        fig.add_trace(go.Scatter(x=df_group['x'], y=df_group['y'], mode='markers', name=label, customdata=custom_ids, marker=dict(size=8, opacity=0.8), hovertemplate=('Label: %{data.name}<br>ID: %{customdata[0]}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>')))
    fig.update_layout(showlegend=False, xaxis_title='Dimension 1', yaxis_title='Dimension 2', margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='#f8f9fa', paper_bgcolor='#f8f9fa')
    drag_mode = 'select' if interaction_mode == 'brush' else 'zoom'
    fig.update_layout(dragmode=drag_mode)
    return fig

@app.callback(
    Output('dendrogram-plot', 'figure'),
    [
        Input('dendro-width-option-toggle', 'value'),
        Input('dr-visualization-plot', 'selectedData')
    ]
)
def update_dendrogram_plot(width_options, selectedData):
    prop_width = 'prop_width' in width_options
    # If real linkage matrix Z exists, generate segments from it. Otherwise use a simple mock.
    try:
        # Attempt to find a variable Z in the module namespace (if user loaded it)
        Z = globals().get('Z', None)
        if Z is None:
            raise KeyError
        Z_arr = np.array(Z)
        segments = get_dendrogram_segments2(Z_arr[:, [0,1,2,3]])

        # prepare selection mask for segments (3 segments per merge)
        is_selecteds = [False] * len(segments)

        # If selection exists and mapping is available, compute highlighted segments
        if selectedData and 'points' in selectedData and selectedData['points']:
            # extract selected point indices
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

            # map to cluster parent ids using POINT_ID_MAP if available
            selected_cluster_ids = []
            for pid in sel_point_indices:
                if pid in POINT_ID_MAP:
                    selected_cluster_ids.append(POINT_ID_MAP[pid])

            # map to new node ids using NODE_ID_MAP
            node_map = globals().get('NODE_ID_MAP', {})
            n_points = Z_arr.shape[0] + 1
            for cid in selected_cluster_ids:
                if cid in node_map and node_map[cid] is not None:
                    mapped = node_map[cid]
                    # If mapping stores merge_index (fallback), use directly
                    try:
                        if isinstance(mapped, int) and 0 <= mapped < (len(segments) // 3):
                            seg_idx = 3 * int(mapped)
                            is_selecteds[seg_idx:seg_idx+3] = [True, True, True]
                            continue
                    except Exception:
                        pass
                    # Otherwise, attempt to interpret mapping as new_parent_id
                    try:
                        i = int(mapped) - n_points
                        if 0 <= i < (len(segments) // 3):
                            seg_idx = 3 * i
                            is_selecteds[seg_idx:seg_idx+3] = [True, True, True]
                            continue
                    except Exception:
                        pass
                    # Final fallback: search Z for parent equal to cid (works for some approximate Z layouts)
                    try:
                        z_parents = Z_arr[:, 1]
                        matches = np.where(z_parents == cid)[0]
                        for j in matches:
                            seg_idx = 3 * int(j)
                            if seg_idx + 2 < len(is_selecteds):
                                is_selecteds[seg_idx:seg_idx+3] = [True, True, True]
                    except Exception:
                        pass

        fig = plot_dendrogram_plotly(segments, is_selecteds=is_selecteds)
    except Exception:
        # Fallback: draw simple predefined connections using dummy nodes
        fig = go.Figure()
        connections = [
            ([0, 0, 1], [0, 1, 1]),
            ([1, 1, 2], [0, 1, 1]),
            ([3, 3, 4], [0, 1, 1]),
            ([6, 6, 7], [0, 1, 1]),
            ([7, 7, 8], [0, 1, 1]),
            ([8, 8, 9], [0, 1, 1]),
            ([1, 1, 3.5], [1, 2, 2]),
            ([7.5, 7.5, 8.5], [1, 2, 2]),
            ([2.25, 2.25, 8], [2, 3, 3])
        ]
        for x_coords, y_coords in connections:
            line_width = 3 if prop_width else 1
            fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines', line=dict(color='black', width=line_width), showlegend=False, hoverinfo='skip'))
        x_positions = list(range(10))
        y_positions = [0] * 10
        fig.add_trace(go.Scatter(x=x_positions, y=y_positions, mode='markers', marker=dict(size=8, color='blue'), showlegend=False, text=[f'Point {i}' for i in range(10)], hovertemplate='Point ID: %{text}<extra></extra>'))
        fig.update_layout(xaxis_title='Data Points', yaxis_title='Distance', showlegend=False, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='#f8f9fa', paper_bgcolor='#f8f9fa')
    return fig

@app.callback(Output('detail-panel-content', 'children'), [Input('detail-info-tabs', 'active_tab'), Input('dr-visualization-plot', 'clickData'), Input('dr-visualization-plot', 'selectedData')])
def update_detail_panel(active_tab, click_data, selected_data):
    if active_tab == 'tab-point-details':
        if click_data:
            point_id = click_data['points'][0]['customdata'][0]
            trace_index = click_data['points'][0]['curveNumber']
            label_mock = ['A', 'B', 'C'][trace_index % 3]
            return html.Div([html.H6(f"Selected Point Details (ID: {point_id})", className="fw-bold"), dbc.ListGroup([dbc.ListGroupItem(f"ID: {point_id}"), dbc.ListGroupItem(f"X: {click_data['points'][0]['x']:.3f}"), dbc.ListGroupItem(f"Y: {click_data['points'][0]['y']:.3f}"), dbc.ListGroupItem(f"Label: {label_mock}" )], flush=True), html.P("Neighbors (k=5): Mock data - Points 1, 5, 12, 23, 34", className="mt-3 small")])
        else:
            return html.Div([html.P("Click a point in the DR view to see details.")])
    elif active_tab == 'tab-selection-stats':
        if selected_data and selected_data['points']:
            n_selected = len(selected_data['points'])
            return html.Div([html.H6("Selection Statistics", className="fw-bold"), html.P(f"Selected Points: {n_selected}"), html.P("Feature Statistics (Mock):"), html.Ul([html.Li("Feature 1: Mean=0.45, Std=0.23"), html.Li("Feature 2: Mean=0.67, Std=0.19"), html.Li("Feature 3: Mean=0.33, Std=0.15")])])
        else:
            return html.Div([html.P("Select points in the DR view to see statistics.")])
    elif active_tab == 'tab-system-log':
        return html.Div([html.H6("System Log", className="fw-bold"), html.Div([html.P("2024-11-16 10:30:15 - Application started"), html.P("2024-11-16 10:30:20 - Dataset loaded: iris (150 samples)"), html.P("2024-11-16 10:30:25 - DR method: UMAP initialized"), html.P("2024-11-16 10:30:30 - Analysis completed (Mock execution)")], style={'fontSize': '12px', 'fontFamily': 'monospace'})])
    return html.Div()

@app.callback(Output('detail-info-tabs', 'style'), Input('execute-button', 'n_clicks'))
def log_execute_button(n_clicks):
    if n_clicks > 0:
        print(f"実行ボタンが押されました (クリック数: {n_clicks})")
    return {}

# If running as script
if __name__ == '__main__':
    app.run_server(debug=True, port=8055)
