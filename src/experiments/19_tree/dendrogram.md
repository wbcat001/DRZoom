# Dendrogram Analysis and Dash Application

## 概要
`dendrogram.ipynb`は、HDBSCANクラスタリング結果の階層構造分析と対話的可視化を行う包括的なノートブックです。HDBSCANから得られた凝縮木（condensed tree）をSciPyのLinkage Matrix形式に変換し、Strahlter数やStability値によるフィルタリングを適用した動的デンドログラム可視化システムを実装しています。

## アプリケーション構成

### レイアウト構造
```
+---------------------------+---------------------------+
|         散布図             |       デンドログラム        |
|    (Scatter Plot)         |      (Dendrogram)        |
|                           |                          |
|                           | スライダーコントロール：        |
|                           | - Strahlter値            |
|                           | - Stability閾値          |
+---------------------------+---------------------------+
|              出力表示エリア                           |
+-------------------------------------------------------+
```

### 主要コンポーネント

#### 1. 散布図 (Left Panel)
- **目的**: 2D埋め込み空間でのデータポイント表示
- **機能**: 
  - ブラシ/ラッソ選択によるポイントの複数選択
  - 選択されたポイントがデンドログラムでハイライト

#### 2. デンドログラム (Right Panel) 
- **目的**: 階層クラスタリング構造の可視化
- **機能**:
  - Strahlter値によるノードフィルタリング
  - Stability閾値による品質フィルタリング  
  - 散布図の選択と連動したクラスター強調表示

#### 3. コントロールパネル
- **Strahlter値スライダー**: データポイント数による階層レベルのフィルタリング
- **Stability閾値スライダー**: クラスターの安定性による品質フィルタリング

## 主要な処理関数

### データ準備
```python
# 埋め込みデータの読み込み
data = np.load("../18_rapids/result/20251112_044404/embedding.npz")
embedding = data["embedding"]
labels = np.load("../18_rapids/result/20251112_044404/data.npz")["labels"]

# DataFrameの作成
df = pd.DataFrame({
    'x': embedding[:, 0],
    'y': embedding[:, 1], 
    'label': labels
})
```

### コールバック関数

#### 1. デンドログラム更新コールバック
```python
@app.callback(
    Output('dendrogram-output', 'figure'),
    Output('output', 'children'),
    [
        Input('strahler-slider', 'value'),
        Input('stability-slider', 'value'),
        Input("scatter-output", 'selectedData')
    ]
)
def update_dendrogram(strahler_threshold, stability_threshold, selectedData):
```
**機能**:
- Strahlter値によるリンケージ行列のフィルタリング
- 選択されたポイントのクラスターID特定
- デンドログラムでの該当クラスター強調表示

#### 2. 散布図更新コールバック
```python
@app.callback(
    Output('scatter-output', 'figure'),
    [Input('strahler-slider', 'value'), Input('stability-slider', 'value')]
)
def update_scatter_chart(strahler_threshold, stability_threshold):
```
**機能**:
- 閾値に応じた散布図の表示更新
- ブラシ選択モードの設定

## 核心的なアルゴリズム

### 1. Strahlter値による階層フィルタリング
```python
def filter_linkage_matrix_by_strahler(Z, S_min=0, N_leaves=None):
    """
    Strahlter値に基づいてリンケージ行列をフィルタリング
    - Z: リンケージ行列
    - S_min: 最小Strahlter値閾値
    - N_leaves: 葉ノード数
    """
```

### 2. ポイントからクラスターIDへの変換
```python
def get_clusters_from_points(point_ids: List[int], point_id_map: dict) -> List[int]:
    """
    選択されたポイントIDリストから対応するクラスターIDを取得
    """
```

### 3. デンドログラムセグメント生成
```python
def get_dendrogram_segments2(Z_filtered):
    """
    フィルタリング済みリンケージ行列からプロット用セグメントを生成
    """
```

### 4. Plotlyデンドログラム描画
```python
def plot_dendrogram_plotly(segments, colors=None, scores=None, is_selecteds=None):
    """
    Plotlyを使用したデンドログラムの描画
    - segments: デンドログラムセグメント
    - is_selecteds: 選択状態の配列（強調表示用）
    """
```

## インタラクション機能

### 散布図 → デンドログラム
1. 散布図でデータポイントをブラシ/ラッソ選択
2. `selectedData`から選択ポイントのインデックスを取得  
3. `get_clusters_from_points()`でクラスターIDに変換
4. デンドログラムで該当クラスターを強調表示

### パラメータ調整
- **Strahlter値**: より大きな値で上位階層のみ表示
- **Stability値**: より高い値で安定したクラスターのみ表示

## データフロー

```
Raw Data → HDBSCAN → Linkage Matrix (Z)
                  ↓
         Filter by Strahlter/Stability
                  ↓  
         Generate Dendrogram Segments
                  ↓
         Plotly Visualization
                  ↓
    User Selection → Cluster Highlighting
```

## 詳細な実装コード

### 1. データ読み込みとHDBSCAN結果の変換

#### HDBSCANオブジェクトの読み込み
```python
import pickle

file_path = "../18_rapids/result/20251112_044404/condensed_tree_object.pkl"
with open(file_path, 'rb') as f:
    clusterer = pickle.load(f)
```

#### HDBSCAN凝縮木からLinkage Matrixへの変換
```python
def get_linkage_matrix_from_hdbscan(condensed_tree):
    """
    HDBSCANの凝縮木からSciPy互換のLinkage Matrixを生成
    """
    # condensed_treeから親-子関係の抽出
    # クラスターのマージ情報をZ行列形式に変換
    # 戻り値: Z (linkage matrix), node_id_map (ID変換辞書)
```

### 2. 階層構造の分析アルゴリズム

#### Strahlter数の計算
```python
def calculate_strahler(Z_matrix: np.ndarray, n_leaves: int) -> np.ndarray:
    """
    Linkage Matrix (Z) に基づいて、各結合ノードのストラー数（Strahler Number）を計算する。
    
    Args:
        Z_matrix: Linkage Matrix (N-1 x 4のNumPy配列)。
        n_leaves: 元の観測値/葉ノードの数。
        
    Returns:
        np.ndarray: 各結合ノード（Zの各行）に対応するストラー数の配列。
    """
    n_merges = Z_matrix.shape[0]
    
    # 葉ノードのストラー数を初期化 (すべての葉ノードは S=1)
    strahler_map = {i: 1 for i in range(n_leaves)}
    
    # Zの各行に対応するストラー数を格納するリスト
    merge_strahler_numbers = np.zeros(n_merges, dtype=int)
    
    # Z行列をボトムアップ（行 0 から N-2）で処理
    for i in range(n_merges):
        u_idx = int(Z_matrix[i, 0])  # 結合されるノード u
        v_idx = int(Z_matrix[i, 1])  # 結合されるノード v
        new_idx = n_leaves + i       # 新しく生成されるノード
        
        # 子ノードのストラー数を取得
        s_u = strahler_map.get(u_idx, 1)
        s_v = strahler_map.get(v_idx, 1)
        
        # ストラー数計算ロジック（二分木）
        if s_u == s_v:
            # S_u = S_v の場合、新しいノードのストラー数は S_u + 1
            s_new = s_u + 1
        else:
            # S_u != S_v の場合、新しいノードのストラー数は Max(S_u, S_v)
            s_new = max(s_u, s_v)
        
        # 結果を記録し、マップを更新
        merge_strahler_numbers[i] = s_new
        strahler_map[new_idx] = s_new

    return merge_strahler_numbers
```

#### Strahlter値によるフィルタリング
```python
def filter_linkage_matrix_by_strahler(Z_matrix: np.ndarray, S_min: int, N_leaves: int) -> tuple:
    """
    Linkage Matrix (Z) にストラー数を計算し、指定された最小ストラー数以上の結合のみを保持する。
    
    Args:
        Z_matrix (np.ndarray): Linkage Matrix (N-1 x 4)。
        S_min (int): フィルタリングのための最小ストラー数。
        N_leaves (int): 葉ノード数

    Returns:
        tuple: (フィルタリングされたZ行列, ノードIDマップ)
    """
    # 1. ストラー数 (Strahler Number) の計算
    strahler_numbers = calculate_strahler(Z_matrix, N_leaves)

    # 2. Z_matrix の拡張 (ストラー数を5列目に追加)
    Z_with_strahler = np.hstack((Z_matrix, strahler_numbers[:, np.newaxis]))

    # 3. フィルタリングの実行
    filtered_Z_by_strahler = Z_with_strahler[Z_with_strahler[:, 5] >= S_min]

    # 4. ノードIDの再マッピング
    node_id_map = {}
    current_id = 0
    leaves = get_leaves(filtered_Z_by_strahler)
    
    for leaf in leaves:
        node_id_map[int(leaf)] = current_id
        current_id += 1
    
    # 内部ノードのIDマッピング
    for row in filtered_Z_by_strahler:
        parent_id = row[2]
        if parent_id not in node_id_map:
            node_id_map[int(parent_id)] = current_id
            current_id += 1

    return filtered_Z_by_strahler, node_id_map
```

### 3. Stability計算

#### HDBSCAN Stabilityのピュア実装
```python
def compute_stability_python(condensed_tree):
    """
    HDBSCAN condensed treeからStabilityスコアを計算（Cython実装の代替）
    
    Args:
        condensed_tree: HDBSCAN condensed tree構造体
    
    Returns:
        dict: {cluster_id: stability_score} の辞書
    """
    # 1. 最小クラスターIDと範囲を特定
    smallest_cluster = condensed_tree['parent'].min()
    largest_cluster = condensed_tree['parent'].max()
    num_clusters = largest_cluster - smallest_cluster + 1
    
    # 2. 各クラスターのlambda_birth（誕生時刻）を計算
    births_arr = np.full(largest_cluster + 1, np.inf, dtype=np.double)
    
    # condensed_treeを子ノードでソート
    sorted_indices = np.argsort(condensed_tree['child'])
    sorted_child_data = condensed_tree[sorted_indices]
    
    current_child = -1
    min_lambda = np.inf
    
    for row in sorted_child_data:
        child = row['child']
        lambda_ = row['lambda_val']

        if child == current_child:
            min_lambda = min(min_lambda, lambda_)
        elif current_child != -1:
            births_arr[current_child] = min_lambda
            current_child = child
            min_lambda = lambda_
        else:
            current_child = child
            min_lambda = lambda_

    if current_child != -1:
        births_arr[current_child] = min_lambda
        
    births_arr[smallest_cluster] = 0.0  # ルートクラスタの lambda_birth は 0
    
    # 3. Stabilityスコアの計算
    result_arr = np.zeros(num_clusters, dtype=np.double)
    
    parents = condensed_tree['parent']
    sizes = condensed_tree['child_size']
    lambdas = condensed_tree['lambda_val']

    for i in range(condensed_tree.shape[0]):
        parent = parents[i]
        lambda_ = lambdas[i]
        child_size = sizes[i]
        result_index = parent - smallest_cluster
        
        lambda_birth = births_arr[parent]
        
        # Stability(C) = Σ (lambda_death - lambda_birth) * size
        result_arr[result_index] += (lambda_ - lambda_birth) * child_size
        
    # 4. ID とスコアを辞書に変換
    node_ids = np.arange(smallest_cluster, condensed_tree['parent'].max() + 1)
    result_pre_dict = np.vstack((node_ids, result_arr)).T

    return dict(zip(result_pre_dict[:, 0].astype(int), result_pre_dict[:, 1]))
```

### 4. デンドログラム座標計算

#### 座標計算アルゴリズム
```python
def compute_dendrogram_coords(Z, n_points):
    """
    Linkage Matrixからデンドログラム描画用の座標を計算
    
    Args:
        Z: Linkage Matrix (N-1 x 4)
        n_points: データポイント数
    
    Returns:
        tuple: (icoord, dcoord, leaf_order)
            - icoord: X座標のリスト
            - dcoord: Y座標のリスト  
            - leaf_order: 葉ノードの順序
    """
    # 1. ノード情報の初期化
    nodes = []
    
    # 葉ノード情報 (X座標は後で計算、Y座標は0)
    for i in range(n_points):
        nodes.append({
            'x': None,
            'y': 0,
            'size': 1
        })
    
    # 内部ノード情報をZから設定
    for i in range(n_points - 1):
        c1, c2, dist, count = Z[i]
        nodes.append({
            'x': None,
            'y': dist,
            'size': int(count),
            'left': int(c1),
            'right': int(c2)
        })

    # 2. 葉の順序決定（サイズベースソート）
    def get_leaf_order_sorted(node_idx):
        node = nodes[node_idx]
        
        if node_idx < n_points:
            return [node_idx]
        
        # 左右の子ノードのサイズを取得
        C1_idx, C2_idx = node['left'], node['right']
        size_C1, size_C2 = nodes[C1_idx]['size'], nodes[C2_idx]['size']

        # サイズの大きい方を左に配置
        if size_C1 < size_C2:
            C1_idx, C2_idx = C2_idx, C1_idx
        
        order_left = get_leaf_order_sorted(C1_idx)
        order_right = get_leaf_order_sorted(C2_idx)
        
        return order_left + order_right

    # 3. X座標の計算
    def calculate_x_coord(node_idx, leaf_to_x):
        node = nodes[node_idx]
        
        # 葉ノードの場合
        if node_idx < n_points:
            x_coord = leaf_to_x[node_idx]
            node['x'] = x_coord
            return x_coord
        
        # 内部ノードの場合：子ノードX座標の平均
        x_left = calculate_x_coord(node['left'], leaf_to_x)
        x_right = calculate_x_coord(node['right'], leaf_to_x)
        
        x_coord = (x_left + x_right) / 2.0
        node['x'] = x_coord
        return x_coord

    # ルートノードから葉の順序を決定
    root_node_idx = n_points - 1 + (n_points - 1)
    leaf_order = get_leaf_order_sorted(root_node_idx)

    # 葉のX座標の割り当て (1, 3, 5, ...)
    leaf_to_x = {leaf_idx: 2 * i + 1 for i, leaf_idx in enumerate(leaf_order)}
    
    # X座標の計算実行
    calculate_x_coord(root_node_idx, leaf_to_x)

    # 4. 描画用線分座標の生成
    icoord = []  # X座標リスト [x1, x2, x3, x4]
    dcoord = []  # Y座標リスト [y1, y2, y3, y4]

    for i in range(n_points - 1):
        P = n_points + i  # 親ノード P のインデックス
        C1 = nodes[P]['left']   # 左の子ノード
        C2 = nodes[P]['right']  # 右の子ノード

        # 座標取得
        y_P = nodes[P]['y']
        y_C1 = nodes[C1]['y']
        y_C2 = nodes[C2]['y']
        x_P = nodes[P]['x']
        x_C1 = nodes[C1]['x']
        x_C2 = nodes[C2]['x']

        # U字型の線分座標
        icoord.append([x_C1, x_C1, x_C2, x_C2])
        dcoord.append([y_C1, y_P, y_P, y_C2])

    return icoord, dcoord, leaf_order
```

#### セグメント生成
```python
def get_dendrogram_segments2(Z: np.ndarray):
    """
    Linkage Matrixからデンドログラム描画に必要な座標データを取得
    
    Args:
        Z: Linkage Matrix
    
    Returns:
        list: 描画用セグメントのリスト
    """
    n_points = Z.shape[0] + 1
    icoord, dcoord, leaf_order = compute_dendrogram_coords(Z, n_points)
    
    segments = []
    
    for icoords, dcoords in zip(icoord, dcoord):
        x1, x2, x3, x4 = icoords
        y1, y2, y3, y4 = dcoords

        # U字型の3つの線分を生成
        # 1. 垂直線 (左の子ノードから結合点まで)
        segments.append([(x1, y1), (x2, y2)]) 
        # 2. 水平線 (結合したノード間)
        segments.append([(x2, y2), (x3, y3)]) 
        # 3. 垂直線 (右の子ノードから結合点まで)
        segments.append([(x4, y4), (x3, y3)]) 

    return segments
```

### 5. Plotly可視化

#### デンドログラム描画関数
```python
def plot_dendrogram_plotly(segments, colors=None, scores=None, is_selecteds=None):
    """
    Plotlyを使用したデンドログラムの描画
    
    Args:
        segments: デンドログラムセグメント
        colors: セグメント色のリスト
        scores: スコア値のリスト（ホバー情報用）
        is_selecteds: 選択状態の配列（強調表示用）
    
    Returns:
        plotly.graph_objects.Figure: デンドログラムのプロット
    """
    fig = go.Figure()
    
    for i, seg in enumerate(segments):
        x_coords = [seg[0][0], seg[1][0]]
        y_coords = [seg[0][1], seg[1][1]]
        
        color = 'blue' if colors is None else colors[i]
        info = "no" if scores is None else f"{scores[i]:.2f}"
        opacity = 0.2 if is_selecteds is not None and not is_selecteds[i] else 1.0
        
        fig.add_trace(go.Scatter(
            x=x_coords, 
            y=y_coords, 
            mode='lines',
            line=dict(color=color, width=1),
            hoverinfo='text',
            text=f'Segment {i}: ({x_coords[0]:.2f}, {y_coords[0]:.2f}) to ({x_coords[1]:.2f}, {y_coords[1]:.2f}, score={info})',
            opacity=opacity
        ))
    
    fig.update_layout(
        title='Simple Dendrogram Visualization',
        xaxis_title='Observation Index',
        yaxis_title='Distance / Height',
        hovermode='closest',
        height=800, 
        width=1000
    )
    
    return fig
```

### 6. ポイント-クラスター変換

#### 選択ポイントからクラスターID取得
```python
def get_clusters_from_points(point_ids: List[int], point_id_map: dict) -> List[int]:
    """
    選択されたポイントIDリストから対応するクラスターIDを取得
    
    Args:
        point_ids: 選択されたデータポイントのIDリスト
        point_id_map: ポイントIDからクラスターIDへのマッピング辞書
    
    Returns:
        List[int]: 対応するクラスターIDのリスト
    """
    cluster_ids = []
    for point_id in point_ids:
        if point_id in point_id_map:
            cluster_ids.append(point_id_map[point_id])
    
    # 重複除去
    return list(set(cluster_ids))
```

## Dashアプリケーション実装

### アプリケーション構成
```
+---------------------------+---------------------------+
|         散布図             |       デンドログラム        |
|    (Scatter Plot)         |      (Dendrogram)        |
|                           |                          |
|                           | スライダーコントロール：        |
|                           | - Strahlter値            |
|                           | - Stability閾値          |
+---------------------------+---------------------------+
|              出力表示エリア                           |
+-------------------------------------------------------+
```

### メインアプリケーションコード
```python
from jupyter_dash import JupyterDash 
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np
import pandas as pd

# データの準備
max_strahler = 4
max_stability = 100

# 埋め込みデータの読み込み
data = np.load("../18_rapids/result/20251112_044404/embedding.npz")
embedding = data["embedding"]
labels = np.load("../18_rapids/result/20251112_044404/data.npz")["labels"]

df = pd.DataFrame({
    'x': embedding[:, 0],
    'y': embedding[:, 1],
    'label': labels
})

# Dashアプリケーションの定義
app = JupyterDash(__name__) 

app_layout = html.Div([
    html.H2("動的散布図とデンドログラムダッシュボード", 
            style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # 左右にプロットを配置するコンテナ
    html.Div([
        # 左側: 散布図
        html.Div([
            html.H3("散布図 (Scatter Plot)", style={'textAlign': 'center'}),
            dcc.Graph(
                id='scatter-output',
                style={'height': '70vh'}
            ),
        ], style={'flex': 1, 'minWidth': '48%', 'padding': '10px'}),
        
        # 右側: デンドログラム + スライダー
        html.Div([
            # スライダーコンテナ
            html.Div([
                # Strahlter スライダー
                html.Div([
                    html.Label("Strahlter値 (データポイント数):"),
                    dcc.Slider(
                        id='strahler-slider',
                        min=0,
                        max=max_strahler,
                        step=1,
                        value=0,
                        marks={i: str(i) for i in range(0, max_strahler + 1, 5) if i != 0}, 
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'flex': 1, 'marginRight': '20px'}),

                # Stability スライダー
                html.Div([
                    html.Label("Stability (%) 閾値:"),
                    dcc.Slider(
                        id='stability-slider',
                        min=0,
                        max=max_stability,
                        step=5,
                        value=20,
                        marks={i: str(i) for i in range(0, 101, 20)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'flex': 1}),

            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'padding': '10px 20px 20px',
                'borderBottom': '1px solid #ddd'
            }),

            html.H3("デンドログラム (Dendrogram)", style={'textAlign': 'center'}),
            dcc.Graph(
                id='dendrogram-output',
                style={'height': '70vh'}
            ),
        ], style={'flex': 1, 'minWidth': '48%', 'padding': '10px'}),

    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'gap': '1%',
        'alignItems': 'flex-start'
    }),
    
    html.Div(id="output", style={"marginTop": 20, "fontWeight": "bold", "color": "blue"}),
])

app.layout = app_layout
```

### 主要コールバック関数

#### デンドログラム更新コールバック
```python
@app.callback(
    Output('dendrogram-output', 'figure'),
    Output('output', 'children'),
    [
        Input('strahler-slider', 'value'),
        Input('stability-slider', 'value'),
        Input("scatter-output", 'selectedData')
    ]
)
def update_dendrogram(strahler_threshold, stability_threshold, selectedData):
    """デンドログラムを動的に更新し、選択されたポイントを強調表示"""
    
    # Strahlter値によるフィルタリング
    _Z, _node_id_map = filter_linkage_matrix_by_strahler(
        Z, S_min=strahler_threshold, N_leaves=len(_get_leaves(clusterer._raw_tree))
    )
    
    # デンドログラムセグメントの生成
    segments = get_dendrogram_segments2(_Z[:, [0, 1, 3, 4]])
    selected_new_cluster_ids = []
    is_selected = []

    # 散布図で選択されたデータがある場合
    if selectedData is not None:
        # 選択ポイントからクラスターIDを取得
        selected_cluster_ids = get_clusters_from_points(
            [int(point['pointIndex']) for index, point in enumerate(selectedData['points']) 
             if point["customdata"][0] is not -1],
            point_id_map
        )
        
        # フィルタ後のノードIDに変換
        selected_new_cluster_ids = [_node_id_map[k] for k in selected_cluster_ids 
                                   if k in _node_id_map]
        
        # デンドログラムでの強調表示用配列生成
        selected_new_cluster_ids = sum([[i]*3 for i in selected_new_cluster_ids], [])
        is_selected = [True if row[2] in selected_new_cluster_ids else False for row in Z]
        
        text = f"選択されたデータポイントのクラスターid: {', '.join(map(str, selected_new_cluster_ids))}"
    else:
        text = "選択されたデータポイントはありません。"
    
    # 強調表示配列の調整
    is_selected = sum([[i]*3 for i in is_selected], [])
    
    # デンドログラムの描画
    fig = plot_dendrogram_plotly(segments, is_selecteds=is_selected)
    
    return fig, text
```

#### 散布図更新コールバック
```python
@app.callback(
    Output('scatter-output', 'figure'),
    [Input('strahler-slider', 'value'), Input('stability-slider', 'value')]
)
def update_scatter_chart(strahler_threshold, stability_threshold):
    """パラメータに応じて散布図を更新"""
    
    fig = px.scatter(df,
                    x='x',
                    y='y',
                    custom_data=['label'])
    
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(dragmode="lasso")  # ラッソ選択モード

    return fig
```

## 技術スタック

- **JupyterDash**: Jupyter環境でのDashアプリ実行
- **Plotly**: インタラクティブグラフ描画
- **NumPy/Pandas**: 数値計算とデータ処理
- **HDBSCAN**: 階層クラスタリング
- **SciPy**: 階層クラスタリング構造体操作

## データフロー

```
Raw Data → HDBSCAN Clustering → Condensed Tree
                                      ↓
                            Linkage Matrix Conversion
                                      ↓
                    Strahlter Number Calculation
                                      ↓
                        Stability Score Computation
                                      ↓
                      Filtering (Strahlter + Stability)
                                      ↓
                     Dendrogram Coordinate Calculation
                                      ↓
                        Plotly Visualization
                                      ↓
            User Selection → Cluster Highlighting → Interactive Feedback
```

## 主要な特徴

1. **階層構造の動的フィルタリング**: Strahlter値による階層レベル調整
2. **品質ベースフィルタリング**: Stability値による低品質クラスター除外
3. **リアルタイム連動**: 散布図選択とデンドログラム強調の即座反映
4. **大規模データ対応**: 効率的な座標計算とフィルタリング
5. **視覚的フィードバック**: 選択クラスターの色分け表示

このシステムにより、数万〜数十万点規模のクラスタリング結果でも効率的な探索的データ分析が可能になります。
