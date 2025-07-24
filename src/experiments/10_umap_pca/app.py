import os
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import time
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate
from tqdm import tqdm
import json
from dash.dependencies import ClientsideFunction

# データロード・前処理のためのユーティリティ関数
def load_mnist_data(sample_size=5000):
    """MNISTデータをロード"""
    from sklearn.datasets import fetch_openml
    print("MNISTデータをロード中...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    if sample_size and sample_size < len(mnist.data):
        # サンプルサイズが指定されている場合はランダムサンプリング
        # indices = np.random.choice(len(mnist.data), sample_size, replace=False)
        # 先頭から
        indices = np.arange(sample_size)        
        data = mnist.data[indices]
        labels = mnist.target[indices]
    else:
        data = mnist.data
        labels = mnist.target
    
    print(f"データロード完了: {data.shape}")
    return data, labels

def compute_pca(data, n_components=2):
    """PCAを計算"""
    print("PCAを計算中...")
    start_time = time.time()
    
    # 標準化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # PCA
    pca = PCA(n_components=n_components)
    result = pca.fit_transform(scaled_data)
    
    print(f"PCA完了: {time.time() - start_time:.2f}秒")
    return result, pca, scaler

def compute_umap(data, n_neighbors=15, min_dist=0.1, n_components=2):
    """UMAPを計算"""
    print("UMAPを計算中...")
    start_time = time.time()
    
    # UMAP
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                n_components=n_components, random_state=42)
    result = umap.fit_transform(data)
    
    print(f"UMAP完了: {time.time() - start_time:.2f}秒")
    return result, umap

def cluster_data(embedding, eps=0.5, min_samples=10):
    """埋め込みデータをクラスタリング"""
    print("クラスタリング中...")
    start_time = time.time()
    
    # DBSCANクラスタリング
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = clustering.fit_predict(embedding)
    
    # クラスタリングの質を評価
    if len(np.unique(clusters)) > 1 and -1 not in np.unique(clusters):
        silhouette = silhouette_score(embedding, clusters)
    else:
        silhouette = 0
    
    print(f"クラスタリング完了: {time.time() - start_time:.2f}秒")
    print(f"クラスタ数: {len(np.unique(clusters))}, シルエットスコア: {silhouette:.4f}")
    
    return clusters, silhouette

def generate_weighted_pca(data, points_indices, scaler, weights_exp=4.0):
    """選択された点に基づいて重み付きPCAを生成（最適化版）"""
    start_time = time.time()
    
    # 標準化データを取得
    scaled_data = scaler.transform(data)
    scaling_time = time.time() - start_time
    
    # 重みベクトルを生成（選択された点は重み1.0、その他は小さい値）
    start_time = time.time()
    weights = np.ones(len(data)) * 1  # 基本重みは1.0
    weights[points_indices] = weights_exp
    
    # 重みを指数関数的に適用（コントラストを高める）
    # if weights_exp != 1.0:
    #     weights = weights ** weights_exp
    
    # 重みの正規化
    total_weight = weights.sum()
    weights_norm = time.time() - start_time
    
    # 中心化
    start_time = time.time()
    weighted_mean = np.average(scaled_data, axis=0, weights=weights)
    centered_data = scaled_data - weighted_mean
    centering_time = time.time() - start_time
    
    # 重み付き共分散行列を計算 (最適化版: 行列演算を使用)
    start_time = time.time()
    
    # 重みの平方根を使った効率的な行列計算
    sqrt_weights = np.sqrt(weights).reshape(-1, 1)
    weighted_data = centered_data * sqrt_weights
    weighted_cov = weighted_data.T @ weighted_data / total_weight
    
    cov_time = time.time() - start_time
    
    # 固有値分解
    start_time = time.time()
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    
    # 固有値を降順にソート
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 固有ベクトルを重み付きPCAの主成分とする
    weighted_pca_components = eigvecs
    eig_time = time.time() - start_time
    
    # 結果を投影
    start_time = time.time()
    projection = centered_data @ weighted_pca_components[:, :2]
    projection_time = time.time() - start_time
    
    # 全体の実行時間
    total_time = scaling_time + weights_norm + centering_time + cov_time + eig_time + projection_time
    
    if len(points_indices) > 0:  # 選択された点がある場合のみログを出力
        print(f"重み付きPCA計算時間内訳 (選択点数: {len(points_indices)}):")
        print(f"  データスケーリング: {scaling_time:.4f}秒")
        print(f"  重み計算: {weights_norm:.4f}秒")
        print(f"  データ中心化: {centering_time:.4f}秒")
        print(f"  共分散行列計算: {cov_time:.4f}秒")
        print(f"  固有値分解: {eig_time:.4f}秒")
        print(f"  データ投影: {projection_time:.4f}秒")
        print(f"  合計: {total_time:.4f}秒")
    
    return projection, weighted_pca_components, eigvals, weighted_mean

def create_weight_animation_frames(data, selected_indices, scaler, num_frames=30, max_weight=10.0):
    """重みのアニメーションフレームを生成（最適化版）"""
    # データサイズと選択点数に基づいてフレーム数を動的に調整
    if len(data) > 10000:
        num_frames = min(num_frames, 15)  # 大規模データではフレーム数を減らす
    elif len(data) > 5000 or len(selected_indices) > 300:
        num_frames = min(num_frames, 20)  # 中規模データや選択点が多い場合もフレーム数を減らす
    
    weights_selected = np.linspace(1.0, max_weight, num_frames)
    frames = []
    
    # キャッシュ用の変数を準備
    # マトリックス演算の中間結果をキャッシュすることでパフォーマンス向上
    scaled_data = scaler.transform(data)
    weights = np.ones(len(data)) * 1
    # base_weights[selected_indices] = 
    
    # キャッシュ用のメモリを確保（メモリ効率が重要）
    # 特に大きいデータセットでは重要
    n_samples, n_features = scaled_data.shape
    
    # 全体の時間計測開始
    total_start_time = time.time()
    print(f"アニメーションフレーム計算中... (フレーム数: {num_frames}, 選択点数: {len(selected_indices)})")
    
    # 並列処理によるパフォーマンス改善（可能な場合）
    # 現在はシリアル処理で最適化のみ実施
    
    frame_times = []
    for i, weight_selected in enumerate(tqdm(weights_selected, desc="フレーム生成")):
        frame_start = time.time()
        
        # 1. 重み計算
        weights[selected_indices] = weight_selected
        total_weight = weights.sum()
        
        # 2. 中心化（重み付き平均による）
        weighted_mean = np.average(scaled_data, axis=0, weights=weights)
        centered_data = scaled_data - weighted_mean
        
        # 3. 重み付き共分散行列（最も計算コストが高い部分なので最適化）
        # 方法1: 重みの平方根を使った最適化
        sqrt_weights = np.sqrt(weights).reshape(-1, 1)
        weighted_data = centered_data * sqrt_weights
        weighted_cov = weighted_data.T @ weighted_data / total_weight
        
        # 4. 固有値分解（サイズが大きくない場合はeighを使用）
        eigvals, eigvecs = np.linalg.eigh(weighted_cov)
        
        # 5. 固有値を降順にソート
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # 6. 次元削減のための投影（最初の2次元のみ計算）
        # 全次元を計算せずに2次元だけ計算することで高速化
        projection = centered_data @ eigvecs[:, :2]
        
        # 7. 結果を保存（最小限のデータのみ）
        frames.append({
            'projection': projection,  # 投影結果
            'components': eigvecs[:, :2].T,  # 主成分（可視化用）
            'eigenvalues': eigvals[:2].tolist()  # 固有値
        })
        
        frame_times.append(time.time() - frame_start)
        
        # 進捗情報（表示頻度を下げてオーバーヘッドを削減）
        if i == 0 or i == num_frames - 1 or (num_frames > 10 and i % (num_frames // 5) == 0):
            elapsed = time.time() - total_start_time
            remaining = elapsed / (i + 1) * (num_frames - i - 1) if i < num_frames - 1 else 0
            print(f"  フレーム {i+1}/{num_frames} 完了 ({elapsed:.2f}秒経過, 残り約{remaining:.2f}秒)")
    
    total_time = time.time() - total_start_time
    print(f"アニメーションフレーム計算完了！ 合計時間: {total_time:.2f}秒")
    print(f"フレームあたり平均時間: {np.mean(frame_times):.4f}秒 (最小: {np.min(frame_times):.4f}秒, 最大: {np.max(frame_times):.4f}秒)")
    
    # メモリを節約するため、投影結果のみをキャッシュしておき、主成分と固有値は必要なときに計算するという方法もある
    # しかし、ここではインタラクティブ性を優先して全データを保持
    
    return frames

def create_mnist_digit_figure(digit_vector, img_size=(28, 28)):
    """MNISTの数字ベクトルをPlotlyのfigureに変換"""
    digit_image = digit_vector.reshape(img_size)
    
    # Plotlyのfigureを作成
    fig = go.Figure(data=go.Heatmap(
        z=digit_image,
        colorscale='gray',
        showscale=False,
    ))
    
    fig.update_layout(
        height=200,
        width=200,
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed")  # 画像の上下を正しく表示するため
    )
    
    return fig

def create_pca_components_figure(pca_components, feature_size=(28, 28)):
    """PCA主成分（2つ）をPlotlyのサブプロットヒートマップとして可視化"""
    # サブプロットで2つの主成分を並べて表示
    fig = make_subplots(rows=1, cols=2, 
                         subplot_titles=(f'PCA Component 1', f'PCA Component 2'))
    
    # 第1主成分
    component1 = pca_components[0].reshape(feature_size)
    fig.add_trace(
        go.Heatmap(z=component1, colorscale='Viridis', showscale=False),
        row=1, col=1
    )
    
    # 第2主成分
    component2 = pca_components[1].reshape(feature_size)
    fig.add_trace(
        go.Heatmap(z=component2, colorscale='Viridis', showscale=True),
        row=1, col=2
    )
    
    fig.update_layout(
        height=300,
        width=600,
        title='PCA Components',
        margin=dict(l=5, r=5, b=5, t=40),
    )
    
    # 各サブプロットの軸を非表示に
    for i in range(1, 3):
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(visible=False, autorange="reversed", row=1, col=i)
    
    return fig

# グローバル変数（データとモデルを保持）
DATA = None
LABELS = None
PCA_RESULT = None
PCA_MODEL = None
SCALER = None
UMAP_RESULT = None
UMAP_MODEL = None
CLUSTERS = None
SILHOUETTE = None
ANIMATION_FRAMES = None
SELECTED_INDICES = []

# データの初期ロード
DATA, LABELS = load_mnist_data(sample_size=500)
PCA_RESULT, PCA_MODEL, SCALER = compute_pca(DATA)
UMAP_RESULT, UMAP_MODEL = compute_umap(DATA)
CLUSTERS, SILHOUETTE = cluster_data(UMAP_RESULT)

# Dashアプリケーション
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# レイアウト
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("UMAP + PCA ", className="text-center my-4"),
        ], width=12),
    ]),
    
    dbc.Row([
        # 設定パネル
        dbc.Col([
            html.H4("設定"),
            dbc.Card([
                dbc.CardBody([
                    html.H5("データ設定"),
                    dbc.Form([
                        dbc.Row([
                            dbc.Label("サンプルサイズ", width=6),
                            dbc.Col([
                                dbc.Input(id="input-sample-size", type="number", value=5000, min=100, max=70000),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Button("データ再ロード", id="btn-reload-data", color="primary", className="me-2"),
                    ]),
                    
                    html.Hr(),
                    
                    html.H5("UMAPパラメータ"),
                    dbc.Form([
                        dbc.Row([
                            dbc.Label("n_neighbors", width=6),
                            dbc.Col([
                                dbc.Input(id="input-umap-neighbors", type="number", value=15, min=2, max=100),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("min_dist", width=6),
                            dbc.Col([
                                dbc.Input(id="input-umap-min-dist", type="number", value=0.1, min=0.0, max=1.0, step=0.05),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Button("UMAP再計算", id="btn-recalculate-umap", color="primary", className="me-2"),
                    ]),
                    
                    html.Hr(),
                    
                    html.H5("クラスタリング"),
                    dbc.Form([
                        dbc.Row([
                            dbc.Label("eps (DBSCAN)", width=6),
                            dbc.Col([
                                dbc.Input(id="input-cluster-eps", type="number", value=0.5, min=0.1, max=10.0, step=0.1),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Label("min_samples", width=6),
                            dbc.Col([
                                dbc.Input(id="input-cluster-min-samples", type="number", value=10, min=2, max=100),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Button("クラスタリング再計算", id="btn-recalculate-clustering", color="primary", className="me-2"),
                    ]),
                    
                    html.Hr(),
                    
                    html.H5("表示設定"),
                    dbc.Form([
                        dbc.Row([
                            dbc.Label("色分け", width=6),
                            dbc.Col([
                                dbc.RadioItems(
                                    options=[
                                        {"label": "ラベル", "value": "labels"},
                                        {"label": "クラスタ", "value": "clusters"},
                                    ],
                                    value="labels",
                                    id="radio-color-mode",
                                ),
                            ], width=6),
                        ], className="mb-3"),
                    ]),
                ]),
            ]),
        ], width=3),
        
        # UMAPとPCA表示部分を並べて表示
        dbc.Col([
            # UMAPビュー (グローバルビュー)
            dbc.Card([
                dbc.CardHeader("UMAPビュー (グローバルビュー)"),
                dbc.CardBody([
                    dcc.Graph(id="umap-plot", style={"height": "40vh"}),
                    html.Div(id="umap-metrics", className="mt-2"),
                ])
            ], className="mb-3"),
            
            # PCAビュー
            dbc.Card([
                dbc.CardHeader("PCAビュー"),
                dbc.CardBody([
                    dcc.Graph(id="pca-plot", style={"height": "40vh"}),
                    
                ])
            ]),
        ], width=6),
        
        # 詳細表示部分
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("選択されたデータの詳細"),
                dbc.CardBody([
                    html.Div(id="selected-point-info"),
                    html.Div(id="selected-image-container"),
                ]),
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("PCA主成分の説明 (第1成分・第2成分)"),
                dbc.CardBody([
                    html.Div(id="pca-component-heatmap", className="text-center"),
                ]),
            ]),
            
            dbc.Card([
                dbc.CardHeader("メトリクス"),
                dbc.CardBody([
                    html.Div(id="metrics-container"),
                ]),
            ], className="mt-4"),
        ], width=3),
    ]),
    
    # アニメーション用のコンテナ
    dcc.Store(id='animation-frames-store'),
    dcc.Store(id='selected-indices-store'),
    dcc.Store(id="animation-state", data={"frame": 0}),
    
    # フッター
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("DRZoom - UMAP + PCA ダッシュボード", className="text-center text-muted"),
        ], width=12),
    ]),
], fluid=True)

# コールバック関数
@app.callback(
    [
        Output("umap-plot", "figure"),
        Output("umap-metrics", "children"),
        Output("metrics-container", "children"),
    ],
    [
        Input("radio-color-mode", "value"),
        Input("btn-reload-data", "n_clicks"),
        Input("btn-recalculate-umap", "n_clicks"),
        Input("btn-recalculate-clustering", "n_clicks"),
    ],
    [
        State("input-sample-size", "value"),
        State("input-umap-neighbors", "value"),
        State("input-umap-min-dist", "value"),
        State("input-cluster-eps", "value"),
        State("input-cluster-min-samples", "value"),
    ]
)
def update_umap_plot(color_mode, reload_clicks, recalc_umap_clicks, recalc_cluster_clicks,
                     sample_size, n_neighbors, min_dist, eps, min_samples):
    global DATA, LABELS, UMAP_RESULT, UMAP_MODEL, CLUSTERS, SILHOUETTE, PCA_RESULT, PCA_MODEL, SCALER
    
    # どのボタンが押されたか確認
    triggered_id = dash.callback_context.triggered_id
    
    if triggered_id == "btn-reload-data":
        # データ再ロード
        DATA, LABELS = load_mnist_data(sample_size=sample_size)
        PCA_RESULT, PCA_MODEL, SCALER = compute_pca(DATA)
        UMAP_RESULT, UMAP_MODEL = compute_umap(DATA, n_neighbors=n_neighbors, min_dist=min_dist)
        CLUSTERS, SILHOUETTE = cluster_data(UMAP_RESULT, eps=eps, min_samples=min_samples)
        
    elif triggered_id == "btn-recalculate-umap":
        # UMAP再計算
        UMAP_RESULT, UMAP_MODEL = compute_umap(DATA, n_neighbors=n_neighbors, min_dist=min_dist)
        CLUSTERS, SILHOUETTE = cluster_data(UMAP_RESULT, eps=eps, min_samples=min_samples)
        
    elif triggered_id == "btn-recalculate-clustering":
        # クラスタリング再計算
        CLUSTERS, SILHOUETTE = cluster_data(UMAP_RESULT, eps=eps, min_samples=min_samples)
    
    # 色分けの選択と色マップの作成
    if color_mode == "labels":
        color_values = LABELS.astype(int)  # 確実に整数に変換
        color_title = "ラベル"
        # カテゴリマッピングの作成
        categories = sorted(np.unique(color_values))
        color_map = {str(cat): px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                    for i, cat in enumerate(categories)}
    else:  # clusters
        # -1（ノイズ）も含まれている可能性があるためマップ作成
        categories = sorted(np.unique(CLUSTERS))
        color_map = {str(cat): px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                    for i, cat in enumerate(categories)}
        color_values = CLUSTERS
        color_title = "クラスタ"
    
    # UMAPプロット作成（正方形レイアウト）- Figureオブジェクト直接作成
    fig = go.Figure(
        data=go.Scatter(
            x=UMAP_RESULT[:, 0], 
            y=UMAP_RESULT[:, 1],
            mode='markers',
            marker=dict(
                color=[color_map[str(val)] for val in color_values],
                opacity=0.7,
                size=5
            ),
            hovertext=[f"点 {i}: ラベル {LABELS[i]}, クラスタ {CLUSTERS[i]}" for i in range(len(DATA))],
            hoverinfo="text",
        )
    )
    
    # レイアウト設定（すべての設定を一度に行う）
    fig.update_layout(
        title="UMAP投影",
        clickmode='event+select',
        dragmode='lasso',
        hoverdistance=5,
        # 正方形に設定
        height=500,
        width=500,
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )    # UMAPメトリクス
    unique_clusters = len(np.unique(CLUSTERS))
    noise_points = np.sum(CLUSTERS == -1)
    umap_metrics = html.Div([
        html.P([
            html.Strong("クラスタ数: "), f"{unique_clusters}",
            html.Br(),
            html.Strong("ノイズ点の数: "), f"{noise_points} ({noise_points/len(DATA)*100:.1f}%)",
            html.Br(),
            html.Strong("シルエットスコア: "), f"{SILHOUETTE:.4f}",
        ])
    ])
    
    # メトリクス表示
    metrics = html.Div([
        html.P([
            html.Strong("データサイズ: "), f"{len(DATA)}",
            html.Br(),
            html.Strong("特徴量次元: "), f"{DATA.shape[1]}",
            html.Br(),
            html.Strong("UMAPパラメータ: "), f"n_neighbors={n_neighbors}, min_dist={min_dist}",
            html.Br(),
            html.Strong("PCA説明分散比: "), f"{PCA_MODEL.explained_variance_ratio_[0]:.4f}, {PCA_MODEL.explained_variance_ratio_[1]:.4f}",
            html.Br(),
            html.Strong("合計説明分散比: "), f"{sum(PCA_MODEL.explained_variance_ratio_[:2]):.4f}",
        ])
    ])
    
    return fig, umap_metrics, metrics

@app.callback(
    [
        Output("animation-frames-store", "data"),
        Output("selected-indices-store", "data"),
        Output("animation-state", "data", allow_duplicate=True),
    ],
    [
        Input("umap-plot", "selectedData"),
    ],
    [
        State("animation-state", "data"),
    ],
    prevent_initial_call=True
)
def prepare_animation_frames(selected_data, animation_state):
    global DATA, SCALER, SELECTED_INDICES, ANIMATION_FRAMES
    
    if not selected_data or not selected_data.get('points'):
        # 選択されていない場合は更新しない
        return no_update, no_update, no_update, no_update
    
    # 選択された点のインデックスを取得
    selected_indices = [p['pointIndex'] for p in selected_data['points']]
    SELECTED_INDICES = selected_indices
    
    print(f"選択された点: {len(selected_indices)}")
    
    # アニメーション再生を停止
    animation_state = animation_state or {"playing": False, "frame": 0}
    animation_state["playing"] = False
    animation_state["frame"] = 0
    
    # アニメーションフレームを生成
    ANIMATION_FRAMES = create_weight_animation_frames(
        DATA, selected_indices, SCALER, num_frames=30)
    
    # フレームデータを構造化して保存
    # 最初と最後のフレームのみをJSONに保存し、中間フレームはグローバル変数に保存
    frame_data = {
        'first': {
            'projection': ANIMATION_FRAMES[0]['projection'].tolist(),
            'eigenvalues': ANIMATION_FRAMES[0]['eigenvalues']
        },
        'last': {
            'projection': ANIMATION_FRAMES[-1]['projection'].tolist(),
            'eigenvalues': ANIMATION_FRAMES[-1]['eigenvalues']
        },
    }
    
    return frame_data, selected_indices, animation_state

@app.callback(
    [
        Output("pca-plot", "figure"),
        Output("animation-state", "data", allow_duplicate=True),
    ],
    [
        Input("animation-frames-store", "data"),
        Input("selected-indices-store", "data"),
        Input("radio-color-mode", "value"),
        Input("umap-plot", "selectedData"),  # UMAPの選択を直接監視
    ],
    [
        State("animation-state", "data"),
    ],
    prevent_initial_call=True
)
def update_pca_plot(frame_data, selected_indices, color_mode, 
                   umap_selected_data, animation_state):
    global DATA, LABELS, PCA_RESULT, CLUSTERS, ANIMATION_FRAMES
    
    # アニメーション状態が初期化されていない場合に初期化
    animation_state = animation_state or {"frame": 0}
    
    # フレーム番号はanimation_stateから取得
    slider_value = animation_state.get("frame", 0)
    
    return create_pca_plot_with_animation(frame_data, selected_indices, slider_value, color_mode), animation_state

def create_pca_plot_with_animation(frame_data, selected_indices, slider_value, color_mode):
    """Plotlyのネイティブアニメーション機能を使ってPCAプロットを作成"""
    global DATA, LABELS, PCA_RESULT, CLUSTERS, ANIMATION_FRAMES
    
    # アニメーションフレームがない場合は標準PCAを表示
    if not frame_data or not selected_indices or ANIMATION_FRAMES is None:
        # 色分けの選択
        if color_mode == "labels":
            color_values = LABELS.astype(int)  # 確実に整数に変換
            color_title = "ラベル"
        else:  # clusters
            color_values = CLUSTERS
            # -1（ノイズ）も含まれている可能性があるためマップ作成
            color_map = {label: i for i, label in enumerate(np.unique(CLUSTERS))}
            color_values = np.array([color_map[c] for c in CLUSTERS])
            color_title = "クラスタ"
            
        # 標準PCAプロット（正方形レイアウト）
        fig = px.scatter(
            x=PCA_RESULT[:, 0], y=PCA_RESULT[:, 1],
            color=color_values,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            labels={"color": color_title},
            title="PCA投影",
        )
        
        fig.update_traces(
            marker=dict(size=5, opacity=0.7),
            hoverinfo="text",
            hovertext=[f"点 {i}: ラベル {LABELS[i]}, クラスタ {CLUSTERS[i]}" for i in range(len(DATA))]
        )
        
        # 正方形レイアウト
        fig.update_layout(
            height=500,
            width=500,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        
        return fig
    
    # アニメーションの準備（全フレームをPlotlyのアニメーションとして準備）
    frames = []
    num_frames = len(ANIMATION_FRAMES)
    
    # 色分けの選択と色マップの作成
    if color_mode == "labels":
        color_values = LABELS.astype(int)  # 確実に整数に変換
        color_title = "ラベル"
        # カテゴリマッピングの作成
        categories = sorted(np.unique(color_values))
        color_map = {str(cat): px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                    for i, cat in enumerate(categories)}
    else:  # clusters
        # -1（ノイズ）も含まれている可能性があるためマップ作成
        categories = sorted(np.unique(CLUSTERS))
        color_map = {str(cat): px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                    for i, cat in enumerate(categories)}
        color_values = CLUSTERS
        color_title = "クラスタ"
    
    # メインの図を作成（最初のフレーム）
    # スライダー値がない場合は0を使用
    if slider_value is None:
        slider_value = 0
    current_frame = slider_value if 0 <= slider_value < num_frames else 0
    projection = ANIMATION_FRAMES[current_frame]['projection']
    eigenvalues = ANIMATION_FRAMES[current_frame]['eigenvalues']
    weight_exp = 1.0 + (3.0 * current_frame / (num_frames - 1))  # 1.0 to 4.0
    
    # 選択された点と選択されていない点でOpacityを変える
    opacity = np.ones(len(DATA)) * 0.3  # 基本は薄く
    opacity[selected_indices] = 1.0  # 選択された点は濃く
    
    # DataFrameを作成
    df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'color_value': color_values,  # 名前をcolor_valueに変更
        'opacity': opacity,
        'index': list(range(len(DATA))),
        'label': [f"ラベル {LABELS[i]}" for i in range(len(DATA))],
        'cluster': [f"クラスタ {CLUSTERS[i]}" for i in range(len(DATA))]
    })
    
    # メイン図（初期表示用）- Scatterオブジェクトを直接作成
    fig = go.Figure(
        data=go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='markers',
            marker=dict(
                color=[color_map[str(val)] for val in color_values],
                opacity=df['opacity'],
                size=6,
            ),
            customdata=np.column_stack((df['index'], df['label'], df['cluster'])),
            hovertemplate="<b>点 %{customdata[0]}</b><br>" +
                        "%{customdata[1]}<br>" +
                        "%{customdata[2]}<br>" +
                        "X: %{x:.3f}<br>" +
                        "Y: %{y:.3f}"
        )
    )
    
    # タイトルとレイアウト設定
    fig.update_layout(
        title=f"重み付きPCA (重み: {weight_exp:.2f}, 選択: {len(selected_indices)}点)",
    )    # 全フレームを作成
    for i in range(num_frames):
        projection = ANIMATION_FRAMES[i]['projection']
        eigenvalues = ANIMATION_FRAMES[i]['eigenvalues']
        weight_exp = 1.0 + (3.0 * i / (num_frames - 1))  # 1.0 to 4.0
        
        # フレームごとのデータ
        frame_df = pd.DataFrame({
            'x': projection[:, 0],
            'y': projection[:, 1],
            'color_value': color_values,  # 名前をcolor_valueに変更
            'opacity': opacity,
            'index': list(range(len(DATA))),
            'label': [f"ラベル {LABELS[i]}" for i in range(len(DATA))],
            'cluster': [f"クラスタ {CLUSTERS[i]}" for i in range(len(DATA))]
        })
        
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=frame_df['x'],
                    y=frame_df['y'],
                    mode='markers',
                    marker=dict(
                        opacity=frame_df['opacity'],
                        size=6,
                        color=[color_map[str(val)] for val in color_values],
                    ),
                    customdata=np.column_stack((
                        frame_df['index'], 
                        frame_df['label'], 
                        frame_df['cluster']
                    )),
                    hovertemplate="<b>点 %{customdata[0]}</b><br>" +
                                 "%{customdata[1]}<br>" +
                                 "%{customdata[2]}<br>" +
                                 "X: %{x:.3f}<br>" +
                                 "Y: %{y:.3f}"
                )
            ],
            name=str(i),
            layout=go.Layout(
                title=f"重み付きPCA (重み: {weight_exp:.2f}, 選択: {len(selected_indices)}点)",
                annotations=[{
                    "text": f"寄与率: PC1={eigenvalues[0]/sum(eigenvalues):.4f}, PC2={eigenvalues[1]/sum(eigenvalues):.4f}",
                    "xref": "paper", "yref": "paper",
                    "x": 0.5, "y": 0.02,
                    "showarrow": False,
                    "font": dict(size=10),
                    "bgcolor": "rgba(255,255,255,0.8)",
                    "bordercolor": "gray",
                    "borderwidth": 1,
                }]
            )
        )
        frames.append(frame)
    
    # フレームを図に追加
    fig.frames = frames
    
    # 正方形レイアウトとアニメーションコントロールを設定
    fig.update_layout(
        height=500,
        width=500,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        margin=dict(l=10, r=10, t=50, b=10),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                    "label": "▶ 再生",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    "label": "⏸ 停止",
                    "method": "animate"
                }
            ],
            "type": "buttons",
            "direction": "left",
            "x": 0.05,
            "y": -0.15,
        }],
        sliders=[{
            "steps": [
                {
                    "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": f"{1.0 + 3.0 * i / (num_frames - 1):.1f}",
                    "method": "animate"
                } 
                for i in range(0, num_frames, max(1, num_frames // 10))  # 10分割程度のラベル
            ],
            "active": current_frame,
            "x": 0.05,
            "y": -0.05,
            "len": 0.9,
            "currentvalue": {
                "visible": True,
                "prefix": "重み: ",
                "suffix": "",
                "font": {"size": 12}
            }
        }]
    )
    
    # 固有値情報をアノテーションとして追加
    if eigenvalues:
        total_variance = sum(eigenvalues)
        var_ratio = [ev / total_variance for ev in eigenvalues]
        fig.add_annotation(
            text=f"寄与率: PC1={var_ratio[0]:.4f}, PC2={var_ratio[1]:.4f}",
            xref="paper", yref="paper",
            x=0.5, y=0.02,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        )
    
    return fig

def update_pca_figure(frame_data, selected_indices, slider_value, color_mode):
    """PCAプロットを更新する内部関数"""
    global DATA, LABELS, PCA_RESULT, CLUSTERS, ANIMATION_FRAMES
    
    # アニメーションフレームがない場合は標準PCAを表示
    if not frame_data or not selected_indices or ANIMATION_FRAMES is None:
        # 色分けの選択
        if color_mode == "labels":
            color_values = LABELS
            color_title = "ラベル"
        else:  # clusters
            color_values = CLUSTERS
            color_title = "クラスタ"
            
        # 標準PCAプロット（正方形レイアウト）
        fig = px.scatter(
            x=PCA_RESULT[:, 0], y=PCA_RESULT[:, 1],
            color=color_values,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            labels={"color": color_title},
            title="PCA投影",
        )
        
        fig.update_traces(
            marker=dict(size=5, opacity=0.7),
            hoverinfo="text",
            hovertext=[f"点 {i}: ラベル {LABELS[i]}, クラスタ {CLUSTERS[i]}" for i in range(len(DATA))]
        )
        
        # 正方形レイアウト
        fig.update_layout(
            height=500,
            width=500,
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        
        return fig
    
    # スライダー値がない場合は0を使用
    if slider_value is None:
        slider_value = 0

    # スライダー値に応じたフレームを選択
    if slider_value == 0:
        # 最初のフレーム（標準PCA）
        if 'first' in frame_data and 'projection' in frame_data['first']:
            current_frame = np.array(frame_data['first']['projection'])
            eigenvalues = frame_data['first']['eigenvalues']
        else:
            current_frame = ANIMATION_FRAMES[0]['projection']
            eigenvalues = ANIMATION_FRAMES[0]['eigenvalues']
    elif slider_value == 29:
        # 最後のフレーム（完全な重み付きPCA）
        if 'last' in frame_data and 'projection' in frame_data['last']:
            current_frame = np.array(frame_data['last']['projection'])
            eigenvalues = frame_data['last']['eigenvalues']
        else:
            current_frame = ANIMATION_FRAMES[-1]['projection']
            eigenvalues = ANIMATION_FRAMES[-1]['eigenvalues']
    else:
        # 中間フレーム（アニメーション内で補間）
        # 有効な範囲内にあるか確認
        if 0 <= slider_value < len(ANIMATION_FRAMES):
            current_frame = ANIMATION_FRAMES[slider_value]['projection']
            eigenvalues = ANIMATION_FRAMES[slider_value]['eigenvalues']
        else:
            # 範囲外の場合は最初のフレームを使用
            current_frame = ANIMATION_FRAMES[0]['projection']
            eigenvalues = ANIMATION_FRAMES[0]['eigenvalues']
    
    # 色分けの選択
    if color_mode == "labels":
        color_values = LABELS
        color_title = "ラベル"
    else:  # clusters
        color_values = CLUSTERS
        color_title = "クラスタ"
    
    # 選択された点と選択されていない点でOpacityを変える
    opacity = np.ones(len(DATA)) * 0.3  # 基本は薄く
    opacity[selected_indices] = 1.0  # 選択された点は濃く
    
    # プロットデータの作成
    plot_df = pd.DataFrame({
        'x': current_frame[:, 0],
        'y': current_frame[:, 1],
        'color': color_values,
        'opacity': opacity,
    })
    
    # プロット作成（正方形レイアウト）
    fig = px.scatter(
        plot_df, x='x', y='y',
        color='color',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        labels={"color": color_title},
        title=f"重み付きPCA (選択: {len(selected_indices)}点)",
        opacity=plot_df['opacity'],
    )
    
    fig.update_traces(
        marker=dict(size=5),
        hoverinfo="text",
        hovertext=[f"点 {i}: ラベル {LABELS[i]}, クラスタ {CLUSTERS[i]}" for i in range(len(DATA))]
    )
    
    # スライダー位置に応じてタイトルと情報を更新
    weight_exp = 1.0 + (3.0 * slider_value / 29)  # 1.0 to 4.0
    fig.update_layout(
        title=f"重み付きPCA (重み: {weight_exp:.2f}, 選択: {len(selected_indices)}点)",
        # 正方形レイアウト
        height=500,
        width=500,
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    # 固有値情報をアノテーションとして追加
    if eigenvalues:
        total_variance = sum(eigenvalues)
        var_ratio = [ev / total_variance for ev in eigenvalues]
        fig.add_annotation(
            text=f"寄与率: PC1={var_ratio[0]:.4f}, PC2={var_ratio[1]:.4f}",
            xref="paper", yref="paper",
            x=0.5, y=0.02,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        )
    
    return fig

@app.callback(
    [
        Output("selected-point-info", "children"),
        Output("selected-image-container", "children"),
    ],
    [
        Input("umap-plot", "clickData"),
        Input("pca-plot", "clickData"),
    ]
)
def update_selected_point_info(umap_click_data, pca_click_data):
    global DATA, LABELS, CLUSTERS
    
    # 最後にクリックされた点の情報を使用
    ctx_triggered = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    
    if ctx_triggered == "umap-plot" and umap_click_data:
        clicked_data = umap_click_data
        plot_type = "UMAP"
    elif ctx_triggered == "pca-plot" and pca_click_data:
        clicked_data = pca_click_data
        plot_type = "PCA"
    else:
        # クリックされていない場合
        return html.P("点をクリックして詳細を表示"), html.Div()
    
    # クリックされた点のインデックスを取得
    point_index = clicked_data['points'][0]['pointIndex']
    
    # 情報表示
    point_info = html.Div([
        html.P([
            html.Strong("選択された点: "), f"{point_index}",
            html.Br(),
            html.Strong("プロット: "), plot_type,
            html.Br(),
            html.Strong("ラベル: "), f"{LABELS[point_index]}",
            html.Br(),
            html.Strong("クラスタ: "), f"{CLUSTERS[point_index]}",
        ])
    ])
    
    # MNIST画像をPlotlyで表示
    fig = create_mnist_digit_figure(DATA[point_index])
    image_container = html.Div([
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        html.P(f"ラベル: {LABELS[point_index]}", className="text-center"),
    ])
    
    return point_info, image_container



@app.callback(
    Output("pca-component-heatmap", "children"),
    [
        Input("btn-reload-data", "n_clicks"),
    ]
)
def update_pca_component_heatmap(n_clicks):
    global PCA_MODEL
    
    if PCA_MODEL is None:
        return html.Div("PCAがロードされていません")
    
    # 両方の主成分を表示するPlotlyヒートマップを生成
    fig = create_pca_components_figure(
        PCA_MODEL.components_, feature_size=(28, 28)
    )
    
    # Plotly図表示
    heatmap_container = html.Div([
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        html.Div([
            html.Span(f"第1成分寄与率: {PCA_MODEL.explained_variance_ratio_[0]:.4f}", className="me-3"),
            html.Span(f"第2成分寄与率: {PCA_MODEL.explained_variance_ratio_[1]:.4f}"),
            html.Br(),
            html.Span(f"合計寄与率: {PCA_MODEL.explained_variance_ratio_[0] + PCA_MODEL.explained_variance_ratio_[1]:.4f}")
        ], className="text-center mt-2"),
    ])
    
    return heatmap_container

if __name__ == '__main__':
    app.run_server(debug=True)
