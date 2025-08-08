# 重み付きPCAとアニメーションのためのDashアプリ
import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

# データを生成または読み込み
def generate_data(data_type='iris', n_samples=500):
    """データを生成または読み込むための関数"""
    if data_type == 'iris':
        # Irisデータセットを使用
        iris = px.data.iris()
        features = iris.drop('species', axis=1)
        labels = iris['species']
        return features, labels
    elif data_type == 'gaussian_clusters':
        # 3つのガウシアンクラスタを生成
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.7, random_state=42)
        df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        return df, y
    elif data_type == 'moons':
        # 半月形のデータセットを生成
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        return df, y
    elif data_type == 'circles':
        # 同心円状のデータセットを生成
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=42)
        df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        return df, y
    else:
        # デフォルトでIrisデータセットを返す
        iris = px.data.iris()
        features = iris.drop('species', axis=1)
        labels = iris['species']
        return features, labels

# 重み付きPCA関数
def weighted_pca(X, weights=None, n_components=2):
    """重み付きPCAを実行する関数"""
    if weights is None:
        weights = np.ones(X.shape[0])
    
    # データのコピーを作成
    X_weighted = X.copy()
    
    # 重みの平方根をサンプルに適用
    sqrt_weights = np.sqrt(weights).reshape(-1, 1)
    X_weighted = X_weighted * sqrt_weights
    
    # PCAを実行
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_weighted)
    
    # 結果を返す
    return X_pca, pca

# アニメーション用の重み計算関数
def calculate_weights(X, focus_point, sigma=1.0, strategy='gaussian'):
    """フォーカスポイントからの距離に基づいて重みを計算する"""
    distances = np.linalg.norm(X - focus_point, axis=1)
    
    if strategy == 'gaussian':
        # ガウス関数を使用した重み: 近いほど重みが大きい
        weights = np.exp(-distances**2 / (2 * sigma**2))
    elif strategy == 'inverse':
        # 距離の逆数: 近いほど重みが大きい
        # 0除算を避けるためにepsilonを追加
        epsilon = 1e-6
        weights = 1.0 / (distances + epsilon)
    else:
        # デフォルト: 均一な重み
        weights = np.ones(X.shape[0])
    
    # 重みを[0, 1]の範囲に正規化
    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-10)
    
    return weights

# アプリの初期化
app = dash.Dash(__name__)

# データの初期化
features, labels = generate_data(data_type='gaussian_clusters', n_samples=500)
df = pd.concat([features, pd.Series(labels, name='label')], axis=1)
feature_columns = features.columns.tolist()

# 標準化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=feature_columns)

# 初期PCA
initial_pca = PCA(n_components=2)
initial_pca_result = initial_pca.fit_transform(features_scaled)
df['PCA1'] = initial_pca_result[:, 0]
df['PCA2'] = initial_pca_result[:, 1]

# PCAの固有ベクトルと特徴量の関係（ローディングスコア）の取得
loadings = pd.DataFrame(
    initial_pca.components_.T * np.sqrt(initial_pca.explained_variance_), 
    columns=['PC1', 'PC2'],
    index=feature_columns
)

# アプリのレイアウト
app.layout = html.Div([
    html.H1('重み付きPCAアニメーションダッシュボード', style={'textAlign': 'center'}),
    
    html.Div([
        # 左カラム: PCAビューとコントロール
        html.Div([
            # PCAグラフ
            dcc.Graph(id='pca-plot', style={'height': '500px'}),
            
            # アニメーションコントロール
            html.Div([
                html.Div([
                    html.Label('フォーカスポイント移動方式:'),
                    dcc.RadioItems(
                        id='movement-type',
                        options=[
                            {'label': '円形軌道', 'value': 'circular'},
                            {'label': 'ジグザグ', 'value': 'zigzag'},
                            {'label': 'クラスタ中心を順番に', 'value': 'cluster_centers'}
                        ],
                        value='circular',
                        labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                    ),
                ], style={'marginBottom': '10px'}),
                
                html.Div([
                    html.Label('重み付け戦略:'),
                    dcc.RadioItems(
                        id='weight-strategy',
                        options=[
                            {'label': 'ガウス関数', 'value': 'gaussian'},
                            {'label': '距離の逆数', 'value': 'inverse'},
                            {'label': '均一 (通常のPCA)', 'value': 'uniform'}
                        ],
                        value='gaussian',
                        labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                    ),
                ], style={'marginBottom': '10px'}),
                
                html.Div([
                    html.Label('ガウス関数の幅 (σ):'),
                    dcc.Slider(
                        id='sigma-slider',
                        min=0.1,
                        max=5.0,
                        step=0.1,
                        value=1.0,
                        marks={i: str(i) for i in range(0, 6)},
                    ),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label('アニメーション速度:'),
                    dcc.Slider(
                        id='animation-speed',
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                        marks={i: str(i) for i in range(1, 11)},
                    ),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Button('アニメーション開始', id='start-button', n_clicks=0, 
                                style={'marginRight': '10px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'padding': '10px 20px'}),
                    html.Button('アニメーション停止', id='stop-button', n_clicks=0,
                                style={'backgroundColor': '#f44336', 'color': 'white', 'border': 'none', 'padding': '10px 20px'}),
                ], style={'textAlign': 'center'}),
                
                # 現在のフレームを表示する隠れた要素
                dcc.Store(id='animation-state', data={'is_running': False, 'frame': 0}),
                dcc.Interval(id='interval-component', interval=500, n_intervals=0, disabled=True),
            ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # 右カラム: パラレルコーディネートと表
        html.Div([
            # パラレルコーディネート
            html.Div([
                dcc.Graph(id='pcp-plot', style={'height': '300px'}),
            ], style={'height': '50%', 'marginBottom': '10px'}),
            
            # データテーブル
            html.Div([
                dash_table.DataTable(
                    id='data-table',
                    columns=[{'name': col, 'id': col} for col in df.columns],
                    data=df.to_dict('records'),
                    page_size=10,
                    style_table={'height': '290px', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'left', 'fontSize': '12px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    row_selectable='multi',
                    selected_rows=[]
                ),
            ], style={'height': '50%'})
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ], style={'padding': '10px'}),
])

# アニメーション制御のコールバック
@app.callback(
    [Output('interval-component', 'disabled'),
     Output('animation-state', 'data')],
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')],
    [State('animation-state', 'data'),
     State('interval-component', 'disabled')]
)
def control_animation(start_clicks, stop_clicks, animation_state, interval_disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        return interval_disabled, animation_state
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-button' and start_clicks > 0:
        # アニメーション開始
        animation_state['is_running'] = True
        return False, animation_state
    elif button_id == 'stop-button' and stop_clicks > 0:
        # アニメーション停止
        animation_state['is_running'] = False
        return True, animation_state
    
    return interval_disabled, animation_state

# アニメーションの間隔を設定するコールバック
@app.callback(
    Output('interval-component', 'interval'),
    [Input('animation-speed', 'value')]
)
def update_interval(speed):
    # 速度が上がるほど間隔は短くなる（速くなる）
    return 1100 - speed * 100  # 1000ms~100ms

# 重み付きPCAの更新とプロット生成のコールバック
@app.callback(
    [Output('pca-plot', 'figure'),
     Output('pcp-plot', 'figure'),
     Output('animation-state', 'data', allow_duplicate=True)],
    [Input('interval-component', 'n_intervals'),
     Input('data-table', 'selected_rows')],
    [State('animation-state', 'data'),
     State('movement-type', 'value'),
     State('weight-strategy', 'value'),
     State('sigma-slider', 'value')],
    prevent_initial_call=True
)
def update_plots(n_intervals, selected_rows, animation_state, movement_type, weight_strategy, sigma):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # 特徴量データ
    X = features_scaled
    
    # フォーカスポイントのアニメーション
    if trigger == 'interval-component' and animation_state['is_running']:
        frame = (animation_state['frame'] + 1) % 100
        animation_state['frame'] = frame
        t = frame / 100.0  # 0から1の範囲に正規化
        
        # フォーカスポイントの軌道を定義
        if movement_type == 'circular':
            # 円形の軌道
            radius = 2.0
            center = np.mean(X, axis=0)
            if len(center) >= 2:
                focus_x = center[0] + radius * np.cos(2 * np.pi * t)
                focus_y = center[1] + radius * np.sin(2 * np.pi * t)
                focus_point = np.array([focus_x, focus_y] + [0] * (X.shape[1] - 2))
            else:
                focus_point = center
        elif movement_type == 'zigzag':
            # ジグザグ軌道
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            range_vals = max_vals - min_vals
            
            # X方向に往復
            if t < 0.25:  # 0->0.25: 左から右へ
                pos = t * 4
                focus_point = min_vals + pos * range_vals
            elif t < 0.5:  # 0.25->0.5: 右から左へ
                pos = 1 - (t - 0.25) * 4
                focus_point = min_vals + pos * range_vals
            elif t < 0.75:  # 0.5->0.75: 下から上へ
                pos_x = (t - 0.5) * 4
                pos_y = (t - 0.5) * 4
                focus_point = min_vals.copy()
                if len(focus_point) >= 2:
                    focus_point[0] += pos_x * range_vals[0]
                    focus_point[1] += pos_y * range_vals[1]
            else:  # 0.75->1: 上から下へ
                pos_x = 1 - (t - 0.75) * 4
                pos_y = 1 - (t - 0.75) * 4
                focus_point = min_vals.copy()
                if len(focus_point) >= 2:
                    focus_point[0] += pos_x * range_vals[0]
                    focus_point[1] += pos_y * range_vals[1]
        elif movement_type == 'cluster_centers':
            # クラスタ中心を順番に巡回
            from sklearn.cluster import KMeans
            n_clusters = 3
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            cluster_centers = kmeans.cluster_centers_
            
            # 現在のクラスタインデックスを計算
            current_cluster = int(t * n_clusters) % n_clusters
            focus_point = cluster_centers[current_cluster]
        else:
            # デフォルト: データの中心
            focus_point = np.mean(X, axis=0)
        
        # 重みを計算
        if weight_strategy == 'uniform':
            weights = np.ones(X.shape[0])
        else:
            weights = calculate_weights(X, focus_point, sigma=sigma, strategy=weight_strategy)
        
        # 重み付きPCAを実行
        X_pca, pca_model = weighted_pca(X, weights, n_components=2)
        
        # PCAの結果をデータフレームに追加
        df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        df_pca['label'] = labels
        df_pca['weight'] = weights
        
        # ローディングスコアの計算
        loadings_df = pd.DataFrame(
            pca_model.components_.T * np.sqrt(pca_model.explained_variance_),
            columns=['PC1', 'PC2'],
            index=feature_columns
        )
        
        # PCAプロットの生成
        pca_fig = px.scatter(
            df_pca, x='PCA1', y='PCA2', color='label', size='weight',
            size_max=15, opacity=0.7,
            title=f'重み付きPCA - Frame {frame}'
        )
        
        # フォーカスポイントを追加
        focus_pca = pca_model.transform(focus_point.reshape(1, -1))
        pca_fig.add_trace(go.Scatter(
            x=[focus_pca[0, 0]], y=[focus_pca[0, 1]],
            mode='markers', marker=dict(size=15, color='black', symbol='x'),
            name='Focus Point'
        ))
        
        # 固有ベクトルを表示
        for i, (col, vec) in enumerate(loadings_df.iterrows()):
            pca_fig.add_annotation(
                x=vec['PC1'] * 3, y=vec['PC2'] * 3,
                ax=0, ay=0,
                xref="x", yref="y",
                text=col,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red'
            )
        
        # パラレルコーディネートプロットの生成
        pcp_fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=df_pca['weight'], colorscale='Viridis', 
                         showscale=True, cmin=0, cmax=1),
                dimensions=[dict(
                    label=col,
                    values=features_scaled_df[col]
                ) for col in feature_columns]
            )
        )
        pcp_fig.update_layout(
            title='特徴量の並行座標プロット（重み付き）',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300
        )
    else:
        # テーブル選択時または初期表示時
        selected_indices = selected_rows if selected_rows else list(range(len(df)))
        highlight = np.zeros(len(df))
        highlight[selected_indices] = 1
        
        # PCAプロットの生成
        pca_fig = px.scatter(
            df, x='PCA1', y='PCA2', color='label',
            title='PCA結果 - 選択データハイライト'
        )
        
        # 選択されたポイントをハイライト
        if selected_rows:
            selected_df = df.iloc[selected_rows]
            pca_fig.add_trace(go.Scatter(
                x=selected_df['PCA1'], y=selected_df['PCA2'],
                mode='markers', marker=dict(size=12, color='black', symbol='circle-open'),
                name='Selected'
            ))
        
        # 固有ベクトルを表示
        for i, (col, vec) in enumerate(loadings.iterrows()):
            pca_fig.add_annotation(
                x=vec['PC1'] * 3, y=vec['PC2'] * 3,
                ax=0, ay=0,
                xref="x", yref="y",
                text=col,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red'
            )
        
        # パラレルコーディネートプロットの生成
        pcp_fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=highlight, colorscale=['lightgray', 'red'], 
                         showscale=True, cmin=0, cmax=1),
                dimensions=[dict(
                    label=col,
                    values=features_scaled_df[col]
                ) for col in feature_columns]
            )
        )
        pcp_fig.update_layout(
            title='特徴量の並行座標プロット（選択ハイライト）',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300
        )
    
    return pca_fig, pcp_fig, animation_state

# サーバーの起動
if __name__ == '__main__':
    app.run_server(debug=True, port=8055)