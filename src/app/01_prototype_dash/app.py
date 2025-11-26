import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
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
    # 簡単な階層構造データ
    nodes = []
    # リーフノード (データポイント)
    for i in range(10):
        nodes.append({
            'x': i,
            'y': 0,
            'size': 1,
            'id': i,
            'type': 'leaf'
        })
    
    # 内部ノード
    internal_nodes = [
        {'x': 1, 'y': 1, 'size': 3, 'id': 10, 'type': 'internal'},
        {'x': 4, 'y': 1, 'size': 2, 'id': 11, 'type': 'internal'},
        {'x': 7, 'y': 1, 'size': 4, 'id': 12, 'type': 'internal'},
        {'x': 2.5, 'y': 2, 'size': 5, 'id': 13, 'type': 'internal'},
        {'x': 5.5, 'y': 2, 'size': 6, 'id': 14, 'type': 'internal'},
        {'x': 4, 'y': 3, 'size': 11, 'id': 15, 'type': 'root'}
    ]
    
    nodes.extend(internal_nodes)
    return nodes

# ダミーデータ生成
DR_DUMMY_DATA = generate_dr_dummy_data(100)
DENDROGRAM_DUMMY_DATA = generate_dendrogram_dummy_data()

# Dashアプリケーション初期化
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP] 
)

# グラフ無限拡大を防ぐためのHTML/Bodyの高さ設定
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <style>
            /* ビューポートの高さを確実に使用するように設定 */
            html, body, #react-entry-point {
                height: 100%;
                margin: 0;
            }
            .dbc-container-fluid {
                height: 100%;
            }
        </style>
    </body>
</html>
'''

# レイアウト定義（DBC版 - グラフ無限拡大対策済み）
if DBC_AVAILABLE:
    app.layout = dbc.Container([
        # Rowに h-100 を適用し、親コンテナ (dbc.Container) の高さを継承
        dbc.Row([
            # A: Control Panel (Col: 2)
            dbc.Col([
                # h-100 と d-flex flex-column で高さ管理
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Control Panel", className="text-center mb-3"),
                        
                        # データセット選択
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
                        
                        # 次元削減手法選択
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
                        
                        # パラメータ設定
                        dbc.Row([
                            dbc.Label("Parameters:", className="fw-bold"),
                            # flex-grow-1: 残りのスペースを占有
                            html.Div(id='parameter-settings', className="flex-grow-1 overflow-auto") 
                        ], className="mb-3 d-flex flex-column"), 
                        
                        # 実行ボタン
                        dbc.Button(
                            'Run Analysis',
                            id='execute-button',
                            n_clicks=0,
                            color="primary",
                            size="lg",
                            className="w-100 mt-auto" # mt-autoで実行ボタンを下に寄せる
                        )
                    ])
                ], className="h-100 d-flex flex-column"), # h-100でRowの高さを継承
            ], width=1, className="p-2 h-100"), # Colにも h-100 が必要
            
            # B: DR View Area (Col: 4)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("DR Visualization", className="text-center mb-1"),
                        
                        # インタラクションモード切替
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
                        
                        # DR可視化プロット
                        # h-100とflex-grow-1で、親CardBody内の残りのスペースを占有
                        dcc.Graph(id='dr-visualization-plot', className="flex-grow-1") 
                    ], className="d-flex flex-column p-3 h-100")
                # calc(100vh - 40px) は、ビューポート全体からRowの上下のパディングを引いたおおよその高さ
                ], style={'height': 'calc(100vh - 40px)'}, className="h-100"), 
            ], width=5, className="p-2 h-100"),
            
            # C: Dendrogram Area (Col: 4)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Cluster Dendrogram", className="text-center mb-1"),
                        
                        # インタラクションモード切替とオプション
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
                        
                        # デンドログラムプロット
                        dcc.Graph(id='dendrogram-plot', className="flex-grow-1")
                    ], className="d-flex flex-column p-3 h-100")
                ], style={'height': 'calc(100vh - 40px)'}, className="h-100"),
            ], width=5, className="p-2 h-100"),
            
            # D: Detail & Info Panel (Col: 2)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Detail & Info", className="text-center mb-3"),
                        
                        # タブコンポーネント
                        dbc.Tabs(
                            id='detail-info-tabs',
                            active_tab='tab-point-details',
                            children=[
                                dbc.Tab(label='Point Details', tab_id='tab-point-details'),
                                dbc.Tab(label='Selection Stats', tab_id='tab-selection-stats'),
                                dbc.Tab(label='System Log', tab_id='tab-system-log')
                            ]
                        ),
                        
                        # タブコンテンツ
                        html.Div(id='detail-panel-content', className="mt-3 flex-grow-1 overflow-auto")
                    ], className="d-flex flex-column p-3 h-100")
                ], style={'height': 'calc(100vh - 40px)'}, className="h-100"),
            ], width=1, className="p-2 h-100")
        ], className="g-0 h-100") # Rowのパディング(p-2)が相殺されるように、Rowの外側のマージンをg-0でなくす
    ], fluid=True, className="h-100")
else:
    # DBC が利用できない場合の従来のレイアウト
    app.layout = html.Div([
        html.H3("Dash Bootstrap Components not available. Please install: pip install dash-bootstrap-components")
    ])

# コールバック関数定義（以前のロジックを維持）

@app.callback(
    Output('parameter-settings', 'children'),
    Input('dr-method-selector', 'value')
)
def update_parameter_settings(method):
    """選択された次元削減手法に応じてパラメータ設定UIを更新"""
    if method == 'UMAP':
        return html.Div([
            html.Label("n_neighbors:"),
            dcc.Slider(
                id='umap-n-neighbors',
                min=5, max=50, step=1, value=15,
                marks={i: str(i) for i in range(5, 51, 10)},
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            html.Br(),
            html.Label("min_dist:"),
            dcc.Slider(
                id='umap-min-dist',
                min=0.0, max=0.99, step=0.01, value=0.1,
                marks={i/10: f"{i/10:.1f}" for i in range(0, 10, 2)},
                tooltip={'placement': 'bottom', 'always_visible': True}
            )
        ])
    elif method == 'TSNE':
        return html.Div([
            html.Label("perplexity:"),
            dcc.Slider(
                id='tsne-perplexity',
                min=5, max=50, step=1, value=30,
                marks={i: str(i) for i in range(5, 51, 10)},
                tooltip={'placement': 'bottom', 'always_visible': True}
            )
        ])
    elif method == 'PCA':
        return html.Div([
            html.Label("n_components:"),
            dcc.Dropdown(
                id='pca-n-components',
                options=[
                    {'label': '2', 'value': 2},
                    {'label': '3', 'value': 3}
                ],
                value=2
            )
        ])
    return html.Div()

# グラフ描画（go.Scatter版でcustomdata問題を解決）
@app.callback(
    Output('dr-visualization-plot', 'figure'),
    [Input('execute-button', 'n_clicks'),
     Input('dr-method-selector', 'value'),
     Input('dr-interaction-mode-toggle', 'value')]
)
def update_dr_plot(n_clicks, method, interaction_mode):
    """DR可視化プロットを更新"""
    df = pd.DataFrame(DR_DUMMY_DATA)
    
    fig = go.Figure()

    # ラベルごとにトレースを分けて描画
    for label in df['label'].unique():
        df_group = df[df['label'] == label]
        
        # customdataをリストのリスト [[id1], [id2], ...] の形式で明示的に生成
        custom_ids = df_group['id'].apply(lambda x: [x]).tolist()
        
        fig.add_trace(go.Scatter(
            x=df_group['x'],
            y=df_group['y'],
            mode='markers',
            name=label,
            customdata=custom_ids,
            marker=dict(size=8, opacity=0.8),
            hovertemplate=(
                'Label: %{data.name}<br>'
                'ID: %{customdata[0]}<br>'
                'X: %{x:.3f}<br>'
                'Y: %{y:.3f}<extra></extra>'
            )
        ))
    
    # フィードバックに基づくコンパクト化設定
    fig.update_layout(
        # height=500,  <-- DBCで高さをCSSで管理するため削除
        showlegend=False,  # 凡例無効化
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        margin=dict(l=10, r=10, t=10, b=10),  # マージン最小化
        plot_bgcolor='#f8f9fa',  # コンテナの色に合わせて背景色統一
        paper_bgcolor='#f8f9fa'  # 外枠背景色統一
    )
    
    # インタラクションモードに応じた設定
    if interaction_mode == 'brush':
        drag_mode = 'select'
    else:
        drag_mode = 'zoom'
    
    fig.update_layout(dragmode=drag_mode)
    
    return fig

@app.callback(
    Output('dendrogram-plot', 'figure'),
    [Input('dendro-width-option-toggle', 'value')]
)
def update_dendrogram_plot(width_options):
    """デンドログラムプロットを更新"""
    prop_width = 'prop_width' in width_options
    
    # 簡単なデンドログラム風の可視化
    fig = go.Figure()
    
    # デンドログラム線を描画
    connections = [
        # レベル1の結合
        ([0, 0, 1], [0, 1, 1]),  # ノード0-1
        ([1, 1, 2], [0, 1, 1]),  # ノード1-2
        ([3, 3, 4], [0, 1, 1]),  # ノード3-4
        ([6, 6, 7], [0, 1, 1]),  # ノード6-7
        ([7, 7, 8], [0, 1, 1]),  # ノード7-8
        ([8, 8, 9], [0, 1, 1]),  # ノード8-9
        
        # レベル2の結合
        ([1, 1, 3.5], [1, 2, 2]),  # グループ0-2とノード3-4
        ([7.5, 7.5, 8.5], [1, 2, 2]),  # グループ6-9
        
        # レベル3の結合
        ([2.25, 2.25, 8], [2, 3, 3]),  # 最終結合
    ]
    
    for x_coords, y_coords in connections:
        line_width = 3 if prop_width else 1
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='lines',
            line=dict(color='black', width=line_width),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # データポイントを描画
    x_positions = list(range(10))
    y_positions = [0] * 10
    
    fig.add_trace(go.Scatter(
        x=x_positions, y=y_positions,
        mode='markers',
        marker=dict(size=8, color='blue'),
        showlegend=False,
        text=[f'Point {i}' for i in range(10)],
        hovertemplate='Point ID: %{text}<extra></extra>'
    ))
    
    # フィードバックに基づくコンパクト化設定
    fig.update_layout(
        xaxis_title='Data Points',
        yaxis_title='Distance',
        # height=500, <-- DBCで高さをCSSで管理するため削除
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),  # マージン最小化
        plot_bgcolor='#f8f9fa',  # コンテナの色に合わせて背景色統一
        paper_bgcolor='#f8f9fa'  # 外枠背景色統一
    )
    
    return fig

@app.callback(
    Output('detail-panel-content', 'children'),
    [Input('detail-info-tabs', 'active_tab'), # <-- ここを 'active_tab' に修正
     Input('dr-visualization-plot', 'clickData'),
     Input('dr-visualization-plot', 'selectedData')]
)
def update_detail_panel(active_tab, click_data, selected_data):
    """詳細パネルの内容を更新"""
    if active_tab == 'tab-point-details':
        if click_data:
            # customdataを確実に取得（go.Scatterへの修正により解決済み）
            point_id = click_data['points'][0]['customdata'][0]
            
            # Labelはgo.Scatterで直接取得できないため、traceのnameから取得（px.scatterの動作を模倣）
            trace_index = click_data['points'][0]['curveNumber']
            # traceNamesはモックなので、ここでは簡略化
            label_mock = ['A', 'B', 'C'][trace_index % 3] 
            
            return html.Div([
                html.H6(f"Selected Point Details (ID: {point_id})", className="fw-bold"),
                dbc.ListGroup([
                    dbc.ListGroupItem(f"ID: {point_id}"),
                    dbc.ListGroupItem(f"X: {click_data['points'][0]['x']:.3f}"),
                    dbc.ListGroupItem(f"Y: {click_data['points'][0]['y']:.3f}"),
                    dbc.ListGroupItem(f"Label: {label_mock}"),
                ], flush=True),
                html.P("Neighbors (k=5): Mock data - Points 1, 5, 12, 23, 34", className="mt-3 small")
            ])
        else:
            return html.Div([
                html.P("Click a point in the DR view to see details.")
            ])
    
    elif active_tab == 'tab-selection-stats':
        if selected_data and selected_data['points']:
            n_selected = len(selected_data['points'])
            return html.Div([
                html.H6("Selection Statistics", className="fw-bold"),
                html.P(f"Selected Points: {n_selected}"),
                html.P("Feature Statistics (Mock):"),
                html.Ul([
                    html.Li("Feature 1: Mean=0.45, Std=0.23"),
                    html.Li("Feature 2: Mean=0.67, Std=0.19"),
                    html.Li("Feature 3: Mean=0.33, Std=0.15")
                ])
            ])
        else:
            return html.Div([
                html.P("Select points in the DR view to see statistics.")
            ])
    
    elif active_tab == 'tab-system-log':
        return html.Div([
            html.H6("System Log", className="fw-bold"),
            html.Div([
                html.P("2024-11-16 10:30:15 - Application started"),
                html.P("2024-11-16 10:30:20 - Dataset loaded: iris (150 samples)"),
                html.P("2024-11-16 10:30:25 - DR method: UMAP initialized"),
                html.P("2024-11-16 10:30:30 - Analysis completed (Mock execution)")
            ], style={'fontSize': '12px', 'fontFamily': 'monospace'})
        ])
    
    return html.Div()

@app.callback(
    Output('detail-info-tabs', 'style'),  # ダミーアウトプット
    Input('execute-button', 'n_clicks')
)
def log_execute_button(n_clicks):
    """実行ボタンが押された際のログ出力（モックアップ）"""
    if n_clicks > 0:
        print(f"実行ボタンが押されました (クリック数: {n_clicks})")
    return {}

if __name__ == '__main__':
    app.run(debug=True, port=8050)