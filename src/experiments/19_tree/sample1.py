import dash
from dash import html
from dash import dcc

# 1. Dashアプリケーションを初期化
app = dash.Dash(__name__)

# 2. アプリケーションのレイアウトを定義
# レイアウトは、ブラウザに表示されるコンテンツの構造を決定します。
app.layout = html.Div(
    style={
        'textAlign': 'center',
        'marginTop': '50px',
        'fontFamily': 'Arial, sans-serif'
    },
    children=[
        # タイトルを表示
        html.H1("Hello, Dash! (最小構成)", style={'color': '#333'}),

        # 簡単な段落を追加
        html.P("これはPythonとFlaskの上に構築されたWebアプリケーションです。", style={'color': '#555'}),

        # (オプション) グラフコンポーネントのプレースホルダー
        dcc.Graph(
            id='example-graph',
            figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'Montréal'},
                ],
                'layout': {
                    'title': '簡単なサンプルグラフ'
                }
            }
        )
    ]
)

# 3. サーバーを起動
if __name__ == '__main__':
    # host='0.0.0.0'で外部からのアクセスも許可
    app.run_server(debug=True, host='0.0.0.0', port=8050)