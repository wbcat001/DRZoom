# Dashと必要なライブラリのインポート
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


# サンプルデータの作成
iris = px.data.iris()

# 主成分分析（PCA）で2次元に次元削減
pca = PCA(n_components=2)
X_pca = pca.fit_transform(iris.drop('species', axis=1))
iris['PCA1'] = X_pca[:, 0]
iris['PCA2'] = X_pca[:, 1]



fig = px.scatter(
                    iris, x='PCA1', y='PCA2', color='species',
                    custom_data=['species'],
                )
fig.update_layout(clickmode='event+select')

app = dash.Dash(__name__)
# CSS Gridレイアウトを使ったビュー配置
app.layout = html.Div([
    html.Div([
        # 左: 次元削減ビュー
        html.Div([
            dcc.Graph(
                id='scatter',
                figure=fig,
                style={'height': '100%', 'width': '100%'}
            )
        ], style={
            'gridArea': 'left',
            'aspectRatio': '1/1',
            'minWidth': '350px',
            'minHeight': '350px',
            'padding': '8px',
        }),
        # 右: 上下分割
        html.Div([
            html.Div([
                dcc.Graph(
                    id='pcp',
                    style={'height': '100%', 'width': '100%'}
                )
            ], style={'height': '50%', 'paddingBottom': '8px'}),
            html.Div([
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in iris.columns],
                    page_size=5,
                    style_table={'height': '200px', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'left', 'fontSize': 12}
                )
            ], style={'height': '50%'})
        ], style={
            'gridArea': 'right',
            'display': 'flex',
            'flexDirection': 'column',
            'minWidth': '350px',
            'padding': '8px',
        })
    ], style={
        'display': 'grid',
        'gridTemplateAreas': '"left right"',
        'gridTemplateColumns': '1fr 1fr',
        'gap': '0px',
        'height': '600px',
        'width': '100%'
    })
])

# 次元削減ビューで選択した点の主成分をPCPに反映、クラスを表に反映
def get_selected_indices(selectedData):
    if selectedData and 'points' in selectedData:
        return [p['pointIndex'] for p in selectedData['points']]
    return []

@app.callback(
    [Output('pcp', 'figure'), Output('table', 'data')],
    [Input('scatter', 'clickData')]
)
def update_views(clickData):
    indices = get_selected_indices(clickData)
    iris_copy = iris.copy()
    # species_id列を追加（Setosa:0, Versicolor:1, Virginica:2）
    iris_copy['species_id'] = iris_copy['species'].astype('category').cat.codes
    # 色分け
    colorscale = [[0, 'purple'], [0.5, 'lightseagreen'], [1, 'gold']]
    # 選択データの強調（太さやopacityはParcoordsで直接不可なので、色で表現）
    line_color = iris_copy['species_id']
    
    pcp_fig = go.Figure(data=[
        go.Parcoords(
        line=dict(color=line_color, colorscale=colorscale, cmin=-1, cmax=2),
        dimensions=[
            dict(label='Sepal Width', values=iris_copy['sepal_width']),
            dict(label='Sepal Length', values=iris_copy['sepal_length']),
            dict(label='Petal Width', values=iris_copy['petal_width']),
            dict(label='Petal Length', values=iris_copy['petal_length'])
        ]

    ),
    # 選択したデータのみ表示するPCP(1つ)
    go.Parcoords(
        line=dict(color=0, colorscale=[[0, "red"]], cmin=-1, cmax=2),
        
        dimensions=[
            dict(label='Sepal Width', values=iris_copy['sepal_width'].iloc[indices]),
            dict(label='Sepal Length', values=iris_copy['sepal_length'].iloc[indices]),
            dict(label='Petal Width', values=iris_copy['petal_width'].iloc[indices]),
            dict(label='Petal Length', values=iris_copy['petal_length'].iloc[indices])
        ]  
    )
    ])
    pcp_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    # 表: 選択データのみ表示
    if indices:
        filtered = iris.iloc[indices]
    else:
        filtered = iris
    table_data = filtered.to_dict('records')
    return pcp_fig, table_data
    return px.scatter(), []

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True, port=8050)