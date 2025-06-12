import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import pandas as pd

# サンプルデータ
iris = px.data.iris()
fig = px.scatter(iris, x='sepal_width', y='sepal_length', color='species')
fig.update_layout(clickmode='event+select')

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='scatter', figure=fig)
])

@app.callback(
    Output('scatter', 'figure'),
    Input('scatter', 'clickData')
)
def print_click(clickData):
    print('Clicked:', clickData)
    # クリックしてもグラフ自体は変えない
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, port=8051)
