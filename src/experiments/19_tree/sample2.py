import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import numpy as np
import pandas as pd

# データ作成
np.random.seed(42)
df = pd.DataFrame({
    "x": np.random.randn(200),
    "y": np.random.randn(200),
    "label": np.random.choice(["A", "B", "C"], 200)
})
fig = px.scatter(df, x="x", y="y", color="label")
fig.update_layout(dragmode='lasso')

# Dash アプリ初期化
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("Scatter Lasso Selection Test"),
    html.P("ドラッグしてラッソ選択してみてください"),
    dcc.Graph(
        id="scatter",
        figure=fig,
        
    ),
    html.Div(id="output", style={"marginTop": 20, "fontWeight": "bold", "color": "blue"})
])

# コールバック: 選択された点を取得
@app.callback(
    Output("output", "children"),
    Input("scatter", "selectedData")
)
def display_selected_points(selectedData):
    if selectedData is None:
        return "まだ何も選択されていません。"
    points = selectedData["points"]
    indices = [p["pointIndex"] for p in points]
    return f"選択されたポイント数: {len(indices)} (例: {indices[:10]})"

if __name__ == "__main__":
    app.run_server(debug=True, port=8052)
