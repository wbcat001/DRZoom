from dash import Dash, dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from evaluate import *
from core3 import *
# サンプルデータの生成
raw_data = np.random.rand(100, 5)

# クラスの初期化
data_manager = DataManager("data/books")
reducer = DimensionalityReducer()
aligner = AlignmentHandler(method="Procrustes")
animator = AnimationManager(data_manager, aligner, reducer)
transition_data = TransitionData(data_manager.data, reducer)

# annotation_manager = AnnotationManager(data_manager, aligner, reducer)
colors = data_manager.get_colors()

def generate_custom_colorscale(n):
    blue = np.array([0, 0, 255])  # 青 (RGB)
    orange = np.array([255, 165, 0])  # オレンジ (RGB)
    colors = [tuple((1 - i / (n - 1)) * blue + (i / (n - 1)) * orange) for i in range(n)]
    colorscale = [(i / (n - 1), f"rgb({int(c[0])}, {int(c[1])}, {int(c[2])})") for i, c in enumerate(colors)]
    return colorscale

## from or to 
def get_plots(data:Data, n=20, colors=colors, from_to="from"):
    visible = True if from_to == "from" else False
    df = data.df.copy()
    df = df.reset_index()

    num_points = len(df)
    split_size = num_points // n
    custom_colorscale = generate_custom_colorscale(n)
    plot_list = []

    ## Draw
    for i in range(n):
        start = i * split_size
        end = (i + 1) * split_size if i < n - 1 else num_points  # 最後のセグメント調整
        segment = df.loc[start:end]
        parts = []
        
        current_part = []
        previous_index = None
        for index, row in segment.iterrows():
            if previous_index is not None and row["index"] != previous_index + 1:
                parts.append(current_part)
                current_part = []
            current_part.append(row)
            previous_index = row["index"]

        if current_part:
            parts.append(current_part)

        # カラースケールからこのセグメントの色を取得
        segment_color = custom_colorscale[i][1]

        for part in parts:
            part_df = pd.DataFrame(part)
            # セグメントをプロット
            
            plot_list.append(go.Scatter(
                x=part_df["x"],
                y=part_df["y"],
                mode='markers', # plot type
                line=dict(
                    color=segment_color,
                    width=2  # ライン幅
                ),
                # merker=dict(
                #     color=segment_color, size=3
                # ),
                showlegend=False,
                name=from_to,
                visible=visible
            ))

    for category in colors.keys():
        filtered = df[df["ERole"] == category] # color

        plot_list.append(go.Scatter(
            x=filtered['x'],
            y=filtered['y'],
            mode="markers",
            marker=dict(color=colors[category], size=4),
            text=filtered["Event"], # hover text
            name=from_to,
            visible=visible
        
        ))
        unique_df = df.drop_duplicates(subset="Event", keep="first")

        plot_list.append(go.Scatter(
            x=unique_df['x'],
            y=unique_df['y'],
            mode="markers",# +text,
            marker=dict(color=colors[category], size=1),
            # text=unique_df["Event"],
            # textfont=dict(size=8),
            name=from_to,
            visible=visible
        
        ))
        

    return plot_list
def generate_fig(transition_data: TransitionData, x_min=-2.5, x_max=2.5, y_min=-2.5, y_max=2.5):
    fig = go.Figure()
    # Todo
    # calc position, make go.Scatter
    frames, transition_data = animator.create_frames(x_min, x_max, y_min, y_max, transition_data)
    

    
    # Todo 
    # get list of go.Scatter, annotation
    plot_from = get_plots(transition_data.from_data, from_to="from")
    plot_to = get_plots(transition_data.to_data, from_to="to")

    len_from, len_to = len(plot_from), len(plot_to)

    # annotate
    # annotation_manager.annotate(fig)
    
    for plot in plot_from + plot_to:
        fig.add_trace(plot)
    fig.frames = [
        go.Frame(data= [go.Scatter(
            x=frame[:, 0],
            y=frame[:, 1],
            mode='markers+lines',
            marker=dict(color='blue', size=2),
            line=dict(color='blue', width=1),
            name='frames',
            # duration= 0 if index == 0 else 3000
        )])
        for index, frame in enumerate(frames)
    ]
    # fig.frames[0]
    # Layout
    x_range0, x_range1, y_range0, y_range1 = transition_data.get_position_range()
    fig.layout = go.Layout(
            xaxis=(dict(range=[x_min, x_max])),
            yaxis=(dict(range=[y_min, y_max])),
            title=dict(text="Start Title"),
          
            # Todo: 
            # アニメーションの始動、遷移後のプロットの表示
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Replay",
                            method="animate",
                             args=[None, {"frame": {"duration": 3000, "redraw": False}, "transition": {"duration": 2500, "easing": "linear"}}, ],
                             execute=True,
                            ),
                        dict(
                            label="Show 'to'",
                            method="restyle",
                            args=[
                                {"visible": [False for _ in range(len_from)] + [True for _ in range(len_to)]},  # "from" を非表示、"to" を表示
                                {"title": "Showing 'to'"},  
                            ],
                        ),
                            ])],
            
        )
    fig.update_layout(width=1000, height=1000)
    
    return fig
fig_default = generate_fig(transition_data)
# Dashアプリケーションの作成
app = Dash(__name__)

app.layout = html.Div([
    # # 次元削減手法の選択
    # html.Label("次元削減手法:"),
    # dcc.Dropdown(
    #     id="reduction-method",
    #     options=[{"label": method, "value": method} for method in ["PCA", "t-SNE"]],
    #     value="PCA"
    # ),
    # # アライメント手法の選択
    # html.Label("アライメント手法:"),
    # dcc.Dropdown(
    #     id="alignment-method",
    #     options=[{"label": "Linear", "value": "linear"}],
    #     value="linear"
    # ),
    # # アニメーション速度
    # html.Label("アニメーション速度 (ms):"),
    # dcc.Slider(
    #     id="animation-speed",
    #     min=100, max=2000, step=100, value=500,
    #     marks={i: f"{i}ms" for i in range(100, 2001, 400)}
    # ),
    # # アニメーションのステップ数
    # html.Label("アニメーションステップ数:"),
    # dcc.Input(id="animation-steps", type="number", value=20),
    # # 更新ボタン
    html.Button("Reset", id="reset-button", n_clicks=0),
    # # グラフ
    dcc.Graph(id="main", figure=fig_default, config={'scrollZoom': True}),
    # インターバルコンポーネント
    dcc.Graph(id="main2"),
    
    dcc.Interval(id="interval", interval=1000, n_intervals=0,max_intervals=10),
    html.Div(id="dummy-output"),
    html.Div(id="dummy-output2"),

])

# サーバーサイドでデータをキャッシュする
# processed_data = data_manager.preprocess()
# reduced_before = None
# reduced_after = None
# frames = None

@app.callback(
    Output('main', 'figure', allow_duplicate=True),
    Input('main', 'relayoutData'),
    prevent_initial_call=True
 
)
def zoom_figure(relayoutData):

    # Get Selected area
    if relayoutData:
        x_min, x_max = relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
        y_min, y_max = relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']

    

    fig = generate_fig(transition_data, x_min, x_max, y_min, y_max)

    return fig


@app.callback(
    Output("main", "figure"),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_animation(n_clicks):
    transition_data.reset()
    fig = generate_fig(transition_data)
    return fig



app.clientside_callback(
    """
    function (n_intervals, fig_data) {
        // グローバルフラグ管理
        if (window.lastFigData === undefined) {
            window.lastFigData = null;
        }
        
        // 新しいfig_dataが渡された場合のみ処理
        if (JSON.stringify(window.lastFigData) !== JSON.stringify(fig_data)) {
            window.lastFigData = fig_data;

            const btn = document.querySelector("#main > div.js-plotly-plot > div > div > svg:nth-child(3) > g.menulayer > g.updatemenu-container > g.updatemenu-header-group > g.updatemenu-button");
            console.log("btn", btn);

            if (btn != null) {
                btn.dispatchEvent(new Event('click'));
            }
        }
        return [];
    }
    """,
    Output('dummy-output', 'children'),  # 必須ダミー
    [Input('interval', 'n_intervals'),  # 定期的に監視
     Input('main', 'figure')]           # figの生成を監視
)



if __name__ == "__main__":
    app.run_server(debug=True)