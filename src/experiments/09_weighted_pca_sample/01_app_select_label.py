import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from scipy.spatial import procrustes
from tqdm import tqdm
# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# MNIST data loading function
def load_mnist_data(num_samples=5000):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    high_dim_data = mnist.data[:num_samples]
    metadata = mnist.target[:num_samples]
    return high_dim_data, metadata

# Global variables to store data
high_dim_data, metadata = load_mnist_data()
normalized_data = high_dim_data / np.linalg.norm(high_dim_data, axis=1, keepdims=True)
pca = PCA(n_components=2)
original_pca_result = pca.fit_transform(normalized_data)
color_map = px.colors.qualitative.Plotly
colors = [color_map[int(label)] for label in metadata]

# Initial PCA result
initial_pca_result = original_pca_result.copy()

# App layout
app.layout = html.Div([
    html.H1("Weighted PCA Dashboard for MNIST Data"),
    
    html.Div([
        html.Div([
            html.H3("PCA Visualization"),
            html.P("Click to select target points or use the digit selector below"),
            dcc.Graph(id='pca-scatter-plot', style={'height': '600px'}),
            
            html.Div([
                html.Label("Select target digit:"),
                dcc.Dropdown(
                    id='target-digit-dropdown',
                    options=[{'label': str(i), 'value': i} for i in range(10)],
                    value=1,
                    clearable=False
                ),
            ], style={'width': '30%', 'display': 'inline-block'}),
            
            html.Div([
                html.Button('Reset', id='reset-button', n_clicks=0),
                html.Button('Run Animation', id='run-animation', n_clicks=0),
            ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),
            
            html.Div([
                html.Label("Number of iterations:"),
                dcc.Slider(
                    id='num-iterations-slider',
                    min=10,
                    max=100,
                    step=10,
                    value=50,
                    marks={i: str(i) for i in range(10, 101, 10)}
                ),
            ], style={'width': '40%', 'display': 'inline-block'}),
        ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.H3("Analysis"),
            dcc.Graph(id='distances-plot', style={'height': '300px'}),
            dcc.Graph(id='ccr-plot', style={'height': '300px'})
        ], style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ]),
    
    html.Div([
        html.H3("Animation Controls"),
        html.Div([
            html.Div([
                dcc.Graph(id='animation-plot', style={'height': '600px'}),
                html.Div([
                    dcc.Slider(
                        id='animation-slider',
                        min=0,
                        max=49,
                        step=1,
                        value=0,
                        marks={i: str(i) for i in range(0, 50, 5)},
                        updatemode='drag'
                    ),
                ]),
                html.Div([
                    html.Button('Play', id='play-button', n_clicks=0),
                    html.Button('Pause', id='pause-button', n_clicks=0),
                    dcc.Interval(
                        id='interval-component',
                        interval=200,  # in milliseconds
                        n_intervals=0,
                        disabled=True
                    ),
                ], style={'textAlign': 'center'})
            ], style={'width': '80%', 'display': 'inline-block', 'vertical-align': 'top'}),
            
            html.Div([
                html.H4("Clicked Digit"),
                html.Div(id='clicked-digit-info', children="Click on a point to see the digit", 
                        style={'textAlign': 'center', 'margin': '10px 0', 'fontWeight': 'bold'}),
                dcc.Graph(id='digit-image-plot', style={'height': '250px'}),
                html.Div(id='digit-details', children="", 
                        style={'textAlign': 'center', 'margin': '10px 0', 'fontSize': '14px'})
            ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '0 10px'})
        ])
    ]),
])

# Callback to update the initial PCA scatter plot
@app.callback(
    Output('pca-scatter-plot', 'figure'),
    [Input('target-digit-dropdown', 'value'),
     Input('reset-button', 'n_clicks')]
)
def update_pca_scatter(target_digit, n_clicks):
    # Create the scatter plot with all points
    fig = px.scatter(
        x=original_pca_result[:, 0], 
        y=original_pca_result[:, 1],
        color=[str(int(label)) for label in metadata],
        color_discrete_map={str(i): color_map[i] for i in range(10)},
        labels={'color': 'Digit'}
    )
    
    # Highlight the selected digit
    target_indices = np.where(metadata.astype(int) == target_digit)[0]
    
    if len(target_indices) > 0:
        target_x = original_pca_result[target_indices, 0]
        target_y = original_pca_result[target_indices, 1]
        
        # Add shape around target digit points
        x_min, x_max = np.min(target_x), np.max(target_x)
        y_min, y_max = np.min(target_y), np.max(target_y)
        
        fig.add_shape(
            type="rect",
            x0=x_min, x1=x_max,
            y0=y_min, y1=y_max,
            line=dict(color="Red", width=2),
            fillcolor="rgba(255, 0, 0, 0.2)",
            name="Target Area"
        )
    
    fig.update_layout(
        title=f"PCA of MNIST Data (Target Digit: {target_digit})",
        width=700,
        height=600,
        legend=dict(
            title="Digit",
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=80)  # Add more bottom margin for the legend
    )
    
    return fig

# Function to run weighted PCA
def run_weighted_pca(target_indices, num_iterations=50):
    target_data = high_dim_data[target_indices]
    
    results = []
    ccrs = []
    combined_data = high_dim_data.copy()
    
    for i in tqdm(range(num_iterations)):
        if i != 0:
            # Combine original data with target data (weighting by repeating)
            # Use same gradual increase in weight as in app_select_area.py
            weight_factor = max(1, int(np.sqrt(i)))  # Gradually increase weight
            repeated_target = np.tile(target_data, (weight_factor, 1))
            combined_data = np.concatenate([combined_data, repeated_target], axis=0)
        
        # Normalize and run PCA
        norm_data = combined_data / np.linalg.norm(combined_data, axis=1, keepdims=True)
        pca_temp = PCA(n_components=2)  # Create new PCA instance for each iteration
        pca_result = pca_temp.fit_transform(norm_data)[:len(high_dim_data)]
        
        # Store results and explained variance ratio
        results.append(pca_result)
        ccrs.append(pca_temp.explained_variance_ratio_.sum())
    
    # Unify sign and align projections
    for i in range(len(results)):
        # Ensure consistent orientation using correlation coefficients
        results[i][:, 0] *= np.sign(np.corrcoef(original_pca_result[:, 0], results[i][:, 0])[0, 1])
        results[i][:, 1] *= np.sign(np.corrcoef(original_pca_result[:, 1], results[i][:, 1])[0, 1])
        # Align using Procrustes
        _, results[i], _ = procrustes(original_pca_result, results[i])
    
    # Calculate distances between consecutive frames
    distances = [np.linalg.norm(results[i+1] - results[i]) for i in range(len(results)-1)]
    
    return results, ccrs, distances

# Callback for running the animation
@app.callback(
    [Output('animation-plot', 'figure'),
     Output('distances-plot', 'figure'),
     Output('ccr-plot', 'figure'),
     Output('animation-slider', 'max'),
     Output('digit-image-plot', 'figure', allow_duplicate=True),
     Output('digit-details', 'children', allow_duplicate=True)],
    [Input('run-animation', 'n_clicks'),
     Input('target-digit-dropdown', 'value')],
    [State('num-iterations-slider', 'value')],
    prevent_initial_call=True
)
def run_animation(n_clicks, target_digit, num_iterations):
    if n_clicks == 0:
        # Return empty figures on initial load
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "Run animation to see results",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 20}
                }
            ]
        )
        return empty_fig, empty_fig, empty_fig, num_iterations-1, go.Figure(), ""
    
    # Get target indices based on selected digit
    target_indices = np.where(metadata.astype(int) == target_digit)[0]
    
    if len(target_indices) == 0:
        return go.Figure(), go.Figure(), go.Figure(), num_iterations-1, go.Figure(), ""
    
    # Run weighted PCA
    results, ccrs, distances = run_weighted_pca(target_indices, num_iterations)
    
    # Create animation figure
    fig = go.Figure()
    
    # Add initial scatter plot
    fig.add_trace(go.Scatter(
        x=results[0][:, 0],
        y=results[0][:, 1],
        mode='markers',
        marker=dict(color=[color_map[int(label)] for label in metadata], size=5),
        text=[f"Digit: {label}" for label in metadata],
        customdata=list(range(len(metadata))),  # Add indices as custom data
        hovertemplate="<b>Digit: %{text}</b><br>" +
                     "PC1: %{x:.3f}<br>" +
                     "PC2: %{y:.3f}<br>" +
                     "Index: %{customdata}<br>" +
                     "<extra></extra>",
        name='All Points'
    ))
    
    # Highlight target points
    fig.add_trace(go.Scatter(
        x=results[0][target_indices, 0],
        y=results[0][target_indices, 1],
        mode='markers',
        marker=dict(size=10, color='red', line=dict(width=2, color='black')),
        name='Target Points',
        text=[f"Target Digit: {metadata[i]}" for i in target_indices],
        customdata=target_indices,
        hovertemplate="<b>Target - %{text}</b><br>" +
                     "PC1: %{x:.3f}<br>" +
                     "PC2: %{y:.3f}<br>" +
                     "Index: %{customdata}<br>" +
                     "<extra></extra>",
    ))
    
    # Create frames for animation
    frames = []
    for i in range(num_iterations):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=results[i][:, 0],
                        y=results[i][:, 1],
                        mode='markers',
                        marker=dict(color=[color_map[int(label)] for label in metadata], size=5),
                        text=[f"Digit: {label}" for label in metadata],
                        customdata=list(range(len(metadata))),
                        hovertemplate="<b>Digit: %{text}</b><br>" +
                                     "PC1: %{x:.3f}<br>" +
                                     "PC2: %{y:.3f}<br>" +
                                     "Index: %{customdata}<br>" +
                                     "<extra></extra>",
                        name='All Points'
                    ),
                    go.Scatter(
                        x=results[i][target_indices, 0],
                        y=results[i][target_indices, 1],
                        mode='markers',
                        marker=dict(size=10, color='red', line=dict(width=2, color='black')),
                        name='Target Points',
                        text=[f"Target Digit: {metadata[idx]}" for idx in target_indices],
                        customdata=target_indices,
                        hovertemplate="<b>Target - %{text}</b><br>" +
                                     "PC1: %{x:.3f}<br>" +
                                     "PC2: %{y:.3f}<br>" +
                                     "Index: %{customdata}<br>" +
                                     "<extra></extra>",
                    )
                ],
                name=str(i)  # Important: name must match the slider step value
            )
        )
    
    fig.frames = frames
    
    fig.update_layout(
        title=f"Weighted PCA Animation (Target: Digit {target_digit})",
        width=1000,
        height=600,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "y": 0
        }],
        sliders=[{
            "steps": [
                {
                    "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": str(i),
                    "method": "animate"
                }
                for i in range(num_iterations)
            ],
            "active": 0,
            "x": 0.1,
            "y": 0,
            "currentvalue": {"prefix": "Frame: ", "visible": True},
            "len": 0.9
        }]
    )
    
    # Create distance plot
    dist_fig = px.line(
        x=list(range(len(distances))), 
        y=distances,
        labels={"x": "Iteration", "y": "Distance"}
    )
    dist_fig.update_layout(
        title="Distance Moved Between Consecutive Frames",
        height=300
    )
    
    # Create CCR plot
    ccr_fig = px.line(
        x=list(range(len(ccrs))), 
        y=ccrs,
        labels={"x": "Iteration", "y": "Explained Variance Ratio"}
    )
    ccr_fig.update_layout(
        title="Cumulative Explained Variance Ratio",
        height=300
    )
    
    # Create initial empty digit display
    initial_digit_fig = go.Figure()
    initial_digit_fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "Click on a point to see digit",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14}
            }
        ],
        width=280,
        height=250,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    
    return fig, dist_fig, ccr_fig, num_iterations-1, initial_digit_fig, ""

# Callback to update animation based on slider
@app.callback(
    Output('animation-plot', 'figure', allow_duplicate=True),
    [Input('animation-slider', 'value')],
    [State('animation-plot', 'figure')],
    prevent_initial_call=True
)
def update_frame(selected_frame, current_fig):
    if not current_fig or not current_fig.get('frames'):
        return dash.no_update
    
    # Update to the selected frame
    return go.Figure(current_fig).update_layout(sliders=[{
        **current_fig['layout']['sliders'][0],
        'active': selected_frame
    }])

# Callbacks for play/pause
@app.callback(
    [Output('interval-component', 'disabled'),
     Output('animation-slider', 'value')],
    [Input('play-button', 'n_clicks'),
     Input('pause-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('animation-slider', 'value'),
     State('animation-slider', 'max'),
     State('interval-component', 'disabled')]
)
def control_animation(play_clicks, pause_clicks, n_intervals, current_frame, max_frame, currently_disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, 0
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'play-button':
        return False, current_frame
    elif button_id == 'pause-button':
        return True, current_frame
    elif button_id == 'interval-component':
        new_frame = current_frame + 1
        if new_frame > max_frame:
            new_frame = 0
        return False, new_frame
    
    return currently_disabled, current_frame

# Callback to handle clicks on animation plot and display digit image
@app.callback(
    [Output('clicked-digit-info', 'children'),
     Output('digit-image-plot', 'figure'),
     Output('digit-details', 'children')],
    [Input('animation-plot', 'clickData')],
    prevent_initial_call=True
)
def display_clicked_digit(clickData):
    if clickData is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "Click on a point in the animation to see the digit",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 16}
                }
            ]
        )
        return "Click on a point to see the digit", empty_fig, ""
    
    # Extract clicked point information
    point = clickData['points'][0]
    
    # Handle different data structures that might come from the click
    if 'customdata' in point:
        point_index = point['customdata']
    elif 'pointIndex' in point:
        point_index = point['pointIndex']
    elif 'pointNumber' in point:
        point_index = point['pointNumber']
    else:
        # If we can't get the index directly, try to match by coordinates
        x, y = point['x'], point['y']
        distances = np.sqrt((original_pca_result[:, 0] - x)**2 + (original_pca_result[:, 1] - y)**2)
        point_index = np.argmin(distances)
    
    # Make sure we have a valid index
    if isinstance(point_index, (list, np.ndarray)) and len(point_index) > 0:
        point_index = point_index[0]
    
    # Get the digit label
    digit_label = int(metadata[point_index])
    
    # Get the 28x28 image
    digit_image = high_dim_data[point_index].reshape(28, 28)
    
    # Create image plot
    fig = go.Figure(data=go.Heatmap(
        z=digit_image,
        colorscale='gray',
        showscale=False,
        hovertemplate="Pixel (%{x}, %{y}): %{z:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Digit: {digit_label}",
        width=280,
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, autorange='reversed')
    )
    
    # Create detailed information
    clicked_info = f"Clicked Digit: {digit_label}"
    details = f"Index: {point_index}\nCoordinates: ({point['x']:.3f}, {point['y']:.3f})"
    
    return clicked_info, fig, details

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
