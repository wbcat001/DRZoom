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

# Store for selected points
selected_points_store = []

# App layout
app.layout = html.Div([
    html.H1("Region-Based Weighted PCA Dashboard for MNIST Data"),
    
    html.Div([
        html.Div([
            html.H3("PCA Visualization"),
            html.P("Click and drag to select a region, or use lasso/box select tools"),
            dcc.Graph(
                id='pca-scatter-plot', 
                style={'height': '600px'},
                config={
                    'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'autoScale2d']
                }
            ),
            
            html.Div([
                html.Label("Selection Method:"),
                dcc.RadioItems(
                    id='selection-method',
                    options=[
                        {'label': 'Box Select', 'value': 'box'},
                        {'label': 'Lasso Select', 'value': 'lasso'},
                        {'label': 'Click Select', 'value': 'click'}
                    ],
                    value='box',
                    inline=True
                ),
            ], style={'margin': '10px 0'}),
            
            html.Div([
                html.Button('Clear Selection', id='clear-selection', n_clicks=0),
                html.Button('Reset', id='reset-button', n_clicks=0),
                html.Button('Run Animation', id='run-animation', n_clicks=0),
            ], style={'margin': '10px 0', 'textAlign': 'center'}),
            
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
            ], style={'width': '100%', 'margin': '10px 0'}),
            
            html.Div([
                html.P(id='selection-info', children="No points selected")
            ], style={'margin': '10px 0', 'fontWeight': 'bold', 'color': 'blue'}),
            
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
                ], style={'textAlign': 'center'}),

                 html.Div([
                html.H4("Clicked Digit"),
                html.Div(id='clicked-digit-info', children="Click on a point to see the digit", 
                        style={'textAlign': 'center', 'margin': '10px 0', 'fontWeight': 'bold'}),
                dcc.Graph(id='digit-image-plot', style={'height': '250px'}),
                html.Div(id='digit-details', children="", 
                        style={'textAlign': 'center', 'margin': '10px 0', 'fontSize': '14px'})
            ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '0 10px'}),
            ], style={'width': '80%', 'display': 'inline-block', 'vertical-align': 'top'}),
            
           
        ])
    ]),
    
    # Store for selected points data
    dcc.Store(id='selected-points-store', data=[]),
    # Store for clicked point data
    dcc.Store(id='clicked-point-store', data={}),
])

# Helper function to get points within a box selection
def get_points_in_box(relayoutData, pca_result):
    print("Relayout Data:", relayoutData)
    print("Relayout Data keys:", list(relayoutData.keys()) if relayoutData else "None")
    
    # Check for box selection in relayoutData
    if relayoutData is None:
        return []
    
    # Handle different types of box selection data structures
    x_range = None
    y_range = None
    
    # Method 1: Direct range specification (zoom/pan operations)
    if 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
        x_range = [relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
        print(f"Found xaxis range: {x_range}")
    if 'yaxis.range[0]' in relayoutData and 'yaxis.range[1]' in relayoutData:
        y_range = [relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']]
        print(f"Found yaxis range: {y_range}")
    
    # Method 2: Selection box structure  
    elif 'selections' in relayoutData and len(relayoutData['selections']) > 0:
        selection = relayoutData['selections'][0]
        print(f"Found selections: {selection}")
        if 'range' in selection:
            x_range = selection['range']['x']
            y_range = selection['range']['y']
            print(f"Selection box - x_range: {x_range}, y_range: {y_range}")
    
    # Method 3: Look for any range-like keys
    else:
        # Check for any selection-related keys
        for key, value in relayoutData.items():
            print(f"Key: {key}, Value: {value}")
            if 'range' in key.lower() or 'select' in key.lower():
                print(f"Potential selection key found: {key} = {value}")
    
    # Apply box selection if ranges are found - using the same logic as notebook
    if x_range is not None and y_range is not None:
        print(f"Applying box selection: x_range={x_range}, y_range={y_range}")
        
        # Use the same logic as the notebook
        target_area_x = np.array([min(x_range), max(x_range)])
        target_area_y = np.array([min(y_range), max(y_range)])
        
        # Find points within the target area (same as notebook logic)
        target_index = np.where((pca_result[:, 0] >= target_area_x[0]) & (pca_result[:, 0] <= target_area_x[1]) &
                               (pca_result[:, 1] >= target_area_y[0]) & (pca_result[:, 1] <= target_area_y[1]))[0]
        
        print(f"Box selection applied: found {len(target_index)} points")
        print(f"First 5 selected indices: {target_index[:5].tolist() if len(target_index) > 0 else []}")
        
        # Verify by printing coordinates of selected points
        if len(target_index) > 0:
            print("Verification - selected point coordinates:")
            for i in range(min(3, len(target_index))):
                idx = target_index[i]
                x_coord = pca_result[idx, 0]
                y_coord = pca_result[idx, 1]
                print(f"  Point {i}: index={idx}, coord=({x_coord:.3f}, {y_coord:.3f})")
        
        return target_index.tolist()
    
    print("No valid box selection found")
    return []

# Helper function to get points within a lasso selection
def get_points_in_lasso(selectedData, pca_result):
    print("Selected Data:", selectedData)
    if selectedData and 'points' in selectedData:
        # Method 1: Try to use pointIndex if available
        selected_indices = []
        selected_coords = []
        
        for i, point in enumerate(selectedData['points']):
            print(f"Point {i}: {point}")
            
            # Collect coordinates from selectedData
            if 'x' in point and 'y' in point:
                selected_coords.append((point['x'], point['y']))
            
            # Try to get index
            if 'pointIndex' in point:
                selected_indices.append(point['pointIndex'])
            elif 'pointNumber' in point:
                selected_indices.append(point['pointNumber'])
            elif 'customdata' in point:
                selected_indices.append(point['customdata'])
        
        print(f"Selected coordinates: {selected_coords[:5]}")  # Show first 5
        print(f"Extracted indices: {selected_indices[:5]}")   # Show first 5
        
        # Method 2: If coordinates are available but indices seem wrong, 
        # find closest points by coordinate matching
        if selected_coords and len(selected_coords) > 0:
            print("Verifying selection by coordinate matching...")
            verified_indices = []
            
            for coord in selected_coords:
                # Find the closest point in PCA result
                distances = np.sqrt((pca_result[:, 0] - coord[0])**2 + (pca_result[:, 1] - coord[1])**2)
                closest_idx = np.argmin(distances)
                min_distance = distances[closest_idx]
                
                if min_distance < 0.01:  # Tolerance for coordinate matching
                    verified_indices.append(closest_idx)
                    print(f"  Coord ({coord[0]:.3f}, {coord[1]:.3f}) -> index {closest_idx} (distance: {min_distance:.6f})")
                else:
                    print(f"  Warning: Coord ({coord[0]:.3f}, {coord[1]:.3f}) -> closest index {closest_idx} but distance {min_distance:.6f} is large")
            
            if len(verified_indices) > 0:
                print(f"Using coordinate-verified indices: {len(verified_indices)} points")
                return verified_indices
        
        # Method 3: Fallback to original indices if available
        if selected_indices:
            # Validate indices
            valid_indices = [idx for idx in selected_indices if 0 <= idx < len(pca_result)]
            if len(valid_indices) != len(selected_indices):
                print(f"Warning: Some indices were out of range. Valid: {len(valid_indices)}, Total: {len(selected_indices)}")
            
            if len(valid_indices) > 0:
                print(f"Using original indices: {len(valid_indices)} points")
                return valid_indices
        
    return []

# Callback to update the PCA scatter plot and handle selections
@app.callback(
    [Output('pca-scatter-plot', 'figure'),
     Output('selected-points-store', 'data'),
     Output('selection-info', 'children')],
    [Input('clear-selection', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('pca-scatter-plot', 'relayoutData'),
     Input('pca-scatter-plot', 'selectedData'),
     Input('selection-method', 'value')],
    [State('selected-points-store', 'data')],
    prevent_initial_call=True
)
def update_pca_scatter(clear_clicks, reset_clicks, relayoutData, selectedData, selection_method, stored_selected):
    ctx = dash.callback_context
    selected_indices = stored_selected.copy() if stored_selected else []
    
    # Debug print
    print(f"Callback triggered: {ctx.triggered}")
    print(f"Selection method: {selection_method}")
    print(f"RelayoutData exists: {relayoutData is not None}")
    print(f"SelectedData exists: {selectedData is not None}")
    
    # Handle clear selection
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'clear-selection.n_clicks':
        print("Clearing selection")
        selected_indices = []
    
    # Handle reset
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-button.n_clicks':
        print("Resetting")
        selected_indices = []
    
    # Handle selections based on method and triggered input
    elif ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id']
        print(f"Trigger ID: {trigger_id}")
        
        # Handle relayoutData (includes box selections, zoom, pan, etc.)
        if trigger_id == 'pca-scatter-plot.relayoutData' and relayoutData:
            print("Processing relayoutData")
            if selection_method == 'box':
                print("Calling get_points_in_box")
                new_indices = get_points_in_box(relayoutData, original_pca_result)
                print(f"Box selection returned: {len(new_indices) if new_indices else 0} points")
                if new_indices:
                    selected_indices = new_indices
        
        # Handle selectedData (includes lasso, click selections)
        elif trigger_id == 'pca-scatter-plot.selectedData' and selectedData:
            print("Processing selectedData")
            if selection_method in ['lasso', 'click']:
                print("Calling get_points_in_lasso")
                new_indices = get_points_in_lasso(selectedData, original_pca_result)
                print(f"Lasso/click selection returned: {len(new_indices) if new_indices else 0} points")
                
                # Debug: Compare expected vs actual coordinates
                if new_indices and 'points' in selectedData:
                    print("Coordinate comparison:")
                    for i, point in enumerate(selectedData['points'][:3]):  # Check first 3 points
                        if i < len(new_indices):
                            idx = new_indices[i]
                            actual_x = original_pca_result[idx, 0]
                            actual_y = original_pca_result[idx, 1]
                            print(f"  Selected point {i}:")
                            print(f"    From selectedData: x={point.get('x', 'N/A')}, y={point.get('y', 'N/A')}")
                            print(f"    From PCA result[{idx}]: x={actual_x:.3f}, y={actual_y:.3f}")
                
                if new_indices:
                    selected_indices = new_indices
            # Even for box method, check if selectedData has meaningful selection
            elif selection_method == 'box' and selectedData.get('points') and len(selectedData['points']) > 0:
                print("Box method but using selectedData as fallback")
                new_indices = get_points_in_lasso(selectedData, original_pca_result)
                print(f"Box fallback selection returned: {len(new_indices) if new_indices else 0} points")
                if new_indices:
                    selected_indices = new_indices
    
    # Create the scatter plot with all points
    # Create a DataFrame to avoid pandas grouping warnings
    df_plot = pd.DataFrame({
        'x': original_pca_result[:, 0],
        'y': original_pca_result[:, 1], 
        'digit': [str(int(label)) for label in metadata],
        'index': list(range(len(metadata))),
        'point_id': list(range(len(metadata)))  # Explicit point ID for selection
    })
    
    fig = px.scatter(
        df_plot,
        x='x', 
        y='y',
        color='digit',
        color_discrete_map={str(i): color_map[i] for i in range(10)},
        labels={'color': 'Digit', 'x': 'PC1', 'y': 'PC2'},
        hover_data={'index': True, 'point_id': False},
        custom_data=['point_id']  # Include point ID in custom data
    )
    
    # Highlight selected points
    if selected_indices:
        selected_x = original_pca_result[selected_indices, 0]
        selected_y = original_pca_result[selected_indices, 1]
        
        # Debug: Print selected points information
        print(f"Highlighting {len(selected_indices)} selected points:")
        for i, idx in enumerate(selected_indices[:5]):  # Show first 5 points
            print(f"  Point {i}: Index={idx}, Coord=({selected_x[i]:.3f}, {selected_y[i]:.3f}), Digit={metadata[idx]}")
        if len(selected_indices) > 5:
            print(f"  ... and {len(selected_indices) - 5} more points")
        
        # Add highlighted points
        fig.add_trace(go.Scatter(
            x=selected_x,
            y=selected_y,
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle-open', line=dict(width=3)),
            name='Selected Points',
            showlegend=True,
            text=[f"Idx:{idx}, Digit:{metadata[idx]}" for idx in selected_indices],
            hovertemplate="<b>Selected Point</b><br>" +
                         "Index: %{text}<br>" +
                         "PC1: %{x:.3f}<br>" +
                         "PC2: %{y:.3f}<br>" +
                         "<extra></extra>"
        ))
        
        # Add bounding box around selected region
        if len(selected_indices) > 1:
            x_min, x_max = np.min(selected_x), np.max(selected_x)
            y_min, y_max = np.min(selected_y), np.max(selected_y)
            
            fig.add_shape(
                type="rect",
                x0=x_min-0.1, x1=x_max+0.1,
                y0=y_min-0.1, y1=y_max+0.1,
                line=dict(color="Red", width=2, dash="dash"),
                fillcolor="rgba(255, 0, 0, 0.1)",
                name="Selected Region"
            )
    
    # Update layout based on selection method
    if selection_method == 'box':
        fig.update_layout(dragmode='select')
    elif selection_method == 'lasso':
        fig.update_layout(dragmode='lasso')
    else:  # click
        fig.update_layout(dragmode='pan')
    
    fig.update_layout(
        title=f"PCA of MNIST Data (Selection Method: {selection_method.title()})",
        width=700,
        height=600,
        legend=dict(
            title="Legend",
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=80),  # Add more bottom margin for the legend
        selectdirection='d'
    )
    
    # Create selection info text
    if selected_indices:
        selected_digits = [int(metadata[i]) for i in selected_indices]
        digit_counts = {i: selected_digits.count(i) for i in range(10) if selected_digits.count(i) > 0}
        info_text = f"Selected {len(selected_indices)} points using {selection_method} method. Digits: {digit_counts}"
    else:
        info_text = f"No points selected. Current method: {selection_method}. Try dragging to select a region!"
    
    return fig, selected_indices, info_text

# Function to run weighted PCA on selected region
def run_weighted_pca_region(selected_indices, num_iterations=30):
    if not selected_indices:
        return [], [], []
    
    target_data = high_dim_data[selected_indices]
    
    results = []
    ccrs = []
    combined_data = high_dim_data.copy()
    
    for i in tqdm(range(num_iterations)):
        if i != 0:
            # Combine original data with target data (weighting by repeating)
            weight_factor = max(1, int(np.sqrt(i)))  # Gradually increase weight
            repeated_target = np.tile(target_data, (weight_factor, 1))
            combined_data = np.concatenate([combined_data, repeated_target], axis=0)
        
        # Normalize and run PCA
        norm_data = combined_data / np.linalg.norm(combined_data, axis=1, keepdims=True)
        pca_temp = PCA(n_components=2)
        pca_result = pca_temp.fit_transform(norm_data)[:len(high_dim_data)]
        
        # Store results and explained variance ratio
        results.append(pca_result)
        ccrs.append(pca_temp.explained_variance_ratio_.sum())
    
    # Align projections using Procrustes analysis
    for i in range(len(results)):
        # Ensure consistent orientation
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
    [Input('run-animation', 'n_clicks')],
    [State('selected-points-store', 'data'),
     State('num-iterations-slider', 'value')],
    prevent_initial_call=True
)
def run_animation(n_clicks, selected_indices, num_iterations):
    if n_clicks == 0:
        # Return empty figures on initial load
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "Select a region and run animation to see results",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 20}
                }
            ]
        )
        return empty_fig, empty_fig, empty_fig, num_iterations-1, go.Figure(), ""
    
    if not selected_indices:
        error_fig = go.Figure()
        error_fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "Please select a region first!",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 20, "color": "red"}
                }
            ]
        )
        return error_fig, error_fig, error_fig, num_iterations-1, go.Figure(), ""
    
    # Run weighted PCA on selected region
    results, ccrs, distances = run_weighted_pca_region(selected_indices, num_iterations)
    
    if not results:
        return go.Figure(), go.Figure(), go.Figure(), num_iterations-1, go.Figure(), ""
    
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
    
    # Highlight selected points
    fig.add_trace(go.Scatter(
        x=results[0][selected_indices, 0],
        y=results[0][selected_indices, 1],
        mode='markers',
        marker=dict(size=8, color='red', symbol='circle-open', line=dict(width=2)),
        name='Selected Region',
        text=[f"Selected Digit: {metadata[i]}" for i in selected_indices],
        customdata=selected_indices,
        hovertemplate="<b>Selected - %{text}</b><br>" +
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
                        x=results[i][selected_indices, 0],
                        y=results[i][selected_indices, 1],
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='circle-open', line=dict(width=2)),
                        name='Selected Region',
                        text=[f"Selected Digit: {metadata[idx]}" for idx in selected_indices],
                        customdata=selected_indices,
                        hovertemplate="<b>Selected - %{text}</b><br>" +
                                     "PC1: %{x:.3f}<br>" +
                                     "PC2: %{y:.3f}<br>" +
                                     "Index: %{customdata}<br>" +
                                     "<extra></extra>",
                    )
                ],
                name=str(i)
            )
        )
    
    fig.frames = frames
    
    fig.update_layout(
        title=f"Weighted PCA Animation (Selected Region: {len(selected_indices)} points)",
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
    if distances:
        dist_fig = px.line(
            x=list(range(len(distances))), 
            y=distances,
            labels={"x": "Iteration", "y": "Distance"}
        )
        dist_fig.update_layout(
            title="Distance Moved Between Consecutive Frames",
            height=300
        )
    else:
        dist_fig = go.Figure()
    
    # Create CCR plot
    if ccrs:
        ccr_fig = px.line(
            x=list(range(len(ccrs))), 
            y=ccrs,
            labels={"x": "Iteration", "y": "Explained Variance Ratio"}
        )
        ccr_fig.update_layout(
            title="Cumulative Explained Variance Ratio",
            height=300
        )
    else:
        ccr_fig = go.Figure()
    
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

# Callback to handle clicks on animation plot and display digit image
@app.callback(
    [Output('clicked-digit-info', 'children'),
     Output('digit-image-plot', 'figure', allow_duplicate=True),
     Output('digit-details', 'children', allow_duplicate=True)],
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
    point_index = point['customdata']
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8052)
