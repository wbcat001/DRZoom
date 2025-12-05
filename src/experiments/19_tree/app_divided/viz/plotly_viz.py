import plotly.graph_objects as go
import plotly.express as px


def plot_dendrogram_plotly(segments, colors=None, scores=None, is_selecteds=None):
    """Render a dendrogram from precomputed `segments`.

    Parameters:
    - segments: list of ((x1,y1),(x2,y2)) tuples representing line segments
    - colors: optional list of colors per segment
    - scores: optional list of scores to show in hovertext
    - is_selecteds: optional boolean mask per segment indicating highlight

    This function performs only rendering. It does not compute coordinates
    or node relationships — those belong in the `dendro` module. Keep the
    I/O between coordinate computation and rendering as lists/dicts so it's
    straightforward to serialize to JSON for D3 clients.
    """
    fig = go.Figure()
    for i, seg in enumerate(segments):
        x_coords = [seg[0][0], seg[1][0]]
        y_coords = [seg[0][1], seg[1][1]]
        color = 'black' if colors is None else (colors[i] if i < len(colors) else 'black')
        info = 'N/A' if scores is None or i >= len(scores) else f"{scores[i]:.2f}"
        opacity = 1.0
        if is_selecteds is not None:
            if i < len(is_selecteds):
                opacity = 1.0 if is_selecteds[i] else 0.2
            else:
                opacity = 0.2
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color=color, width=1),
            showlegend=False,
            hoverinfo='text' if (is_selecteds is None or (i < len(is_selecteds) and is_selecteds[i])) else 'skip',
            text=f'Segment {i}: ({x_coords[0]:.2f},{y_coords[0]:.2f})→({x_coords[1]:.2f},{y_coords[1]:.2f}) score={info}',
            opacity=opacity
        ))
    fig.update_layout(
        title='Dendrogram',
        xaxis_title='Observation Index',
        yaxis_title='Distance / Height',
        hovermode='closest',
        height=800,
        width=1000,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig


def get_clusters_from_points(point_ids, point_id_map):
    """Map a list of point ids to cluster ids using `point_id_map`.

    Returns a list of unique cluster ids corresponding to the input points.
    """
    cluster_ids = set()
    for pid in point_ids:
        if pid in point_id_map:
            cluster_ids.add(point_id_map[pid])
    return list(cluster_ids)


def build_point_id_map_from_dummy(nodes):
    """Utility: from a simple node list produce a leaf->parent map.

    This helper mirrors the behaviour used in the original notebook for
    small synthetic datasets and is convenient for local testing.
    """
    mapping = {}
    for n in nodes:
        if n.get('type') == 'leaf':
            mapping[n['id']] = n.get('parent', n['id'])
    return mapping


def get_dr_figure(dr_data, interaction_mode='zoom'):
    """Create a Plotly scatter figure from `dr_data` list.

    `dr_data` should be a list of dict-like objects with keys `x`,`y`,`label`,`id`.
    This function is intentionally simple: it produces a Plotly Figure and keeps
    the data representation (list of dicts) independent from the figure so the
    same `dr_data` can be serialized for a D3 client.
    """
    import pandas as pd
    df = pd.DataFrame(dr_data)
    fig = go.Figure()
    for label in df['label'].unique():
        df_group = df[df['label'] == label]
        custom_ids = df_group['id'].apply(lambda x: [x]).tolist()
        fig.add_trace(go.Scatter(x=df_group['x'], y=df_group['y'], mode='markers', name=label, customdata=custom_ids, marker=dict(size=6, opacity=0.8), hovertemplate=('Label: %{data.name}<br>ID: %{customdata[0]}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>')))
    fig.update_layout(showlegend=False, xaxis_title='Dimension 1', yaxis_title='Dimension 2', margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='#f8f9fa', paper_bgcolor='#f8f9fa')
    drag_mode = 'select' if interaction_mode == 'brush' else 'zoom'
    fig.update_layout(dragmode=drag_mode)
    return fig


def plot_dendrogram_from_model(model, colors=None, scores=None, is_selecteds=None):
    """Convenience: render a dendrogram model (icoord/dcoord) by building segments.

    This keeps rendering code separate from coordinate generation. A D3 server
    could call the coordinate-generating functions in `dendro` and then send
    the `model` JSON to the client for rendering.
    """
    from ..dendro.linkage import get_segments_from_model
    segments = get_segments_from_model(model)
    return plot_dendrogram_plotly(segments, colors=colors, scores=scores, is_selecteds=is_selecteds)
