
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scatter plot for 2D embeddings
def plot_2d(X, labels=None, title="2D Scatter Plot", label_names=None):
    fig = px.scatter(
        x=X[:, 0],
        y=X[:, 1],
        color=labels if labels is not None else None,
        title=title,
        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    if label_names and labels is not None:
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.for_each_trace(lambda t: t.update(name=label_names[int(t.name)]))
    else:
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    fig.show()


