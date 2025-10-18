
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
import pandas as pd
# Scatter plot for 2D embeddings
def plot_2d(X, labels=None, title="2D Scatter Plot", label_names=None):
    fig = px.scatter(
        x=X[:, 0],
        y=X[:, 1],
        color=labels.astype(str) if labels is not None else None,
        title=title,
        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    if label_names and labels is not None:
        fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))
        fig.for_each_trace(lambda t: t.update(name=label_names[int(t.name)]))
    else:
        fig.update_traces(marker=dict(size=3, color="black"), selector=dict(mode='markers'))

    fig.update_layout(width=800, height=800)
    fig.show()

def plot_umap(X, labels=None):
    reducer = umap.UMAP(n_components=2, random_state=42)
    proj = reducer.fit_transform(X)
    plot_2d(proj, labels)

# receive array of points, labels, params
"""
df = pd.DataFrame(proj, columns=[f"UMAP_{i+1}" for i in range(n_components)])
df['cluster'] = labels
df['eps'] = eps
dfs.append(df)
"""
def plot_2d_facets(df_all, x_col='x', y_col='y', color_col='label', facet_col='param', title='2d scatter facet'):
    """
    UMAP座標とクラスタラベルをもとに、facet_colで横並べの散布図を描画する関数。
    """
    fig = px.scatter(
        df_all, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        facet_col=facet_col,
        title=title,
    )

    n_facets = len(df_all[facet_col].unique())
    print(f"n_facets: {n_facets}")  
    
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        height=500 ,
        width=500 * n_facets
    )
    fig.show()

    
def plot_hierarchical_scatter(proj, labels, params):
    dfs = []
    for i, p in enumerate(params):
        df = pd.DataFrame(proj, columns=['x', 'y'])
        df['param'] = p
        df['label'] = labels[i].astype(str)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    plot_2d_facets(df_all, x_col='x', y_col='y', color_col='label', facet_col='param', title='Hierarchical Clustering with UMAP Projection')
