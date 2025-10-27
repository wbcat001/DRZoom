import os
import cupy as cp
from cuml.preprocessing import normalize
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
import plotly.express as px
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "result/hdbscan_umap_plot.html")

n_data = 10000
# mnistデータロード
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X_cpu = mnist["data"][:n_data].astype(float)
X = normalize(cp.asarray(X_cpu))

n_neighbors_list = [10, 15, 30, 50]

for n_neighbors in n_neighbors_list:
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
    embeddings = reducer.fit_transform(X)

    fig = px.scatter(
        x=embeddings[:, 0].get(),
        y=embeddings[:, 1].get(),
        title=f"UMAP Projection (n_neighbors={n_neighbors})",
        labels={"x": "UMAP 1", "y": "UMAP 2"},
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(height=800, width=800)
    html_file = file_path.replace(".html", f"_umap_nn{n_neighbors}.html")
    fig.write_html(html_file)
    print(f"Plot saved: {html_file}")