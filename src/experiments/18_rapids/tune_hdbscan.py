import os
import cupy as cp
from cuml.preprocessing import normalize
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
import plotly.express as px
import numpy as np
import time
from sklearn.datasets import fetch_openml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "result/hdbscan_umap_plot.html")

n_data = 10000
# mnistデータロード
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X_cpu = mnist["data"][:n_data].astype(float)
X = normalize(cp.asarray(X_cpu))

hdbscan_params = {
    "min_samples": [5, 10, 20],
    "min_cluster_size": [10, 20, 50]
}
results = []

for min_samples in hdbscan_params["min_samples"]:
    for min_cluster_size in hdbscan_params["min_cluster_size"]:
        reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embeddings = reducer.fit_transform(X)

        hdbscan_model = HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            cluster_selection_method='leaf'
        )
        labels = hdbscan_model.fit_predict(X)

        fig = px.scatter(
            x=embeddings[:, 0].get(),
            y=embeddings[:, 1].get(),
            color=labels.get().astype(str),
            title=f"HDBSCAN + UMAP (min_samples={min_samples}, min_cluster_size={min_cluster_size})",
            labels={"x": "UMAP 1", "y": "UMAP 2"},
        )
        fig.update_traces(marker=dict(size=2))
        fig.update_layout(height=800, width=800)
        html_file = file_path.replace(
            ".html",
            f"_hdbscan_ms{min_samples}_mc{min_cluster_size}.html"
        )
        fig.write_html(html_file)
        print(f"Plot saved: {html_file}")

        results.append({
            "min_samples": min_samples,
            "min_cluster_size": min_cluster_size,
           
        })