import cupy as cp
from cuml.preprocessing import normalize
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
import plotly.express as px
import numpy as np
import time

# ===== テスト用データ生成 =====
np.random.seed(42)
cluster_1 = np.random.randn(500, 10) + 2
cluster_2 = np.random.randn(500, 10) - 2
X_cpu = np.vstack([cluster_1, cluster_2])

# GPU上にコピー & L2正規化
start = time.time()
X = normalize(cp.array(X_cpu))
print(f"データをGPUに転送・正規化: {time.time() - start:.2f}秒")

# ===== UMAPで次元削減 =====
start = time.time()
umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embeddings = umap_model.fit_transform(X)
print(f"UMAP計算時間: {time.time() - start:.2f}秒")

# ===== HDBSCANでクラスタリング =====
start = time.time()
hdbscan_model = HDBSCAN(min_samples=10, min_cluster_size=20, cluster_selection_method='leaf')
labels = hdbscan_model.fit_predict(X)
print(f"HDBSCAN計算時間: {time.time() - start:.2f}秒")

# ===== Plotlyで可視化（HTML保存） =====
start = time.time()
x = embeddings[:, 0].get()
y = embeddings[:, 1].get()
labels_nonoise = labels.get()

fig = px.scatter(
    x=x,
    y=y,
    color=labels_nonoise.astype(str),
    title='HDBSCAN + UMAP Test (GPU)',
    opacity=0.7
)
fig.update_traces(marker=dict(size=6))

# HTMLとして保存
html_file = "hdbscan_umap_plot.html"
fig.write_html(html_file)
print(f"Plot saved: {html_file}")
print(f"Plotly描画準備時間: {time.time() - start:.2f}秒")

# ===== クラスタ情報を表示 =====
unique_labels = np.unique(labels_nonoise)
print(f"クラスタ数 (ノイズ含む): {len(unique_labels)}")
for c in unique_labels:
    count = np.sum(labels_nonoise == c)
    print(f"Cluster {c}: {count} points")
