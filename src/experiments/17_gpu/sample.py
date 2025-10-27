import time
import cupy as cp
from cuml.manifold import UMAP
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
# MNIST データセットを取得
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype('float32') / 255.0  # 正規化
y = mnist.target.astype(int)

# GPU用に転送
X_gpu_full = cp.array(X)

# サンプル数リスト（MNIST全体は70000サンプル）
sizes = [1000, 5000, 10000, 30000, 60000]

gpu_times = []

for n in sizes:
    print(f"\n=== {n} samples ===")
    X_gpu = X_gpu_full[:n]

    # UMAPモデル
    umap = UMAP(n_neighbors=15, n_components=2, random_state=42)

    # 時間計測
    start = time.time()
    embedding = umap.fit_transform(X_gpu)
    elapsed = time.time() - start

    gpu_times.append(elapsed)
    print(f"GPU time: {elapsed:.3f} sec")
    plt.figure(figsize=(8, 6))
    plt.scatter(cp.asnumpy(embedding[:, 0]), cp.asnumpy(embedding[:, 1]), c=cp.asnumpy(y[:n]), cmap='Spectral', s=5)
    plt.colorbar(boundaries=range(11))  
    plt.title(f'UMAP on MNIST ({n} samples)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()