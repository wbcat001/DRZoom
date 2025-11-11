import numpy as np
import cupy as cp
from cuml.manifold import UMAP
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
random_state = 42
dir_name = "20251112_044404"

def umap_from_npz(input_path: str, output_path: str, n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean"):
    """
    npzファイルからベクトルを読み込み、UMAPをGPUで計算して保存する
    """
    # --- 1. npz読み込み ---
    data = np.load(input_path, allow_pickle=True)
    vectors = data['X']  # shape=(N, 300)
    words = data['labels']      # shape=(N,)

    print(f"Loaded {vectors.shape[0]} vectors with {vectors.shape[1]} dimensions.")

    # --- 2. GPU に転送 ---
    data_gpu = cp.asarray(vectors, dtype=cp.float32)

    # --- 3. UMAP モデル初期化 ---
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )

    # --- 4. GPUでUMAP実行 ---
    embedding_gpu = umap_model.fit_transform(data_gpu)
    cp.cuda.runtime.deviceSynchronize()

    # --- 5. CPUに戻す ---
    embedding_cpu = cp.asnumpy(embedding_gpu)

    # --- 6. npzで保存 ---
    np.savez(output_path, embedding=embedding_cpu, words=words)
    print(f"✅ UMAP embedding saved to {output_path}, shape={embedding_cpu.shape}")

# --- 実行例 ---
if __name__ == "__main__":
    input_npz = os.path.join(BASE_DIR, "result", dir_name, "data.npz")
    output_npz = os.path.join(BASE_DIR, "result", dir_name, "embedding.npz")

    umap_from_npz(input_npz, output_npz)
