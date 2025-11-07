import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import cudf
import cupy as cp
from typing import List, Optional
from cuml.manifold import UMAP 
from sklearn.datasets import make_blobs # ダミーデータ生成用
from gensim.models import KeyedVectors
import time
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- グローバルデータ設定 ---
N_TOTAL = 500000  # 全データ点数 
N_FEATURES = 300   # 特徴量数
GLOBAL_DATA_DF: Optional[cudf.DataFrame] = None
GLOBAL_DATA_X: Optional[cp.ndarray] = None # UMAPに直接渡すためのCuPy配列


def load_w2v(n_samples=5000, is_random=True):
    print(os.getcwd())
    print(BASE_DIR)
    file_path = os.path.join(BASE_DIR, "data", "GoogleNews-vectors-negative300.bin")
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)

    words = model.index_to_key
    print(f"Number of words in the model: {len(words)}")

    if is_random:
        np.random.seed(42)
        ramdom_indices = np.random.choice(len(words), size=n_samples, replace=False)

    else:
        ramdom_indices = np.arange(n_samples)
    
    selected_words = [words[i] for i in ramdom_indices]
    selected_vectors = model.vectors[ramdom_indices]

    return selected_vectors, selected_words

app = FastAPI(title="RAPIDS GPU API (Extended)")

# --- サーバー起動時の処理: グローバルデータをGPUにロード ---
@app.on_event("startup")
async def startup_event():
    global GLOBAL_DATA_DF, GLOBAL_DATA_X
    print("--- サーバー起動: グローバルデータセットをGPUにロード中 ---")
    
    # 1. CPUでダミーデータを生成 (ここでは make_blobs を使用)
    # 実際のアプリケーションでは、ここでParquetやCSVなどをロードします
    data_cpu, _ = load_w2v(n_samples=N_TOTAL, is_random=True)
    
    # 2. CuPy配列に変換 (UMAPが直接使える形式)
    GLOBAL_DATA_X = cp.asarray(data_cpu, dtype=cp.float32)
    
    # 3. cuDF DataFrameに変換 (インデックスフィルタリングに最適)
    # デフォルトのインデックス (0からN_TOTAL-1) が自動的に設定されます
    GLOBAL_DATA_DF = cudf.DataFrame(GLOBAL_DATA_X)
    
    # GPU処理の完了を待つ
    cp.cuda.runtime.deviceSynchronize()
    
    print(f"グローバルデータ (N={N_TOTAL}, F={N_FEATURES}) をGPUメモリにロード完了.")
    print(f"データセットのGPUメモリ上のサイズ: {GLOBAL_DATA_DF.shape}")

# --- Pydantic モデル定義 ---

# 統計情報計算用の入力モデル (変更なし)
class DataInput(BaseModel):
    # Python 3.9 以降の 'list[float]' 記法を使用
    values: list[float]

# 汎用UMAP入力モデル (クライアントから全データを受け取る場合)
class UMAPInput(BaseModel):
    # 2次元配列: [[x1, x2, ...], [x1, x2, ...], ...]
    data: List[List[float]]
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    
# サブセットUMAP入力モデル (サーバー上のデータをIDでフィルタする場合)
class SubsetInput(BaseModel):
    # サーバー上のグローバルデータのインデックスのリスト
    id_list: List[int]
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"


# --- API エンドポイント (変更なし) ---

@app.post("/gpu_stats")
def gpu_stats(data: DataInput):
    """クライアントから受け取ったデータに対してGPUで統計量を計算します。"""
    
    # GPU上にデータをロード
    gpu_series = cudf.Series(data.values)
    
    # GPUで計算（平均・分散）
    mean_gpu = gpu_series.mean()
    var_gpu = gpu_series.var()

    # 結果を返す
    return {
        "count": int(len(gpu_series)),
        "mean": float(mean_gpu),
        "variance": float(var_gpu)
    }

@app.post("/umap_full")
def compute_umap_full(params: UMAPInput):
    """クライアントから渡された全データに対してUMAPを実行します。"""
    start_time = time.time()
    
    # 入力をGPU上の配列に変換 (CPU -> GPU)
    data_gpu = cp.asarray(params.data, dtype=cp.float32)
    
    umap_model = UMAP(
        n_components=params.n_components,
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        metric=params.metric
    )

    # GPUでUMAPを計算
    embedding = umap_model.fit_transform(data_gpu)
    
    # GPU 処理の完了を待つ (同期)
    cp.cuda.runtime.deviceSynchronize()
    
    # CPU に戻してリスト化 (GPU -> CPU)
    embedding_cpu = cp.asnumpy(embedding).tolist()
    
    end_time = time.time()
    execution_time = end_time - start_time

    return {
        "data_shape": list(data_gpu.shape),
        "embedding": embedding_cpu,
        "execution_time_sec": execution_time
    }

# --- API エンドポイント (新規作成: サブセットUMAP) ---

@app.post("/umap_subset")
def compute_umap_subset(params: SubsetInput):
    """
    サーバー上のグローバルデータをクライアントのIDリストでフィルタし、
    そのサブセットに対してUMAPを実行します。
    """
    global GLOBAL_DATA_DF
    
    if GLOBAL_DATA_DF is None:
        return {"error": "Global dataset not initialized."}, 500

    start_time = time.time()

    # 1. IDリストをcuDF Seriesに変換し、GPUに移動
    id_series = cudf.Series(params.id_list)
    
    # 2. cuDFのインデックス機能を使ってグローバルデータからサブセットを抽出
    # .loc[id_series] で、IDリストに一致するインデックスの行を抽出
    try:
        subset_df = GLOBAL_DATA_DF.loc[id_series]
    except Exception as e:
        return {"error": f"Filtering failed (Invalid IDs or Indexing error): {str(e)}"}, 400
        
    # 3. UMAPが要求する CuPy 配列に変換
    data_gpu = subset_df.to_cupy()
    
    # 4. UMAPモデルを初期化して計算
    umap_model = UMAP(
        n_components=params.n_components,
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        metric=params.metric
    )

    embedding = umap_model.fit_transform(data_gpu)
    
    # GPU 処理の完了を待つ (同期)
    cp.cuda.runtime.deviceSynchronize()
    
    # 5. CPU に戻してリスト化
    embedding_cpu = cp.asnumpy(embedding).tolist()
    
    end_time = time.time()
    execution_time = end_time - start_time

    return {
        "requested_ids_count": len(params.id_list),
        "subset_shape": list(data_gpu.shape),
        "embedding": embedding_cpu,
        "execution_time_sec": execution_time # 実行時間を追加
    }

@app.get("/")
def root():
    """APIの稼働状況を確認します。"""
    return {"message": "RAPIDS GPU API is running"}

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    # 実際の環境に合わせて uvicorn の実行を設定してください
    # print("--- 実行するには 'uvicorn rapids_gpu_api_extended:app --reload' を使用してください ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)