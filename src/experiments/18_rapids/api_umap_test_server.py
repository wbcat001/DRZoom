from fastapi import FastAPI
from pydantic import BaseModel
import cudf
import cupy as cp
# 1. 'List' を typing からインポート
from typing import List 
# 2. 'UMAP' を cuml.manifold からインポート (RAPIDSの場合)
from cuml.manifold import UMAP 
import time
app = FastAPI(title="RAPIDS GPU API")

# クライアントから受け取るデータ形式
# 'values: list[float]' は Python 3.9 以降の記法です。
# 互換性のため、または Python 3.9 未満の場合は 'values: List[float]' とします。
# 今回は Python の組み込み型である 'list[float]' を使用します。
class DataInput(BaseModel):
    values: list[float]

@app.post("/gpu_stats")
def gpu_stats(data: DataInput):
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

class UMAPInput(BaseModel):
    # 'List' の代わりに 'list' を使っても良いですが、元のコードに倣い 'List' を使用します
    # ただし、ファイル先頭で 'from typing import List' が必要です。
    data: List[List[float]]  # 2次元配列: [[x1, x2, ...], [x1, x2, ...], ...]
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean",
    random_state: int = 42
@app.post("/umap")
def compute_umap(params: UMAPInput):
    # 処理開始前の時刻を記録
    start_time = time.time()
    
    # 入力をGPU上の配列に変換 (CPU -> GPU)
    data_gpu = cp.asarray(params.data)
    
    # UMAPモデルを初期化
    umap_model = UMAP(
        n_components=params.n_components,
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        metric=params.metric,
        random_state=params.random_state
    )

    # GPUでUMAPを計算 (非同期処理)
    embedding = umap_model.fit_transform(data_gpu)
    
    # 🔥 **重要:** GPU 処理の完了を待つ (同期)
    # これにより、GPU 上での UMAP 処理が完了するまでの正確な時間を計測できます。
    cp.cuda.runtime.deviceSynchronize()
    
    # CPU に戻してリスト化 (GPU -> CPU)
    embedding_cpu = cp.asnumpy(embedding).tolist()
    
    # 処理終了後の時刻を記録
    end_time = time.time()
    
    # 処理時間の計算
    execution_time = end_time - start_time

    return {
        "embedding": embedding_cpu,
        "execution_time_sec": execution_time # 実行時間を追加
    }
@app.get("/")
def root():
    return {"message": "RAPIDS GPU API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)