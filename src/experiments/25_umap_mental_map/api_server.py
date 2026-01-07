# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import cupy as cp
from cuml.manifold import UMAP
import base64
import json
import io

app = FastAPI()
random_state = 42

# --- データ構造定義 ---
class UmapRequest(BaseModel):
    # Base64エンコードされた特徴量ベクトル (CPU NumPy array)
    vectors_b64: str
    # Base64エンコードされた既存の埋め込み座標 (CPU NumPy array)
    # メンタルマップ維持のための初期配置として使用
    initial_embedding_b64: str | None = None
    
    # UMAP パラメータ
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    n_epochs: int = 200 # UMAPの実行ステップ数

# --- ヘルパー関数 ---

def _b64_to_numpy(data_b64: str) -> np.ndarray:
    """Base64文字列からNumPy配列にデコードする"""
    decoded = base64.b64decode(data_b64)
    # NumPyの形式でデータを読み込む
    return np.load(io.BytesIO(decoded))

def _numpy_to_b64(array: np.ndarray) -> str:
    """NumPy配列をBase64文字列にエンコードする"""
    buff = io.BytesIO()
    # allow_pickle=Falseにしてセキュリティを確保
    np.save(buff, array, allow_pickle=False) 
    return base64.b64encode(buff.getvalue()).decode('utf-8')


# --- APIエンドポイント ---

@app.post("/recalculate_umap")
async def recalculate_umap(request: UmapRequest):
    """
    特徴量ベクトルと既存座標を基に、GPUでUMAP座標を再計算する
    """
    try:
        # 1. データのデコード (CPU NumPyへ)
        vectors_cpu = _b64_to_numpy(request.vectors_b64)
        
        initial_embedding_cpu = None
        if request.initial_embedding_b64:
            initial_embedding_cpu = _b64_to_numpy(request.initial_embedding_b64)
            print(f"Loaded initial embedding shape: {initial_embedding_cpu.shape}")
        
        print(f"Loaded vectors shape: {vectors_cpu.shape}")

        if initial_embedding_cpu is not None and vectors_cpu.shape[0] != initial_embedding_cpu.shape[0]:
            raise ValueError("Vector count and initial embedding count must match.")

        # 2. GPU に転送
        data_gpu = cp.asarray(vectors_cpu, dtype=cp.float32)
        
        init_gpu = None
        if initial_embedding_cpu is not None:
            # UMAP initには float32 が推奨される
            init_gpu = cp.asarray(initial_embedding_cpu, dtype=cp.float32) 

        # 3. UMAP モデル初期化
        umap_model = UMAP(
            n_components=request.n_components,
            n_neighbors=request.n_neighbors,
            min_dist=request.min_dist,
            metric=request.metric,
            random_state=random_state,
            init=init_gpu,  # 🌟 既存の埋め込みを初期配置として使用
            n_epochs=request.n_epochs
        )

        # 4. GPUでUMAP実行
        embedding_gpu = umap_model.fit_transform(data_gpu)
        cp.cuda.runtime.deviceSynchronize() # GPU処理が完了するのを待つ

        # 5. CPUに戻す
        embedding_cpu = cp.asnumpy(embedding_gpu)

        # 6. Base64でエンコードして返却
        embedding_b64 = _numpy_to_b64(embedding_cpu)

        return {
            "status": "success",
            "embedding_b64": embedding_b64,
            "shape": list(embedding_cpu.shape)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during UMAP calculation.")

# サーバーの起動方法:
# uvicorn api_server:app --reload --host 0.0.0.0 --port 8001