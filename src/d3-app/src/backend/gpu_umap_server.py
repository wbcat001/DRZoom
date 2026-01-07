"""
GPU UMAP Zoom Server - GPU加速次元削減専用APIサーバー
GPU カーネル上で実行する必要があるため、main_d3.py から分離

使用方法:
  uvicorn gpu_umap_server:app --host 0.0.0.0 --port 8001
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import base64
import io
import json

# GPU ライブラリ
try:
    import cupy as cp
    from cuml.manifold import UMAP
    HAS_GPU = True
from fastapi.middleware.cors import CORSMiddleware
    print("✓ GPU support available (CuPy, cuML)")
except ImportError as e:
    print(f"⚠ GPU support not available: {e}")
    HAS_GPU = False

app = FastAPI(
    title="GPU UMAP Zoom Server",
    description="GPU加速次元削減エンドポイント",
    version="1.0.0"
)

random_state = 42


# ============================================================================
# Request/Response Models
# ============================================================================

class ZoomRedrawRequest(BaseModel):
    """ズーム再描画リクエスト"""
    # Base64エンコードされた高次元ベクトル
    vectors_b64: str

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
    # Base64エンコードされた初期2D座標（メンタルマップ保持用）
    initial_embedding_b64: Optional[str] = None
    
    # UMAP パラメータ
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    n_epochs: int = 200


class ZoomRedrawResponse(BaseModel):
    """ズーム再描画レスポンス"""
    status: str  # "success" or "error"
    coordinates: Optional[str] = None  # Base64-encoded (N, 2) array
    shape: Optional[List[int]] = None
    message: Optional[str] = None


# ============================================================================
# Helper Functions - Base64 Encoding/Decoding
# ============================================================================

def _b64_to_numpy(data_b64: str) -> np.ndarray:
    """Base64文字列からNumPy配列にデコード"""
    decoded = base64.b64decode(data_b64)
    return np.load(io.BytesIO(decoded))


def _numpy_to_b64(array: np.ndarray) -> str:
    """NumPy配列をBase64文字列にエンコード"""
    buff = io.BytesIO()
    np.save(buff, array, allow_pickle=False)
    return base64.b64encode(buff.getvalue()).decode('utf-8')


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "gpu_available": HAS_GPU
    }


# ============================================================================
# Zoom Redraw Endpoint
# ============================================================================

@app.post("/api/zoom/redraw", response_model=ZoomRedrawResponse)
async def zoom_redraw(request: ZoomRedrawRequest):
    """
    GPU UMAPを使用して2D座標を再計算
    
    Args:
        request: ZoomRedrawRequest
            - vectors_b64: 高次元ベクトル (Base64)
            - initial_embedding_b64: 初期2D座標 (Base64, オプション)
            - n_neighbors, min_dist, n_epochs: UMAP パラメータ
    
    Returns:
        ZoomRedrawResponse
            - status: "success" or "error"
            - coordinates: 新しい2D座標 (Base64)
            - shape: [N, 2]
    """
    
    if not HAS_GPU:
        return ZoomRedrawResponse(
            status="error",
            message="GPU UMAP not available. Install cupy and cuml."
        )
    
    try:
        # ============================================================
        # 1. データのデコード
        # ============================================================
        print("[1/6] Decoding input data...")
        vectors_cpu = _b64_to_numpy(request.vectors_b64)
        
        initial_embedding_cpu = None
        if request.initial_embedding_b64:
            initial_embedding_cpu = _b64_to_numpy(request.initial_embedding_b64)
            print(f"  ✓ Loaded initial embedding: {initial_embedding_cpu.shape}")
        
        print(f"  ✓ Loaded vectors: {vectors_cpu.shape}")
        
        # 次元チェック
        if initial_embedding_cpu is not None:
            if vectors_cpu.shape[0] != initial_embedding_cpu.shape[0]:
                raise ValueError(
                    f"Vector count ({vectors_cpu.shape[0]}) != "
                    f"initial embedding count ({initial_embedding_cpu.shape[0]})"
                )
        
        # ============================================================
        # 2. GPU に転送
        # ============================================================
        print("[2/6] Transferring to GPU...")
        vectors_gpu = cp.asarray(vectors_cpu, dtype=cp.float32)
        
        init_gpu = None
        if initial_embedding_cpu is not None:
            init_gpu = cp.asarray(initial_embedding_cpu, dtype=cp.float32)
            print(f"  ✓ GPU transfer complete: vectors {vectors_gpu.shape}, init {init_gpu.shape}")
        else:
            print(f"  ✓ GPU transfer complete: vectors {vectors_gpu.shape}")
        
        # ============================================================
        # 3. UMAP モデル初期化
        # ============================================================
        print("[3/6] Initializing UMAP model...")
        
        # n_neighbors を調整（ポイント数より小さくする）
        n_neighbors_adjusted = min(request.n_neighbors, vectors_cpu.shape[0] - 1)
        if n_neighbors_adjusted < request.n_neighbors:
            print(f"  ⚠ Adjusted n_neighbors: {request.n_neighbors} → {n_neighbors_adjusted}")
        
        # CuPy 配列またはNone をinitに渡す
        umap_model = UMAP(
            n_components=request.n_components,
            n_neighbors=n_neighbors_adjusted,
            min_dist=request.min_dist,
            metric=request.metric,
            random_state=random_state,
            init=init_gpu,  # 🌟 メンタルマップ保持のため初期位置を使用（CuPy配列またはNone）
            n_epochs=request.n_epochs,
            verbose=1
        )
        print("  ✓ UMAP model created")
        
        # ============================================================
        # 4. GPU で UMAP 実行
        # ============================================================
        print("[4/6] Running UMAP on GPU...")
        embedding_gpu = umap_model.fit_transform(vectors_gpu)
        cp.cuda.runtime.deviceSynchronize()
        print("  ✓ UMAP computation complete")
        
        # ============================================================
        # 5. CPU に転送
        # ============================================================
        print("[5/6] Transferring results to CPU...")
        embedding_cpu = cp.asnumpy(embedding_gpu)
        print(f"  ✓ Results ready: {embedding_cpu.shape}")
        
        # ============================================================
        # 6. Base64 でエンコード
        # ============================================================
        print("[6/6] Encoding results...")
        embedding_b64 = _numpy_to_b64(embedding_cpu)
        print(f"  ✓ Encoded size: {len(embedding_b64)} characters")
        
        return ZoomRedrawResponse(
            status="success",
            coordinates=embedding_b64,
            shape=list(embedding_cpu.shape)
        )
    
    except ValueError as e:
        print(f"❌ ValueError: {e}")
        return ZoomRedrawResponse(
            status="error",
            message=str(e)
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return ZoomRedrawResponse(
            status="error",
            message=f"Internal error: {str(e)}"
        )


# ============================================================================
# Info Endpoint
# ============================================================================

@app.get("/api/info")
async def get_info():
    """サーバー情報を取得"""
    return {
        "name": "GPU UMAP Zoom Server",
        "version": "1.0.0",
        "gpu_available": HAS_GPU,
        "endpoints": [
            {"method": "GET", "path": "/health", "description": "ヘルスチェック"},
            {"method": "GET", "path": "/api/info", "description": "サーバー情報"},
            {"method": "POST", "path": "/api/zoom/redraw", "description": "GPU UMAPズーム"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    # GPU 確認
    if HAS_GPU:
        print("=" * 60)
        print("GPU UMAP Zoom Server - Production Mode")
        print("=" * 60)
        print("✓ GPU support enabled")
        try:
            print(f"✓ CuPy version: {cp.__version__}")
            print(f"✓ CUDA version: {cp.cuda.runtime.getVersion()}")
        except Exception as e:
            print(f"⚠ Could not get GPU info: {e}")
    else:
        print("=" * 60)
        print("GPU UMAP Zoom Server - Development Mode (CPU)")
        print("=" * 60)
        print("⚠ GPU support disabled")
        print("  To enable GPU:")
        print("  conda install -c rapids -c conda-forge cuml cupy cudatoolkit=11.2")
    
    print()
    print("Starting server on http://0.0.0.0:8001")
    print("API docs: http://localhost:8001/docs")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
