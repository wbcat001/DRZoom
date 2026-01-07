"""
CPU UMAP Zoom Server - CPU版次元削減専用APIサーバー
GPU不要、scikit-learn の UMAP を使用してメンタルマップ保持

使用方法:
  uvicorn cpu_umap_server:app --host 0.0.0.0 --port 8002
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import base64
import io
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cpu_umap_server")

# CPU UMAP (scikit-learn)
try:
    import umap
    HAS_UMAP = True
    logger.info("UMAP support available (umap-learn)")
except ImportError as e:
    logger.warning(f"UMAP not available: {e}")
    HAS_UMAP = False

app = FastAPI(
    title="CPU UMAP Zoom Server",
    description="CPU版次元削減エンドポイント（メンタルマップ保持）",
    version="1.0.0"
)

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

random_state = 42


# ============================================================================
# Request/Response Models
# ============================================================================

class ZoomRedrawRequest(BaseModel):
    """ズーム再描画リクエスト"""
    # Base64エンコードされた高次元ベクトル
    vectors_b64: str
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

def _b64_to_float32(data_b64: str) -> np.ndarray:
    """Decode Base64 -> raw float32 array"""
    decoded = base64.b64decode(data_b64)
    return np.frombuffer(decoded, dtype=np.float32)


def _numpy_to_b64(array: np.ndarray) -> str:
    """NumPy配列をBase64エンコード (raw float32)"""
    # Convert to float32 and flatten to raw bytes
    arr_f32 = array.astype(np.float32)
    return base64.b64encode(arr_f32.tobytes()).decode('utf-8')


def _log_range(name: str, arr: np.ndarray) -> None:
    """Log min/max for debugging input ranges"""
    try:
        print(f"{name} range: [{float(np.min(arr)):.4f}, {float(np.max(arr)):.4f}]")
    except Exception as exc:
        logger.debug("Could not compute range for %s: %s", name, exc)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "umap_available": HAS_UMAP,
        "backend": "CPU"
    }


# ============================================================================
# Zoom Redraw Endpoint
# ============================================================================

@app.post("/api/zoom/redraw", response_model=ZoomRedrawResponse)
async def zoom_redraw(request: ZoomRedrawRequest):
    """
    CPU UMAPを使用して2D座標を再計算
    
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
    
    if not HAS_UMAP:
        return ZoomRedrawResponse(
            status="error",
            message="UMAP not available. Install with: pip install umap-learn"
        )
    
    try:
        # ============================================================
        # 1. データのデコード
        # ============================================================
        # Decode initial embedding first to infer N
        if not request.initial_embedding_b64:
            raise ValueError("initial_embedding_b64 is required to infer point count")
        init_flat = _b64_to_float32(request.initial_embedding_b64)
        if init_flat.size % 2 != 0:
            raise ValueError("initial_embedding_b64 length must be divisible by 2")
        n_points = init_flat.size // 2
        initial_embedding = init_flat.reshape(n_points, 2)
        logger.debug("Loaded initial embedding: %s", initial_embedding.shape)
        _log_range("initial_embedding_x", initial_embedding[:, 0])
        _log_range("initial_embedding_y", initial_embedding[:, 1])

        # Decode vectors as flat float32 and reshape using inferred N
        vectors_flat = _b64_to_float32(request.vectors_b64)
        if vectors_flat.size % n_points != 0:
            raise ValueError("vectors_b64 length is not divisible by point count")
        n_features = vectors_flat.size // n_points
        vectors = vectors_flat.reshape(n_points, n_features)
        logger.debug("Loaded vectors: %s", vectors.shape)
        _log_range("vectors", vectors)
        
        # ============================================================
        # 2. UMAP モデル初期化
        # ============================================================
        
        # n_neighbors を調整（ポイント数より小さくする）
        n_neighbors_adjusted = min(request.n_neighbors, vectors.shape[0] - 1)
        if n_neighbors_adjusted < request.n_neighbors:
            logger.debug(
                "Adjusted n_neighbors: %s → %s", request.n_neighbors, n_neighbors_adjusted
            )
        
        # scikit-learn UMAP は numpy 配列を init に直接渡せる
        umap_model = umap.UMAP(
            n_components=request.n_components,
            n_neighbors=n_neighbors_adjusted,
            min_dist=request.min_dist,
            metric=request.metric,
            random_state=random_state,
            init=initial_embedding if initial_embedding is not None else 'spectral',  # 🌟 メンタルマップ保持
            n_epochs=request.n_epochs,
            verbose=False
        )
        logger.debug("UMAP model created")
        
        # ============================================================
        # 3. CPU で UMAP 実行
        # ============================================================
        embedding = umap_model.fit_transform(vectors)
        logger.debug("UMAP computation complete")
        
        # ============================================================
        # 4. 結果確認
        # ============================================================
        _log_range("result_x", embedding[:, 0])
        _log_range("result_y", embedding[:, 1])
        logger.debug("Result shape: %s", embedding.shape)
        
        # ============================================================
        # 5. Base64 でエンコード
        # ============================================================
        embedding_b64 = _numpy_to_b64(embedding)
        logger.debug("Encoded size: %s characters", len(embedding_b64))
        
        return ZoomRedrawResponse(
            status="success",
            coordinates=embedding_b64,
            shape=list(embedding.shape)
        )
    
    except ValueError as e:
        logger.warning("ValueError: %s", e)
        return ZoomRedrawResponse(
            status="error",
            message=str(e)
        )
    except Exception as e:
        logger.exception("Unhandled error during zoom_redraw")
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
        "name": "CPU UMAP Zoom Server",
        "version": "1.0.0",
        "backend": "CPU (scikit-learn)",
        "umap_available": HAS_UMAP,
        "endpoints": [
            {"method": "GET", "path": "/health", "description": "ヘルスチェック"},
            {"method": "GET", "path": "/api/info", "description": "サーバー情報"},
            {"method": "POST", "path": "/api/zoom/redraw", "description": "CPU UMAPズーム"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    # UMAP 確認
    if HAS_UMAP:
        logger.info("=" * 60)
        logger.info("CPU UMAP Zoom Server")
        logger.info("=" * 60)
        logger.info("UMAP support enabled (umap-learn)")
        try:
            import umap
            logger.info("UMAP version: %s", umap.__version__)
        except Exception as e:
            logger.warning("Could not get UMAP info: %s", e)
    else:
        logger.info("=" * 60)
        logger.info("CPU UMAP Zoom Server - UMAP Not Available")
        logger.info("=" * 60)
        logger.info("UMAP not installed")
        logger.info("To enable UMAP: pip install umap-learn")
    
    logger.info("Starting server on http://0.0.0.0:8002")
    logger.info("API docs: http://localhost:8002/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
