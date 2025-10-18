from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
from data_manager import DataManager, init_data_manager, get_data_manager
from analysis_engine import AnalysisEngine
from fastapi.middleware.cors import CORSMiddleware # Reactとの連携に必要
import numpy as np

# --- 1. 初期化 ---
# アプリケーション起動時にデータマネージャーを初期化（ここではダミーデータ）
init_data_manager(num_points=5000, dim=768)

app = FastAPI(title="Hierarchical Visualization API")

# CORS設定：Reactのフロントエンドからアクセスできるようにする
origins = [
    "http://localhost:3000", # React開発サーバー
    # ... 他のフロントエンドのURL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AnalysisEngineの依存性注入のためのヘルパー関数
def get_analysis_engine(dm: DataManager = Depends(get_data_manager)) -> AnalysisEngine:
    return AnalysisEngine(dm)

# --- 2. リクエストボディの定義 ---
class ZoomRequest(BaseModel):
    zoom_level: int = 1
    prev_level_id: str = None

# --- 3. エンドポイントの定義 ---

@app.get("/api/levels")
def get_zoom_levels(dm: DataManager = Depends(get_data_manager)) -> List[int]:
    """意味のあるズームレベル（1, 2, 3...）のリストを返す"""
    # フロントエンド用にインデックスベースのレベルを返す
    return list(range(1, len(dm.level_lambdas) + 1))

@app.post("/api/zoom")
def perform_semantic_zoom(
    request: ZoomRequest, 
    engine: AnalysisEngine = Depends(get_analysis_engine)
) -> Dict[str, Any]:
    """階層的ズームリクエストに基づいて次元削減とアライメントを実行する"""
    
    result = engine.generate_umap_embedding(
        zoom_level=request.zoom_level,
        prev_embedding_id=request.prev_level_id
    )
    
    return result

@app.get("/api/overview")
def get_initial_overview(engine: AnalysisEngine = Depends(get_analysis_engine)) -> Dict[str, Any]:
    """初期のOverviewビューを返す（Zoom Level 1）"""
    return engine.generate_umap_embedding(zoom_level=1, prev_embedding_id=None)


class ContourRequest(BaseModel):
    level_id: str
    cluster_id: int

@app.post("/api/contours")
def get_contour_data(
    request: ContourRequest, 
    engine: AnalysisEngine = Depends(get_analysis_engine)
) -> Dict[str, Any]:
    """階層構造に基づくクラスタの境界（等高線）データを返す"""
    return engine.generate_contour_data(
        level_id=request.level_id,
        cluster_id=request.cluster_id
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)