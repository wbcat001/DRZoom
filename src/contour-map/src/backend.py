# backend/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import umap
from scipy.spatial import ConvexHull
import json
from sklearn.decomposition import PCA
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MNIST データロード（全体）
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist["data"][:1000].astype(float)
y = mnist["target"][:1000].astype(int)

def compute_hull(coords):
    if len(coords) < 3:
        return list(range(len(coords)))
    hull = ConvexHull(coords)
    return list(hull.vertices)

@app.get("/clusters")
def get_clusters(
    level: int = 0,
    parent_indices: str = None,
    n_clusters: int = 5
):
    """
    level=0: 全体
    level>0: 子階層
    parent_indices: カンマ区切りで親階層の点番号を指定
    """
    print(f"Request: level={level}, parent_indices={parent_indices}, n_clusters={n_clusters}")
    
    if parent_indices:
        try:
            idx = list(map(int, parent_indices.split(",")))
            X_sub = X[idx]
            print(f"Using parent indices: {idx}, X_sub shape: {X_sub.shape}")
        except Exception as e:
            print(f"Error parsing parent_indices: {e}")
            return {"error": "Invalid parent_indices format"}
    else:
        X_sub = X
        idx = list(range(len(X)))
        print(f"Using full dataset, X_sub shape: {X_sub.shape}")

    if len(X_sub) < n_clusters:
        n_clusters = len(X_sub)

    # クラスタリング
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(X_sub)
    print(f"Clustering labels: {np.unique(labels)}")

    # 次元削減（PCA）
    reducer = PCA(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X_sub)
    print(f"2D coordinates shape: {X_2d.shape}")

    # 点のデータ作成（グローバルインデックスを保持）
    points = []
    for i, (x, y) in enumerate(X_2d):
        points.append({
            "id": idx[i],  # グローバルインデックス
            "x": float(x), 
            "y": float(y), 
            "cluster": str(labels[i]),
            "local_id": i  # ローカルインデックス
        })

    # クラスター作成
    clusters = []
    for cl in np.unique(labels):
        cluster_indices = np.where(labels == cl)[0]  # ローカルインデックス
        cluster_coords = X_2d[cluster_indices]
        
        # 凸包計算
        if len(cluster_coords) >= 3:
            hull_vertices = compute_hull(cluster_coords)
            # グローバルインデックスに変換
            hull_global_indices = [idx[cluster_indices[v]] for v in hull_vertices]
        else:
            # 3点未満の場合は全ての点を使用
            hull_global_indices = [idx[i] for i in cluster_indices]
        
        clusters.append({
            "id": f"level_{level}_cluster_{cl}",
            "points": hull_global_indices,
            "cluster_points": [idx[i] for i in cluster_indices],  # クラスター内の全ての点
            "child_level": None
        })
        print(f"Cluster {cl}: {len(cluster_indices)} points, hull: {len(hull_global_indices)} vertices")

    result = {
        "level": level, 
        "points": points, 
        "clusters": clusters,
        "total_points": len(points),
        "total_clusters": len(clusters)
    }
    print(f"Returning: {len(points)} points, {len(clusters)} clusters")
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)