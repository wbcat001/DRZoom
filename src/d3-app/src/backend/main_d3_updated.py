"""
d3-app backend API - Main API routes
Provides endpoints for the D3.js-based HDBSCAN cluster explorer
"""

"""
command to run the server:
uvicorn main_d3:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import random

# Support both package and script execution
try:  # pragma: no cover
    from .services.d3_data_manager import D3DataManager  # type: ignore
except Exception:  # pragma: no cover
    from services.d3_data_manager import D3DataManager

# Initialize FastAPI app
app = FastAPI(title="D3 Cluster Explorer API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data manager
data_manager = D3DataManager()

# ============================================================================
# Pydantic Models
# ============================================================================

class PointToClusterRequest(BaseModel):
    """Request model for mapping points to clusters"""
    point_ids: List[int]
    containment_ratio_threshold: float = 0.1


class PointToClusterResponse(BaseModel):
    """Response model for point to cluster mapping"""
    cluster_ids: List[int]
    stats: Dict[str, Any]


class HeatmapRequest(BaseModel):
    """Request model for heatmap data"""
    metric: str = "kl_divergence"
    top_n: int = 200
    cluster_ids: Optional[List[int]] = None


class DendrogramFilterRequest(BaseModel):
    """Request model for dendrogram filtering"""
    strahler_range: Optional[List[float]] = None
    stability_range: Optional[List[float]] = None


class ClusterDetailResponse(BaseModel):
    """Response model for cluster details"""
    cluster_id: int
    label: str
    words: List[str]
    stability: float
    strahler: int
    size: int
    related_clusters: List[Dict[str, Any]]


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.get("/api/datasets")
async def get_datasets():
    """Get available datasets"""
    try:
        datasets = data_manager.get_available_datasets()
        return {
            "success": True,
            "datasets": datasets,
            "timestamp": data_manager.get_timestamp()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/initial_data")
async def get_initial_data(
    dataset: str = Query(...),
    dr_method: str = Query("umap"),
    dr_params: Optional[str] = Query(None),
    color_mode: str = Query("cluster")
):
    """
    Get initial data for visualization
    
    Parameters:
    - dataset: Dataset name
    - dr_method: Dimensionality reduction method (umap, tsne, pca)
    - dr_params: JSON string of DR parameters
    - color_mode: Color assignment mode ('cluster' or 'distance')
    
    Returns:
    - points: Array of point objects with coordinates and cluster info
    - zMatrix: Linkage matrix for dendrogram
    - clusterMeta: Cluster metadata (stability, strahler, size)
    - clusterNames: Cluster label names
    - clusterWords: Representative words for each cluster
    """
    try:
        # Parse DR parameters
        dr_params_dict = {}
        if dr_params:
            dr_params_dict = json.loads(dr_params)
        
        # Get data from manager
        data = data_manager.get_initial_data(dataset, dr_method, dr_params_dict, color_mode)
        
        return {
            "success": True,
            "data": data,
            "timestamp": data_manager.get_timestamp()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/initial_data_no_noise")
async def get_initial_data_no_noise(
    dataset: str = Query(...),
    dr_method: str = Query("umap"),
    dr_params: Optional[str] = Query(None),
    color_mode: str = Query("cluster")
):
    """
    Get initial data for visualization with noise points (cluster=-1) filtered out
    
    Same parameters and response as /api/initial_data but excludes noise points
    for better rendering performance with large datasets.
    
    Parameters:
    - dataset: Dataset name
    - dr_method: Dimensionality reduction method (umap, tsne, pca)
    - dr_params: JSON string of DR parameters
    - color_mode: Color assignment mode ('cluster' or 'distance')
    
    Returns:
    - points: Array of point objects WITHOUT noise points (c != -1)
    - zMatrix: Linkage matrix for dendrogram
    - clusterMeta: Cluster metadata (only for non-noise clusters)
    - clusterNames: Cluster label names (only for non-noise clusters)
    - clusterWords: Representative words for each cluster (only for non-noise clusters)
    """
    try:
        # Parse DR parameters
        dr_params_dict = {}
        if dr_params:
            dr_params_dict = json.loads(dr_params)
        
        # Get data from manager (with noise filtering)
        data = data_manager.get_initial_data_no_noise(dataset, dr_method, dr_params_dict, color_mode)
        
        return {
            "success": True,
            "data": data,
            "timestamp": data_manager.get_timestamp()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/point_to_cluster")
async def map_points_to_clusters(request: PointToClusterRequest):
    """
    Map selected points to clusters with containment filtering
    
    Takes a list of point IDs and returns:
    - cluster_ids: List of clusters containing these points (filtered by threshold)
    - stats: Statistics about the mapping
    """
    try:
        result = data_manager.get_clusters_from_point_selection(
            request.point_ids,
            request.containment_ratio_threshold
        )
        
        return {
            "success": True,
            "cluster_ids": result["cluster_ids"],
            "stats": result["stats"],
            "timestamp": data_manager.get_timestamp()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/heatmap")
async def get_heatmap(
    metric: str = Query("kl_divergence"),
    top_n: int = Query(200),
    cluster_ids: Optional[str] = Query(None)
):
    """
    Get similarity/heatmap data for clusters
    
    Parameters:
    - metric: Similarity metric (kl_divergence, bhattacharyya_coefficient, mahalanobis_distance)
    - top_n: Maximum number of clusters to include
    - cluster_ids: JSON array of specific cluster IDs (optional)
    
    Returns:
    - matrix: 2D array of similarity values
    - clusterOrder: Order of clusters in the matrix
    """
    try:
        cluster_ids_list = None
        if cluster_ids:
            cluster_ids_list = json.loads(cluster_ids)
        
        heatmap_data = data_manager.get_heatmap_data(metric, top_n, cluster_ids_list)
        
        return {
            "success": True,
            "data": heatmap_data,
            "timestamp": data_manager.get_timestamp()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cluster/{cluster_id}")
async def get_cluster_detail(cluster_id: int):
    """
    Get detailed information about a specific cluster
    
    Returns:
    - cluster: Cluster information (id, label, words, metadata)
    - related_clusters: Similar clusters with similarity scores
    """
    try:
        cluster_data = data_manager.get_cluster_detail(cluster_id)
        
        return {
            "success": True,
            "cluster": cluster_data,
            "timestamp": data_manager.get_timestamp()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/point/{point_id}")
async def get_point_detail(point_id: int):
    """
    Get detailed information about a specific point
    
    Returns:
    - point: Point information (id, label, coordinates, cluster)
    - nearbyPoints: Nearby points in the DR space
    """
    try:
        point_data = data_manager.get_point_detail(point_id)
        
        return {
            "success": True,
            "point": point_data,
            "timestamp": data_manager.get_timestamp()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dendrogram/filter")
async def filter_dendrogram(request: DendrogramFilterRequest):
    """
    Filter dendrogram by Strahler and Stability values
    
    Parameters:
    - strahler_range: [min, max] range for Strahler numbers
    - stability_range: [min, max] range for Stability scores
    
    Returns:
    - filteredLinkageMatrix: Filtered linkage matrix
    - visibleClusterIds: IDs of clusters that pass the filter
    """
    try:
        filtered_data = data_manager.filter_dendrogram(
            request.strahler_range,
            request.stability_range
        )
        
        return {
            "success": True,
            "data": filtered_data,
            "timestamp": data_manager.get_timestamp()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clusters/{cluster_id}/nearby")
async def get_nearby_clusters(cluster_id: int):
    """
    Get nearby clusters for a given cluster ID.
    
    Currently returns a dummy list of nearby cluster IDs.
    This is a placeholder for future implementation with actual similarity/stability logic.
    
    Parameters:
    - cluster_id: The target cluster ID
    
    Returns:
    - nearbyClusterIds: List of nearby cluster IDs
    """
    try:
        # TODO: Implement actual logic to determine nearby clusters
        # Candidates: stability-based, size-ratio-based, similarity-based, etc.
        
        # Placeholder: return dummy nearby clusters (for testing)
        # Generate 10-15 dummy cluster IDs based on the target cluster ID
        
        # Seed based on cluster_id for consistency
        random.seed(cluster_id)
        
        # Generate nearby clusters with IDs close to the target
        num_nearby = random.randint(10, 15)
        nearby_ids = []
        
        for _ in range(num_nearby):
            # Add some variation (within ±500 of the target)
            offset = random.randint(-500, 500)
            nearby_id = cluster_id + offset
            if nearby_id > 0 and nearby_id != cluster_id:
                nearby_ids.append(nearby_id)
        
        # Ensure we have unique IDs and return sorted
        nearby_ids = sorted(list(set(nearby_ids)))[:15]  # Limit to 15
        
        return {
            "success": True,
            "nearbyClusterIds": nearby_ids,
            "timestamp": data_manager.get_timestamp()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Error handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "success": False,
        "error": {
            "code": exc.status_code,
            "message": exc.detail
        },
        "timestamp": data_manager.get_timestamp()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
