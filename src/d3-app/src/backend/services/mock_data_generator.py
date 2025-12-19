"""
Mock Data Generator for D3 App Development

============================================================
MOCK DATA GENERATOR - REMOVE WHEN REAL DATA IS AVAILABLE
============================================================
This module generates synthetic data to allow frontend development
without requiring actual HDBSCAN results.

To use real data instead:
1. Set use_mock_data = False in D3DataManager._load_configuration()
2. Prepare data files in the path specified in datasets_config
3. Ensure files follow the required format
============================================================
"""

from typing import Dict, Any, List
import numpy as np


def generate_mock_data(
    dataset: str,
    dr_method: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate mock data for development when real data files don't exist
    
    Args:
        dataset: Dataset identifier
        dr_method: Dimensionality reduction method
        config: Dataset configuration dictionary
        
    Returns:
        Dictionary with points, zMatrix, clusterMeta, clusterNames, clusterWords, datasetInfo
    """
    print(f"⚠️  Using MOCK DATA for dataset '{dataset}' (real data files not found)")
    
    # Generate synthetic points (smaller for demo)
    point_count = 500
    np.random.seed(42)
    
    # Create 5 clusters with some noise
    cluster_centers = [
        (0.2, 0.3), (0.7, 0.2), (0.5, 0.8), (0.1, 0.7), (0.8, 0.7)
    ]
    
    points = []
    for cluster_id, (cx, cy) in enumerate(cluster_centers):
        # Generate 80 points per cluster
        cluster_size = 80
        x = np.random.normal(cx, 0.08, cluster_size)
        y = np.random.normal(cy, 0.08, cluster_size)
        for i in range(cluster_size):
            points.append({
                "i": len(points),
                "x": float(x[i]),
                "y": float(y[i]),
                "c": cluster_id,
                "l": f"point_{len(points)}"
            })
    
    # Add noise points
    noise_count = 100
    noise_x = np.random.uniform(0, 1, noise_count)
    noise_y = np.random.uniform(0, 1, noise_count)
    for i in range(noise_count):
        points.append({
            "i": len(points),
            "x": float(noise_x[i]),
            "y": float(noise_y[i]),
            "c": -1,  # Noise cluster
            "l": f"noise_{i}"
        })
    
    # Generate hierarchical structure (simplified dendrogram)
    # Format: {child1, child2, distance, size}
    num_clusters = len(cluster_centers)
    z_matrix = []
    
    # Merge pairs of clusters
    z_matrix.append({"child1": 0, "child2": 1, "distance": 0.15, "size": 160})
    z_matrix.append({"child1": 2, "child2": 3, "distance": 0.18, "size": 160})
    z_matrix.append({"child1": num_clusters, "child2": 4, "distance": 0.25, "size": 240})
    z_matrix.append({"child1": num_clusters + 1, "child2": num_clusters + 2, "distance": 0.35, "size": 400})
    
    # Cluster metadata
    cluster_meta = {}
    for i in range(num_clusters):
        cluster_meta[str(i)] = {
            "s": 0.8 - i * 0.1,  # Stability (decreasing)
            "h": 1,               # Strahler number
            "z": 80               # Size
        }
    
    # Merged cluster metadata
    cluster_meta[str(num_clusters)] = {"s": 0.7, "h": 2, "z": 160}
    cluster_meta[str(num_clusters + 1)] = {"s": 0.6, "h": 2, "z": 160}
    cluster_meta[str(num_clusters + 2)] = {"s": 0.5, "h": 2, "z": 240}
    cluster_meta[str(num_clusters + 3)] = {"s": 0.4, "h": 3, "z": 400}
    
    # Cluster names and words
    cluster_names = {
        0: "Topic A",
        1: "Topic B",
        2: "Topic C",
        3: "Topic D",
        4: "Topic E",
        -1: "Noise"
    }
    
    cluster_words = {
        0: ["machine", "learning", "algorithm"],
        1: ["neural", "network", "deep"],
        2: ["data", "analysis", "statistics"],
        3: ["visualization", "graph", "plot"],
        4: ["clustering", "similarity", "distance"],
        -1: []
    }
    
    result = {
        "points": points,
        "zMatrix": z_matrix,
        "clusterMeta": cluster_meta,
        "clusterNames": cluster_names,
        "clusterWords": cluster_words,
        "datasetInfo": {
            "name": f"{config['name']} (Mock Data)",
            "pointCount": len(points),
            "clusterCount": num_clusters,
            "description": "⚠️ Synthetic data for development - replace with real data"
        }
    }
    
    return result


def get_mock_cache_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract cache data from mock result for use by other manager methods
    
    Returns:
        Dictionary with _points, _z_json, _cluster_metadata, etc.
    """
    return {
        "_points": result["points"],
        "_z_json": result["zMatrix"],
        "_cluster_metadata": result["clusterMeta"],
        "_cluster_names": result["clusterNames"],
        "_cluster_words": result["clusterWords"],
        "_point_cluster_map": {p["i"]: p["c"] for p in result["points"]}
    }
