"""
D3 Data Manager - Handles all data loading and processing for D3.js frontend
Integrates with existing HDBSCAN analysis infrastructure
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, cast
import numpy as np
import pickle
from pathlib import Path
import base64
import io

try:
    import cupy as cp
    from cuml.manifold import UMAP as cuMLUMAP
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# Mock data generator for development without real data files
try:
    from .mock_data_generator import generate_mock_data, get_mock_cache_data
    from .color_embedding import compute_cluster_colors_from_similarity
except ImportError:
    from mock_data_generator import generate_mock_data, get_mock_cache_data
    from color_embedding import compute_cluster_colors_from_similarity
def _get_leaves(condensed_tree):
    cluster_tree = condensed_tree[condensed_tree['child_size'] > 1]
    print(len(cluster_tree))
    if cluster_tree.shape[0] == 0:
        # Return the only cluster, the root
        return [condensed_tree['parent'].min()]

    root = cluster_tree['parent'].min()
    return _recurse_leaf_dfs(cluster_tree, root)
  
def _recurse_leaf_dfs(cluster_tree, current_node):
  children = cluster_tree[cluster_tree['parent'] == current_node]['child']
  if len(children) == 0:
      return [current_node,]
  else:
      return sum([_recurse_leaf_dfs(cluster_tree, child) for child in children], [])
class D3DataManager:
    """
    Manages data loading and transformation for the D3.js-based cluster explorer
    """
    
    def __init__(self):
        """Initialize the data manager"""
        # Base path set to project src directory (…/src/d3-app/src)
        self.base_path = Path(__file__).resolve().parents[2]
        self.current_dataset = None
        self.current_dr_method = None
        self.cached_data = {}
        self._similarity_dict = None  # Cache for similarity dictionary
        self._last_file_status = {}  # Track last checked file existence
        self._vectors = None  # Lazy-loaded high-dimensional vectors cache
        self._load_configuration()
        self._load_similarity_dict()  # Load similarity dictionary at startup
    
    def _load_configuration(self):
        """Load configuration for available datasets"""
        # ============================================================
        # DATA PATH CONFIGURATION - CHANGE THIS TO YOUR DATA LOCATION
        # ============================================================
        # データファイルのパス設定（実際のデータを用意したら変更してください）
        # Required files in data_path:
        #   - projection.npy: (N, 2) UMAP projection coordinates
        #   - word.npy: (N,) word labels
        #   - hdbscan_label.npy: (N,) HDBSCAN cluster labels
        #   - cluster_to_label.csv: cluster representative words
        #   - vector.npy: (N, 300) high-dimensional vectors
        #   - condensed_tree_object.pkl: HDBSCAN condensed tree (optional)
        # ============================================================
        self.datasets_config = {
            "default": {
                "name": "RAPIDS HDBSCAN Result",
                "description": "HDBSCAN clustering result from RAPIDS GPU acceleration",
                "data_path": "__RESOLVE__",  # resolved below
                "point_count": 100000,
                "cluster_count": 0,  # Will be determined from data
                "dr_methods": ["umap", "tsne", "pca"]
            }
        }

        # Resolve data path from candidates
        candidates = [
            self.base_path / "../data",     # src/d3-app/data
            self.base_path / "../../data"    # project-root/data
        ]
        resolved = None
        for c in candidates:
            if c.exists():
                resolved = c
                break
        if resolved is None:
            resolved = candidates[0]  # fallback
        self.datasets_config["default"]["data_path"] = str(resolved)
        # Use real data by default; set to True only if you want mock
        self.use_mock_data = False
    
    def _load_similarity_dict(self):
        """Load similarity dictionary for color embedding"""
        # Try to load from cluster_similarities.pkl
        similarity_file = self.base_path / "../data/cluster_similarities.pkl"
        
        if similarity_file.exists():
            try:
                with open(similarity_file, 'rb') as f:
                    self._similarity_dict = pickle.load(f)
                print(f"✓ Loaded similarity dictionary with metrics: {list(self._similarity_dict.keys())}")
            except Exception as e:
                print(f"⚠️  Could not load similarity dictionary: {e}")
                self._similarity_dict = None
        else:
            print(f"⚠️  Similarity file not found at {similarity_file}")
            self._similarity_dict = None
    
    def get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()
    
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Get list of available datasets"""
        return [
            {
                "name": config["name"],
                "id": dataset_id,
                "description": config["description"],
                "pointCount": config["point_count"],
                "drMethods": config["dr_methods"]
            }
            for dataset_id, config in self.datasets_config.items()
        ]

    def get_last_file_status(self) -> Dict[str, Any]:
        """Return last checked file existence snapshot"""
        return self._last_file_status or {}
    
    def get_initial_data(
        self,
        dataset: str,
        dr_method: str,
        dr_params: Optional[Dict[str, Any]] = None,
        color_mode: str = 'cluster',
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """Load and process initial data for visualization (simplified resilient version)."""

        print(f"[INIT_DATA] dataset={dataset} dr_method={dr_method} color_mode={color_mode} params={dr_params}", flush=True)

        # Cache handling
        cache_key = f"{dataset}_{dr_method}"
        cached = self.cached_data.get(cache_key)
        if (not force_reload) and cached is not None:
            print(f"[INIT_DATA] cache hit for {cache_key}", flush=True)
            return cached

        if dataset not in self.datasets_config:
            raise ValueError(f"Dataset '{dataset}' not found")

        config = self.datasets_config[dataset]
        data_path = Path(config["data_path"]).resolve()
        print(f"Loading data from: {data_path}", flush=True)

        # Required files
        required_files = {
            "projection.npy": data_path / "projection.npy",
            "word.npy": data_path / "word.npy",
            "point_cluster_map.npy": data_path / "point_cluster_map.npy",
            "hdbscan_label.npy": data_path / "hdbscan_label.npy",
            "vector.npy": data_path / "vector.npy",
        }
        self._last_file_status = {
            name: {"exists": p.exists(), "path": str(p)} for name, p in required_files.items()
        }
        for name, path in required_files.items():
            print(f"[DATA CHECK] {name}: exists={path.exists()} path={path}", flush=True)

        # If mock mode, short-circuit
        if self.use_mock_data or not required_files["projection.npy"].exists():
            print("⚠️  Using mock data (missing real files or mock enabled)", flush=True)
            result = generate_mock_data(dataset, dr_method, config)
            cache_data = get_mock_cache_data(result)
            for key, value in cache_data.items():
                setattr(self, key, value)
            self.cached_data[cache_key] = result
            return result

        try:
            embedding = np.load(required_files["projection.npy"])  # (N, 2)
            labels = np.load(required_files["word.npy"])  # (N,)
            point_cluster_labels = np.load(required_files["point_cluster_map.npy"])  # (N,)
            hdbscan_labels = np.load(required_files["hdbscan_label.npy"])  # (N,)
            point_count = len(embedding)
        except Exception as e:
            raise ValueError(f"Error loading data files: {e}")

        # Build points list
        points = []
        unique_clusters = set()
        for i in range(point_count):
            cluster_id = -1 if hdbscan_labels[i] == -1 else int(point_cluster_labels[i])
            if cluster_id != -1:
                unique_clusters.add(cluster_id)
            point = {
                "i": int(i),
                "x": float(embedding[i, 0]),
                "y": float(embedding[i, 1]),
                "c": cluster_id,
                "l": str(labels[i])
            }
            points.append(point)

        # Load HDBSCAN condensed tree (optional for dendrogram)
        hdbscan_file = data_path / "condensed_tree_object.pkl"
        hdbscan_condensed_tree = None
        linkage_matrix = None
        old_new_id_map = {}
        
        if hdbscan_file.exists():
            try:
                with open(hdbscan_file, 'rb') as f:
                    hdbscan_condensed_tree = pickle.load(f)
                print(f"✓ Loaded HDBSCAN condensed tree")
                
                # Convert to linkage matrix format
                linkage_matrix, old_new_id_map = self._get_linkage_matrix_from_hdbscan(
                    hdbscan_condensed_tree
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not load HDBSCAN tree: {e}")
        else:
            print(f"⚠️  No condensed_tree_object.pkl found")
            linkage_matrix = []
        
        # Load cluster metadata
        if hdbscan_condensed_tree is not None:
            cluster_meta = self._compute_cluster_metadata(hdbscan_condensed_tree)
            # Normalize keys: 'size' -> 'z', 'stability' -> 's', 'strahler' -> 'h'
            cluster_meta_normalized = {}
            for cid, meta in cluster_meta.items():
                cluster_meta_normalized[str(cid)] = {
                    "s": float(meta.get("stability", 0)),
                    "h": int(meta.get("strahler", 1)),
                    "z": int(meta.get("size", 0))
                }
            cluster_meta = cluster_meta_normalized
        else:
            cluster_meta = {}
        
        # Load cluster labels and words
        cluster_label_file = data_path / "cluster_to_label.csv"
        cluster_names, cluster_words = self._load_cluster_labels(cluster_label_file)
        
        # Build point_cluster_map from loaded data
        point_cluster_map = {}
        for i in range(point_count):
            point_cluster_map[i] = int(point_cluster_labels[i])
        
        # Build clusterIdMap if we have old_new_id_map from HDBSCAN
        clusterIdMap = {v: k for k, v in old_new_id_map.items()} if old_new_id_map else {}

        # Cache raw arrays
        self._embedding2d = embedding
        self._labels = labels
        self._cluster_labels = point_cluster_labels
        self._hdbscan_labels = hdbscan_labels

        # Convert linkage matrix (c1, c2, parent, dist, size) to JSON-compatible format
        z_matrix = []
        if linkage_matrix is not None and len(linkage_matrix) > 0:
            z_matrix = [
                {
                    "child1": int(row[0]),
                    "child2": int(row[1]),
                    "parent": int(row[2]),
                    "distance": float(row[3]),
                    "size": int(row[4])
                }
                for row in linkage_matrix
            ]
        
        # Count unique clusters (excluding noise = -1)
        unique_clusters = set()
        for i in range(point_count):
            cluster_id = -1 if hdbscan_labels[i] == -1 else int(point_cluster_labels[i])
            if cluster_id != -1:
                unique_clusters.add(cluster_id)

        result = {
            "points": points,
            "zMatrix": z_matrix,
            "clusterMeta": cluster_meta,
            "clusterNames": cluster_names,
            "clusterWords": cluster_words,
            "clusterIdMap": clusterIdMap,
            "clusterSimilarities": None,
            "datasetInfo": {
                "name": config.get("name", dataset),
                "pointCount": point_count,
                "clusterCount": len(unique_clusters),
                "description": config.get("description", "")
            }
        }

        self.cached_data[cache_key] = result
        return result


    def get_vectors(self, point_ids: List[int], dataset: str = "default") -> np.ndarray:
        """Return high-dimensional vectors for given point IDs.

        Lazily loads vector.npy and caches it in memory to serve multiple requests
        without reloading from disk.
        """
        if dataset not in self.datasets_config:
            raise ValueError(f"Dataset '{dataset}' not found")

        # Lazy-load vectors file
        if self._vectors is None:
            data_path = Path(self.datasets_config[dataset]["data_path"]).resolve()
            vectors_file = data_path / "vector.npy"
            if not vectors_file.exists():
                raise FileNotFoundError(f"Vector file not found at {vectors_file}")
            self._vectors = np.load(vectors_file)
            print(f"✓ Loaded vectors into memory: {self._vectors.shape}")

        max_id = self._vectors.shape[0] - 1
        for pid in point_ids:
            if pid < 0 or pid > max_id:
                raise ValueError(f"Point ID {pid} out of range [0, {max_id}]")

        return self._vectors[point_ids]
    
    def get_initial_data_no_noise(
        self,
        dataset: str,
        dr_method: str,
        dr_params: Optional[Dict[str, Any]] = None,
        color_mode: str = 'cluster'
    ) -> Dict[str, Any]:
        """
        Load and process initial data for visualization with noise points filtered out
        
        Same as get_initial_data but excludes points with cluster_id = -1 (noise)
        
        Parameters:
        - dataset: Dataset name
        - dr_method: Dimensionality reduction method
        - dr_params: DR parameters
        - color_mode: Color assignment mode ('cluster' or 'distance')
        
        Returns JSON-compatible data structure with noise-filtered data
        """
        # First get the full data
        full_data = self.get_initial_data(dataset, dr_method, dr_params, color_mode)

        if not full_data or "points" not in full_data or full_data["points"] is None:
            raise ValueError("Initial data could not be loaded (no points returned). Check dataset path and files.")
        
        # Filter out noise points (cluster_id = -1)
        filtered_points = [p for p in full_data["points"] if p["c"] != -1]
        
        # Reindex points
        point_id_map = {}  # old_id -> new_id
        new_points = []
        for new_idx, point in enumerate(filtered_points):
            point_id_map[point["i"]] = new_idx
            new_point = point.copy()
            new_point["i"] = new_idx
            new_points.append(new_point)
        
        # Filter linkage matrix to only include non-noise clusters
        non_noise_cluster_ids = set(p["c"] for p in filtered_points)
        print(f"DEBUG: non_noise_cluster_ids size: {len(non_noise_cluster_ids)}")
        print(f"DEBUG: Sample non_noise_cluster_ids: {list(non_noise_cluster_ids)[:10]}")
        print(f"DEBUG: full_data clusterNames keys sample: {list(full_data['clusterNames'].keys())[:10]}")
        
        # Helper function to safely convert key to int
        def safe_int(val):
            if isinstance(val, (int, np.integer)):
                return int(val)
            elif isinstance(val, str):
                return int(val)
            else:
                # If it's an array or something else, try to extract scalar
                return int(np.asarray(val).item())
        
        filtered_metadata = {
            cid: meta for cid, meta in full_data["clusterMeta"].items()
            if safe_int(cid) in non_noise_cluster_ids
        }
        
        # Update point count in datasetInfo
        result = {
            "points": new_points,
            "zMatrix": full_data["zMatrix"],  # Keep unchanged
            "clusterMeta": filtered_metadata,
            "clusterNames": {
                k: v for k, v in full_data["clusterNames"].items()
                if safe_int(k) in non_noise_cluster_ids
            },
            "clusterWords": {
                k: v for k, v in full_data["clusterWords"].items()
                if safe_int(k) in non_noise_cluster_ids
            },
            "clusterIdMap": full_data.get("clusterIdMap", {}),  # Include clusterIdMap from full_data
            "datasetInfo": {
                **full_data["datasetInfo"],
                "pointCount": len(new_points),
                "originalPointCount": full_data["datasetInfo"]["pointCount"],
                "description": full_data["datasetInfo"]["description"] + " (noise filtered)"
            }
        }
        
        return result
    
    def get_clusters_from_point_selection(
        self,
        point_ids: List[int],
        containment_ratio_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Map selected points to clusters with containment filtering
        
        Args:
            point_ids: List of selected point IDs
            containment_ratio_threshold: Minimum ratio of selected points in cluster
                                         to include that cluster in results
        
        Returns:
            Dict with cluster IDs and statistics
        """
        if not hasattr(self, '_point_cluster_map') or not self._point_cluster_map:
            # Build the map if not cached
            # This assumes we have loaded HDBSCAN data
            return {
                "cluster_ids": [],
                "stats": {
                    "totalSelectedPoints": len(point_ids),
                    "uniqueClusters": 0,
                    "details": {}
                }
            }
        
        # Count points per cluster in selection
        cluster_counts: Dict[int, int] = {}

        for point_id in point_ids:
            if point_id in self._point_cluster_map:
                cluster_id = self._point_cluster_map[point_id]
                # Skip noise cluster
                if cluster_id == -1:
                    continue
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        # Build total cluster sizes from cached metadata
        total_cluster_sizes: Dict[int, int] = {}
        if hasattr(self, '_cluster_metadata') and self._cluster_metadata:
            for cid, meta in self._cluster_metadata.items():
                # Skip noise
                if cid == -1:
                    continue
                size = int(meta.get('size', meta.get('z', 0)) or 0)
                if size > 0:
                    total_cluster_sizes[cid] = size

        # Filter by containment ratio using actual cluster sizes
        selected_clusters = {}
        for cluster_id, count in cluster_counts.items():
            total_size = total_cluster_sizes.get(cluster_id, 0)
            if total_size <= 0:
                continue
            containment_ratio = count / total_size

            if containment_ratio >= containment_ratio_threshold:
                selected_clusters[cluster_id] = {
                    "selectedPoints": count,
                    "totalSize": total_size,
                    "containmentRatio": containment_ratio
                }
        
        return {
            "cluster_ids": list(selected_clusters.keys()),
            "stats": {
                "totalSelectedPoints": len(point_ids),
                "uniqueClusters": len(selected_clusters),
                "details": selected_clusters
            }
        }
    
    def get_heatmap_data(
        self,
        metric: str = "mahalanobis",
        top_n: int = 200,
        cluster_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Get similarity matrix for heatmap visualization
        
        Args:
            metric: Similarity metric name (e.g., 'jaccard', 'cosine', 'kl', 'mahalanobis')
            top_n: Number of top clusters to include
            cluster_ids: Specific cluster IDs to include (if None, use top_n)
        
        Returns:
            Similarity matrix and cluster ordering
        """
        print(f"Loading heatmap data with metric: {metric}, top_n: {top_n}")
        try:
            # Try to load precomputed similarity matrices
            data_path = self.base_path / self.datasets_config["default"]["data_path"]
            similarity_file = data_path / "cluster_similarities.pkl"
            print(f"Loading similarity data from: {similarity_file}")
            print(f"File exists: {similarity_file.exists()}")
            metric = "mahalanobis_distance"
            
            if similarity_file.exists():
                with open(similarity_file, 'rb') as f:
                    similarities = pickle.load(f)[metric]
                
                
                
                # similarities is a dict with (cluster_id, cluster_id) as keys
                # Extract unique cluster IDs
                if isinstance(similarities, dict):
                    cluster_id_set = set()
                    for key in similarities.keys():
                        if isinstance(key, tuple) and len(key) == 2:
                            cluster_id_set.add(key[0])
                            cluster_id_set.add(key[1])
                    
                    available_clusters = sorted(list(cluster_id_set))
                    print(f"✓ Found {len(available_clusters)} clusters in similarity data")
                    
                    # Determine which clusters to include
                    if cluster_ids:
                        order = [c for c in cluster_ids if c in available_clusters]
                    else:
                        order = available_clusters[:top_n]
                    
                    if not order:
                        print(f"⚠️  No valid clusters found")
                    else:
                        # Build similarity matrix from dict
                        n = len(order)
                        sim_matrix = np.zeros((n, n))
                        
                        for i, c1 in enumerate(order):
                            for j, c2 in enumerate(order):
                                # Try both key orders (symmetric)
                                key1 = (c1, c2)
                                key2 = (c2, c1)
                                if key1 in similarities:
                                    sim_matrix[i, j] = similarities[key1]
                                elif key2 in similarities:
                                    sim_matrix[i, j] = similarities[key2]
                                elif i == j:
                                    sim_matrix[i, j] = 1.0  # Self-similarity
                        
                        print(f"✓ Built {n}x{n} similarity matrix")
                        return {
                            "matrix": sim_matrix.tolist(),
                            "clusterOrder": order,
                            "metric": metric
                        }
                else:
                    print(f"⚠️  Unexpected similarity data format: {type(similarities)}")
            else: 
                print(f"⚠️  Similarity file not found: {similarity_file}")
            
            # Fallback: generate mock similarity matrix
            print(f"⚠️  Using mock similarity matrix (no precomputed data)")
            if cluster_ids:
                n_clusters = len(cluster_ids)
                order = cluster_ids
            else:
                n_clusters = min(top_n, 10)  # Limit to 10 for mock
                order = list(range(n_clusters))
            
            # Generate random but symmetric similarity matrix
            np.random.seed(42)
            sim_matrix = np.random.rand(n_clusters, n_clusters)
            sim_matrix = (sim_matrix + sim_matrix.T) / 2  # Make symmetric
            np.fill_diagonal(sim_matrix, 1.0)  # Self-similarity = 1
            
            return {
                "matrix": sim_matrix.tolist(),
                "clusterOrder": order,
                "metric": metric
            }
            
        except Exception as e:
            print(f"Error loading heatmap data: {e}")
            import traceback
            traceback.print_exc()
            return {
                "matrix": [],
                "clusterOrder": [],
                "metric": metric
            }
    
    def get_cluster_detail(self, cluster_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a cluster
        
        Args:
            cluster_id: ID of the cluster to retrieve
            
        Returns:
            Dict with cluster metadata, label, and statistics
        """
        try:
            # Use cached cluster metadata if available
            if not hasattr(self, '_cluster_metadata'):
                return {
                    "id": cluster_id,
                    "size": 0,
                    "label": f"Cluster {cluster_id}",
                    "top_words": [],
                    "stability": 0.0,
                    "strahler": 0,
                    "parent_cluster": -1,
                    "child_clusters": [],
                    "exemplars": []
                }
            
            cluster_meta = self._cluster_metadata.get(cluster_id, {})
            
            return {
                "id": cluster_id,
                "size": cluster_meta.get("size", 0),
                "label": self._cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
                "top_words": self._cluster_words.get(cluster_id, []),
                "stability": float(cluster_meta.get("stability", 0.0)),
                "strahler": int(cluster_meta.get("strahler", 0)),
                "parent_cluster": cluster_meta.get("parent_cluster", -1),
                "child_clusters": cluster_meta.get("child_clusters", []),
                "exemplars": cluster_meta.get("exemplars", [])
            }
        except Exception as e:
            print(f"Error getting cluster detail: {e}")
            return {
                "id": cluster_id,
                "size": 0,
                "label": f"Cluster {cluster_id}",
                "top_words": [],
                "stability": 0.0,
                "strahler": 0,
                "parent_cluster": -1,
                "child_clusters": [],
                "exemplars": []
            }
    
    def get_point_detail(self, point_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a point
        
        Args:
            point_id: ID of the point to retrieve
            
        Returns:
            Dict with point metadata and coordinates
        """
        try:
            # Use cached point data if available
            if not hasattr(self, '_points'):
                return {
                    "id": point_id,
                    "label": "",
                    "coordinates": {"x": 0.0, "y": 0.0},
                    "clusterId": -1,
                    "nearbyPoints": []
                }
            
            # Find point in cached data
            for point in self._points:
                if point["i"] == point_id:
                    # Best-effort typed extraction with ignores for pylance type hints
                    x_val = float(point.get("x", 0.0))  # type: ignore[arg-type]
                    y_val = float(point.get("y", 0.0))  # type: ignore[arg-type]
                    c_val = int(point.get("c", -1))     # type: ignore
                    return {
                        "id": point_id,
                        "label": str(point.get("l", "")),
                        "coordinates": {"x": x_val, "y": y_val},
                        "clusterId": c_val,
                        "nearbyPoints": []  # Would need KNN search to populate
                    }
            
            return {
                "id": point_id,
                "label": "",
                "coordinates": {"x": 0.0, "y": 0.0},
                "clusterId": -1,
                "nearbyPoints": []
            }
        except Exception as e:
            print(f"Error getting point detail: {e}")
            return {
                "id": point_id,
                "label": "",
                "coordinates": {"x": 0.0, "y": 0.0},
                "clusterId": -1,
                "nearbyPoints": []
            }
    
    def filter_dendrogram(
        self,
        strahler_range: Optional[List[float]] = None,
        stability_range: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Filter dendrogram by Strahler and Stability values
        
        Args:
            strahler_range: [min, max] Strahler order to include
            stability_range: [min, max] stability range to include
            
        Returns:
            Filtered linkage matrix and visible cluster IDs
        """
        try:
            if not hasattr(self, '_linkage_matrix') or not hasattr(self, '_cluster_metadata'):
                return {
                    "filteredLinkageMatrix": [],
                    "visibleClusterIds": []
                }
            
            visible_clusters = set()
            
            # Filter by Strahler and Stability
            for cluster_id, meta in self._cluster_metadata.items():
                strahler = meta.get("strahler", 0)
                stability = meta.get("stability", 0.0)
                
                include = True
                if strahler_range and (strahler < strahler_range[0] or strahler > strahler_range[1]):
                    include = False
                if stability_range and (stability < stability_range[0] or stability > stability_range[1]):
                    include = False
                
                if include:
                    visible_clusters.add(cluster_id)
            
            # Filter linkage matrix
            filtered_linkage = []
            for entry in self._linkage_matrix:
                # Include if both children/parent are in visible clusters
                if entry.get("parent_cluster") in visible_clusters:
                    filtered_linkage.append(entry)
            
            return {
                "filteredLinkageMatrix": filtered_linkage,
                "visibleClusterIds": list(visible_clusters)
            }
        except Exception as e:
            print(f"Error filtering dendrogram: {e}")
            return {
                "filteredLinkageMatrix": [],
                "visibleClusterIds": []
        }
    
    # ========================================================================
    # Private helper methods
    # ========================================================================
    
    def _get_linkage_matrix_from_hdbscan(self, condensed_tree) -> Tuple[np.ndarray, Dict]:
        """
        Convert HDBSCAN condensed tree to scipy-compatible linkage matrix
        
        Returns:
            Z (np.ndarray): Linkage matrix of shape (n_merges, 5) with columns
                [child1, child2, parent, distance, cluster_size]
            old_new_id_map (Dict): Mapping from original node IDs to contiguous IDs
        """
        try:
            import pandas as pd
            
            print("Generating linkage matrix from HDBSCAN condensed tree...")
            linkage_matrix = []
            raw_tree = condensed_tree._raw_tree
            condensed_df = condensed_tree.to_pandas()
            
            # Filter for merge operations (child_size > 1)
            cluster_tree = condensed_df[condensed_df['child_size'] > 1]
            sorted_condensed_tree = cluster_tree.sort_values(
                by=['lambda_val', 'parent'], 
                ascending=True
            )
            print(f"Processing {len(sorted_condensed_tree)} merge operations")
            
            # Reconstruct merge operations from sorted tree
            for i in range(0, len(sorted_condensed_tree), 2):
                if i + 1 < len(sorted_condensed_tree):
                    row_a = sorted_condensed_tree.iloc[i]
                    row_b = sorted_condensed_tree.iloc[i + 1]
                    
                    # Validate that pairs have same lambda and parent
                    if row_a['lambda_val'] != row_b['lambda_val']:
                        print(f"Warning: Lambda value mismatch at rows {i},{i+1}")
                        continue
                    
                    child_a = row_a['child']
                    child_b = row_b['child']
                    lam = row_a['lambda_val']
                    parent_id = row_a['parent']
                    
                    # Get total cluster size from raw tree
                    total_size_rows = raw_tree[raw_tree['child'] == parent_id]['child_size']
                    if len(total_size_rows) == 0:
                        total_size = row_a['child_size'] + row_b['child_size']
                    else:
                        total_size = total_size_rows[0]
                    
                    linkage_matrix.append([
                        int(child_a),
                        int(child_b),
                        int(parent_id),
                        lam,
                        int(total_size)
                    ])
            
            print(f"Generated {len(linkage_matrix)} linkage operations")
            
            # Map leaf nodes to 0..N-1
            old_new_id_map = {}
            current_id = 0
            
            # Extract leaf nodes (child_size == 1)
            # leaf_rows = raw_tree[raw_tree['child_size'] == 1]
            # leaves = sorted(set(leaf_rows['child'].tolist()))
            # print(f"Found {len(leaves)} leaf nodes")
            leaves = _get_leaves(raw_tree)
            
            for leaf in leaves:
                old_new_id_map[int(leaf)] = current_id
                current_id += 1
            
            # Map internal nodes
            for row in reversed(linkage_matrix):
                parent_id = row[2]
                if int(parent_id) not in old_new_id_map:
                    old_new_id_map[int(parent_id)] = current_id
                    current_id += 1
            
            print(f"Created node ID mapping: {len(old_new_id_map)} nodes")
            # max
            
            
            # Remap linkage matrix to contiguous IDs
            max_lambda = max(row[3] for row in linkage_matrix) if linkage_matrix else 1.0
            linkage_matrix_mapped = [
                [
                    old_new_id_map[int(row[0])],
                    old_new_id_map[int(row[1])],
                    old_new_id_map[int(row[2])],
                    max_lambda - row[3],  # Invert distance
                    row[4]  # Cluster size
                ]
                for row in reversed(linkage_matrix)
            ]
            
            return np.array(linkage_matrix_mapped), old_new_id_map
            
        except Exception as e:
            print(f"Error generating linkage matrix: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), {}
    
    def _compute_cluster_metadata(self, hdbscan_condensed_tree) -> Dict[int, Dict]:
        """
        Compute stability and Strahler numbers for clusters
        
        Returns:
            Dict mapping cluster ID to metadata dict with 'stability' and 'strahler'
        """
        try:
            metadata = {}
            raw_tree = hdbscan_condensed_tree._raw_tree
            
            # Compute stability (lambda * cluster_size for persistent clusters)
            for row in raw_tree:
                cluster_id = int(row['parent'])
                stability = float(row['lambda_val'] * row['child_size'])
                
                if cluster_id not in metadata:
                    metadata[cluster_id] = {
                        'stability': stability,
                        'strahler': 1,
                        'size': int(row['child_size'])
                    }
                else:
                    # Update with max stability
                    metadata[cluster_id]['stability'] = max(
                        metadata[cluster_id]['stability'],
                        stability
                    )
            
            # Compute Strahler numbers (tree depth measure)
            def compute_strahler(cluster_id, tree_data):
                # Find child clusters of this cluster
                children = tree_data[tree_data['parent'] == cluster_id]['child'].unique()
                if len(children) == 0:
                    return 1
                
                strahler_values = []
                for child in children:
                    strahler_values.append(compute_strahler(child, tree_data))
                
                strahler_values.sort(reverse=True)
                if len(strahler_values) >= 2:
                    if strahler_values[0] == strahler_values[1]:
                        return strahler_values[0] + 1
                    else:
                        return strahler_values[0]
                elif len(strahler_values) == 1:
                    return strahler_values[0]
                else:
                    return 1
            
            # Update Strahler numbers
            root_cluster = raw_tree['parent'].max()
            for cluster_id in metadata:
                try:
                    metadata[cluster_id]['strahler'] = compute_strahler(cluster_id, raw_tree)
                except:
                    metadata[cluster_id]['strahler'] = 1
            
            print(f"Computed metadata for {len(metadata)} clusters")
            return metadata
            
        except Exception as e:
            print(f"Error computing cluster metadata: {e}")
            return {}
    
    def _load_cluster_labels(self, label_file: Path) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
        """
        Load cluster labels and representative words from cluster_to_label.csv
        
        Expected CSV format:
        cluster_id,representative_label,word1,word2,word3,...,word10
        """
        cluster_names: Dict[int, str] = {}
        cluster_words: Dict[int, List[str]] = {}
        
        if not label_file.exists():
            print(f"⚠️  Cluster label file not found: {label_file}")
            return cluster_names, cluster_words
        
        try:
            import csv
            with open(label_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cluster_id = int(row.get('cluster_id', -1))
                    if cluster_id < 0:
                        continue
                    
                    # Get representative label
                    label = row.get('representative_label', f'Cluster {cluster_id}')
                    cluster_names[cluster_id] = label
                    
                    # Collect word columns (word1, word2, ..., word10)
                    words = []
                    for i in range(1, 11):  # word1 to word10
                        word_key = f'word{i}'
                        if word_key in row and row[word_key]:
                            words.append(row[word_key].strip())
                    
                    cluster_words[cluster_id] = words
            
            print(f"✓ Loaded labels for {len(cluster_names)} clusters")
        except Exception as e:
            print(f"Warning: Error loading cluster labels: {e}")
        
        return cluster_names, cluster_words
    
    def _create_simplified_linkage_from_labels(self, cluster_labels: np.ndarray) -> np.ndarray:
        """
        Create a simplified linkage matrix from cluster labels when condensed tree is not available
        
        Args:
            cluster_labels: (N,) array of cluster labels
            
        Returns:
            Linkage matrix in scipy format: (n_clusters-1, 5) [child1, child2, parent, dist, size]
        """
        unique_clusters = [c for c in np.unique(cluster_labels) if c >= 0]
        n_clusters = len(unique_clusters)
        
        if n_clusters <= 1:
            return np.array([])
        
        # Create simple binary tree where clusters merge in order
        linkage = []
        current_parent = n_clusters
        
        for i in range(n_clusters - 1):
            child1 = unique_clusters[i]
            child2 = unique_clusters[i + 1] if i < n_clusters - 1 else current_parent - 1
            
            # Count sizes
            size1 = np.sum(cluster_labels == child1)
            size2 = np.sum(cluster_labels == child2) if child2 < n_clusters else size1
            
            linkage.append([
                child1,
                child2,
                current_parent,
                float(i + 1),  # Distance as merge order
                size1 + size2
            ])
            current_parent += 1
        
        return np.array(linkage)
    
    def _compute_cluster_metadata_from_labels(self, cluster_labels: np.ndarray) -> Dict[int, Dict]:
        """
        Compute basic cluster metadata from labels when condensed tree is not available
        
        Args:
            cluster_labels: (N,) array of cluster labels
            
        Returns:
            Dict mapping cluster ID to metadata dict
        """
        metadata = {}
        unique_clusters = [c for c in np.unique(cluster_labels) if c >= 0]
        
        for cluster_id in unique_clusters:
            size = int(np.sum(cluster_labels == cluster_id))
            metadata[int(cluster_id)] = {
                'stability': float(size),  # Use size as proxy for stability
                'strahler': 1,  # Default Strahler number
                'size': size
            }
        
        print(f"✓ Computed basic metadata for {len(metadata)} clusters")
        return metadata
    
    def _create_point_cluster_map(
        self,
        hdbscan_condensed_tree,
        point_count: int
    ) -> Dict[int, int]:
        """
        Create mapping from point ID to cluster ID
        
        Args:
            hdbscan_condensed_tree: HDBSCAN condensed tree object
            point_count: Total number of points in dataset
            
        Returns:
            Dict mapping point_id (0..point_count-1) to cluster_id
        """
        try:
            point_cluster_map = {}
            raw_tree = hdbscan_condensed_tree._raw_tree
            
            # Extract leaf nodes (representing individual points)
            leaf_rows = raw_tree[raw_tree['child_size'] == 1]
            
            for row in leaf_rows:
                point_id = int(row['child'])
                cluster_id = int(row['parent'])
                point_cluster_map[point_id] = cluster_id
            
            print(f"Created point-to-cluster mapping for {len(point_cluster_map)} points")
            return point_cluster_map
            
        except Exception as e:
            print(f"Error creating point-cluster map: {e}")
            return {}
    
    # ========================================================================
    # Zoom Feature: GPU-Accelerated UMAP Redraw with Initial Position Preservation
    # ========================================================================
    
    @staticmethod
    def _b64_to_numpy(data_b64: str) -> np.ndarray:
        """Decode Base64 string to NumPy array"""
        decoded = base64.b64decode(data_b64)
        return np.load(io.BytesIO(decoded))
    
    @staticmethod
    def _numpy_to_b64(array: np.ndarray) -> str:
        """Encode NumPy array to Base64 string"""
        buff = io.BytesIO()
        np.save(buff, array, allow_pickle=False)
        return base64.b64encode(buff.getvalue()).decode('utf-8')
    
    def zoom_redraw(
        self,
        point_ids: List[int],
        dr_method: str = "umap",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_epochs: int = 200
    ) -> Dict[str, Any]:
        """
        Redraw 2D projection for selected points using GPU-accelerated UMAP.
        Preserves mental map by using current coordinates as initial positions.
        
        Args:
            point_ids: List of point indices to zoom into
            dr_method: Dimensionality reduction method (currently only 'umap' supported)
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            n_epochs: Number of UMAP epochs
            
        Returns:
            Dict with:
                - status: "success" or "error"
                - coordinates: Base64-encoded (N_selected, 2) embedding
                - shape: [N_selected, 2]
                - point_ids: Original point indices for mapping
        """
        if not HAS_GPU:
            return {
                "status": "error",
                "message": "GPU UMAP not available. Install cupy and cuml."
            }
        
        if dr_method != "umap":
            return {
                "status": "error",
                "message": f"Only 'umap' is supported for zoom redraw, got '{dr_method}'"
            }
        
        try:
            # Validate point_ids
            point_ids = [int(p) for p in point_ids]
            if not self._embedding2d is None:
                max_id = self._embedding2d.shape[0] - 1
                if any(p < 0 or p > max_id for p in point_ids):
                    return {
                        "status": "error",
                        "message": f"Point IDs out of range [0, {max_id}]"
                    }
            
            # Load high-dimensional vectors for selected points
            vectors_file = self.base_path / self.datasets_config["default"]["data_path"] / "vector.npy"
            if not vectors_file.exists():
                return {
                    "status": "error",
                    "message": f"Vector file not found at {vectors_file}"
                }
            
            all_vectors = np.load(vectors_file)  # (N, D)
            selected_vectors = all_vectors[point_ids]  # (N_selected, D)
            
            print(f"✓ Loaded {len(point_ids)} vectors from high-dimensional space: {selected_vectors.shape}")
            
            # Get current 2D coordinates as initial positions
            current_coords = self._embedding2d[point_ids]  # (N_selected, 2)
            print(f"✓ Extracted current coordinates for initial positions: {current_coords.shape}")
            
            # Transfer to GPU
            vectors_gpu = cp.asarray(selected_vectors, dtype=cp.float32)
            init_gpu = cp.asarray(current_coords, dtype=cp.float32)
            
            print(f"✓ Transferred data to GPU: vectors {vectors_gpu.shape}, init {init_gpu.shape}")
            
            # Create UMAP model with initial positions
            umap_model = cuMLUMAP(
                n_components=2,
                n_neighbors=min(n_neighbors, len(point_ids) - 1),  # Adjust if fewer points
                min_dist=min_dist,
                metric="euclidean",
                random_state=42,
                init=init_gpu,  # Use current positions as mental map reference
                n_epochs=n_epochs,
                verbose=True
            )
            
            # Execute GPU UMAP
            embedding_gpu = umap_model.fit_transform(vectors_gpu)
            cp.cuda.runtime.deviceSynchronize()  # Ensure GPU computation completes
            
            # Transfer back to CPU
            embedding_cpu = cp.asnumpy(embedding_gpu)
            print(f"✓ UMAP computation complete: {embedding_cpu.shape}")
            
            # Encode result to Base64
            embedding_b64 = self._numpy_to_b64(embedding_cpu)
            
            return {
                "status": "success",
                "coordinates": embedding_b64,
                "shape": list(embedding_cpu.shape),
                "point_ids": point_ids
            }
        
        except ValueError as e:
            print(f"ValueError in zoom_redraw: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
        except Exception as e:
            print(f"Error in zoom_redraw: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Internal error during zoom redraw: {str(e)}"
            }
    
    # ========================================================================
    # Private helper methods for HDBSCAN data processing
    # ========================================================================
    
    def _get_linkage_matrix_from_hdbscan(self, condensed_tree) -> Tuple[np.ndarray, Dict]:
        """
        Convert HDBSCAN condensed tree to scipy-compatible linkage matrix
        
        Returns:
            Z (np.ndarray): Linkage matrix of shape (n_merges, 5) with columns
                [child1, child2, parent, distance, cluster_size]
            old_new_id_map (Dict): Mapping from original node IDs to contiguous IDs
        """
        try:
            import pandas as pd
            
            print("Generating linkage matrix from HDBSCAN condensed tree...")
            linkage_matrix = []
            raw_tree = condensed_tree._raw_tree
            condensed_df = condensed_tree.to_pandas()
            
            # Filter for merge operations (child_size > 1)
            cluster_tree = condensed_df[condensed_df['child_size'] > 1]
            sorted_condensed_tree = cluster_tree.sort_values(
                by=['lambda_val', 'parent'], 
                ascending=True
            )
            print(f"Processing {len(sorted_condensed_tree)} merge operations")
            
            # Reconstruct merge operations from sorted tree
            for i in range(0, len(sorted_condensed_tree), 2):
                if i + 1 < len(sorted_condensed_tree):
                    row_a = sorted_condensed_tree.iloc[i]
                    row_b = sorted_condensed_tree.iloc[i + 1]
                    
                    # Validate that pairs have same lambda and parent
                    if row_a['lambda_val'] != row_b['lambda_val']:
                        print(f"Warning: Lambda value mismatch at rows {i},{i+1}")
                        continue
                    
                    child_a = row_a['child']
                    child_b = row_b['child']
                    lam = row_a['lambda_val']
                    parent_id = row_a['parent']
                    
                    # Get total cluster size from raw tree
                    total_size_rows = raw_tree[raw_tree['child'] == parent_id]['child_size']
                    if len(total_size_rows) == 0:
                        total_size = row_a['child_size'] + row_b['child_size']
                    else:
                        total_size = total_size_rows[0]
                    
                    linkage_matrix.append([
                        int(child_a),
                        int(child_b),
                        int(parent_id),
                        lam,
                        int(total_size)
                    ])
            
            print(f"Generated {len(linkage_matrix)} linkage operations")
            
            # Map leaf nodes to 0..N-1
            old_new_id_map = {}
            current_id = 0
            
            # Extract leaf nodes using helper function
            leaves = _get_leaves(raw_tree)
            
            for leaf in leaves:
                old_new_id_map[int(leaf)] = current_id
                current_id += 1
            
            # Map internal nodes
            for row in reversed(linkage_matrix):
                parent_id = row[2]
                if int(parent_id) not in old_new_id_map:
                    old_new_id_map[int(parent_id)] = current_id
                    current_id += 1
            
            print(f"Created node ID mapping: {len(old_new_id_map)} nodes")
            
            # Remap linkage matrix to contiguous IDs
            max_lambda = max(row[3] for row in linkage_matrix) if linkage_matrix else 1.0
            linkage_matrix_mapped = [
                [
                    old_new_id_map[int(row[0])],
                    old_new_id_map[int(row[1])],
                    old_new_id_map[int(row[2])],
                    max_lambda - row[3],  # Invert distance
                    row[4]  # Cluster size
                ]
                for row in reversed(linkage_matrix)
            ]
            
            return np.array(linkage_matrix_mapped), old_new_id_map
            
        except Exception as e:
            print(f"Error generating linkage matrix: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), {}
    
    def _compute_cluster_metadata(self, hdbscan_condensed_tree) -> Dict[int, Dict]:
        """
        Compute stability and Strahler numbers for clusters
        
        Returns:
            Dict mapping cluster ID to metadata dict with 'stability' and 'strahler'
        """
        try:
            metadata = {}
            raw_tree = hdbscan_condensed_tree._raw_tree
            
            # Compute stability (lambda * cluster_size for persistent clusters)
            for row in raw_tree:
                cluster_id = int(row['parent'])
                stability = float(row['lambda_val'] * row['child_size'])
                
                if cluster_id not in metadata:
                    metadata[cluster_id] = {
                        'stability': stability,
                        'strahler': 1,
                        'size': int(row['child_size'])
                    }
                else:
                    # Update with max stability
                    metadata[cluster_id]['stability'] = max(
                        metadata[cluster_id]['stability'],
                        stability
                    )
            
            # Compute Strahler numbers (tree depth measure)
            def compute_strahler(cluster_id, tree_data):
                # Find child clusters of this cluster
                children = tree_data[tree_data['parent'] == cluster_id]['child'].unique()
                if len(children) == 0:
                    return 1
                
                strahler_values = []
                for child in children:
                    strahler_values.append(compute_strahler(child, tree_data))
                
                strahler_values.sort(reverse=True)
                if len(strahler_values) >= 2:
                    if strahler_values[0] == strahler_values[1]:
                        return strahler_values[0] + 1
                    else:
                        return strahler_values[0]
                elif len(strahler_values) == 1:
                    return strahler_values[0]
                else:
                    return 1
            
            # Update Strahler numbers
            root_cluster = raw_tree['parent'].max()
            for cluster_id in metadata:
                try:
                    metadata[cluster_id]['strahler'] = compute_strahler(cluster_id, raw_tree)
                except:
                    metadata[cluster_id]['strahler'] = 1
            
            print(f"Computed metadata for {len(metadata)} clusters")
            return metadata
            
        except Exception as e:
            print(f"Error computing cluster metadata: {e}")
            return {}
    
    def _load_cluster_labels(self, label_file: Path) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
        """
        Load cluster labels and representative words from cluster_to_label.csv
        
        Expected CSV format:
        cluster_id,representative_label,word1,word2,word3,...,word10
        """
        cluster_names: Dict[int, str] = {}
        cluster_words: Dict[int, List[str]] = {}
        
        if not label_file.exists():
            print(f"⚠️  Cluster label file not found: {label_file}")
            return cluster_names, cluster_words
        
        try:
            import csv
            with open(label_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cluster_id = int(row.get('cluster_id', -1))
                    if cluster_id < 0:
                        continue
                    
                    # Get representative label
                    label = row.get('representative_label', f'Cluster {cluster_id}')
                    cluster_names[cluster_id] = label
                    
                    # Collect word columns (word1, word2, ..., word10)
                    words = []
                    for i in range(1, 11):  # word1 to word10
                        word_key = f'word{i}'
                        if word_key in row and row[word_key]:
                            words.append(row[word_key].strip())
                    
                    cluster_words[cluster_id] = words
            
            print(f"✓ Loaded labels for {len(cluster_names)} clusters")
        except Exception as e:
            print(f"Warning: Error loading cluster labels: {e}")
        
        return cluster_names, cluster_words

