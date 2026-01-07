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
                "data_path": "../data",  # 実データの配置先（base_path配下）
                "point_count": 100000,
                "cluster_count": 0,  # Will be determined from data
                "dr_methods": ["umap", "tsne", "pca"]
            }
        }
        # Enable mock data mode when files don't exist
        self.use_mock_data = False  # ← 実データを読み込むモードに変更
    
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
    
    def get_initial_data(
        self,
        dataset: str,
        dr_method: str,
        dr_params: Optional[Dict[str, Any]] = None,
        color_mode: str = 'cluster'
    ) -> Dict[str, Any]:
        """
        Load and process initial data for visualization
        
        Args:
            dataset: Dataset ID
            dr_method: Dimensionality reduction method
            dr_params: Optional parameters for DR method
            color_mode: Color assignment mode
                - 'cluster': Default cluster coloring
                - 'distance': Distance-based coloring using similarity-to-HSV embedding
        
        Returns JSON-compatible data structure with:
        - points: Array of point objects
        - zMatrix: Linkage matrix for dendrogram
        - clusterMeta: Cluster metadata
        - clusterNames: Cluster labels
        - clusterWords: Representative words
        """
        
        if dataset not in self.datasets_config:
            raise ValueError(f"Dataset '{dataset}' not found")
        
        # Check if already cached
        cache_key = f"{dataset}_{dr_method}"
        if cache_key in self.cached_data:
            return self.cached_data[cache_key]
        
        config = self.datasets_config[dataset]
        data_path = self.base_path / config["data_path"]
        print(f"Loading data from: {data_path}")
        
        # Define required data files
        projection_file = data_path / "projection.npy"
        word_file = data_path / "word.npy"
        point_cluster_file = data_path / "point_cluster_map.npy"
        label_file = data_path / "hdbscan_label.npy"  # For noise detection only (-1 = noise)
        
        # ============================================================
        # If data files don't exist, use mock data for development
        # ============================================================
        if not projection_file.exists() or self.use_mock_data:
            print("⚠️  Data files not found or mock data mode enabled, generating mock data")
            result = generate_mock_data(dataset, dr_method, config)
            # Cache for other methods
            cache_data = get_mock_cache_data(result)
            for key, value in cache_data.items():
                setattr(self, key, value)
            # Cache the result
            cache_key = f"{dataset}_{dr_method}"
            self.cached_data[cache_key] = result
            return result
        else: 
            print("✓ Data files found, loading real data")
        
        try:
            # Load 2D projection coordinates
            embedding = np.load(projection_file)  # (N, 2)
            # Load word labels
            labels = np.load(word_file)  # (N,)
            # Load point-to-cluster mapping (contains original cluster IDs: 115760+)
            point_cluster_labels = np.load(point_cluster_file)  # (N,)
            # Load HDBSCAN labels for noise detection only
            hdbscan_labels = np.load(label_file)  # (N,) - -1 for noise, >=0 for clusters
            point_count = len(embedding)
            
            print(f"✓ Loaded projection: {embedding.shape}")
            print(f"✓ Loaded words: {labels.shape}")
            print(f"✓ Loaded point-to-cluster mapping: {point_cluster_labels.shape}")
            print(f"✓ Loaded HDBSCAN labels: {hdbscan_labels.shape}")
        except Exception as e:
            raise ValueError(f"Error loading data files: {e}")
        
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
            print(f"⚠️  No condensed_tree_object.pkl found, creating simplified dendrogram")
            # Create simplified linkage matrix from cluster labels
            linkage_matrix = self._create_simplified_linkage_from_labels(cluster_labels)
        
        # Load cluster metadata
        if hdbscan_condensed_tree is not None:
            cluster_meta = self._compute_cluster_metadata(hdbscan_condensed_tree)
        else:
            cluster_meta = self._compute_cluster_metadata_from_labels(cluster_labels)

        # Cache raw arrays for downstream queries (e.g., point detail/NN search)
        self._embedding2d = embedding
        self._labels = labels
        self._cluster_labels = point_cluster_labels
        self._hdbscan_labels = hdbscan_labels  # For noise detection
        
        # Load cluster labels and words
        cluster_label_file = data_path / "cluster_to_label.csv"
        cluster_names, cluster_words = self._load_cluster_labels(cluster_label_file)
        
        print(f"DEBUG: Loaded cluster_names: {len(cluster_names)} items")
        if cluster_names:
            sample = list(cluster_names.items())[:5]
            print(f"DEBUG: Sample cluster_names: {sample}")
        
        print(f"DEBUG: Loaded cluster_words: {len(cluster_words)} items")
        if cluster_words:
            sample = list(cluster_words.items())[:5]
            print(f"DEBUG: Sample cluster_words keys: {[k for k, v in sample]}")
        
        # point_cluster_labels already contains original cluster IDs (115760+)
        # Build point_cluster_map from loaded data
        point_cluster_map = {}
        for i in range(point_count):
            point_cluster_map[i] = int(point_cluster_labels[i])
        
        print(f"DEBUG: point_cluster_map size: {len(point_cluster_map)}")
        print(f"DEBUG: Sample mappings: {list(point_cluster_map.items())[:10]}")
        
        # Compute cluster colors based on color_mode
        cluster_colors = None
        if color_mode == 'distance' and self._similarity_dict is not None:
            try:
                metric = 'mahalanobis_distance'
                if metric in self._similarity_dict:
                    cluster_colors = compute_cluster_colors_from_similarity(
                        self._similarity_dict[metric],
                        scaling_type='robust',
                        value_range=(0.5, 1)  # Avoid very dark colors
                    )
                else:
                    print(f"⚠️  Metric '{metric}' not found in similarity dict")
            except Exception as e:
                import traceback
                print(f"⚠️  Error computing distance-based colors: {e}")
                print(traceback.format_exc())
                cluster_colors = None
        
        # Format points for JSON
        points = []
        for i in range(point_count):
            # Check if point is noise using hdbscan_labels (-1 = noise)
            cluster_id = -1 if hdbscan_labels[i] == -1 else int(point_cluster_labels[i])
            
            point = {
                "i": i,                              # Index
                "x": float(embedding[i, 0]),
                "y": float(embedding[i, 1]),
                "c": cluster_id,                     # Cluster ID (-1 for noise, or 115760+ for valid cluster)
                "l": str(labels[i])                  # Label
            }
            
            # Add color if available
            if cluster_colors and cluster_id in cluster_colors:
                point["color"] = cluster_colors[cluster_id]
            
            points.append(point)
        
        # Convert linkage matrix (c1, c2, parent, dist, size) to JSON-compatible format
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
        
        # Convert cluster metadata to JSON
        cluster_meta_json = {
            str(cluster_id): {
                "s": float(meta.get("stability", 0)),
                "h": int(meta.get("strahler", 0)),
                "z": int(meta.get("size", 0))
            }
            for cluster_id, meta in cluster_meta.items()
        }
        
        # Cache frequently used data for other methods
        self._points = points
        self._linkage_matrix_array = linkage_matrix
        self._z_json = z_matrix
        self._cluster_metadata = cluster_meta
        self._cluster_names = cluster_names
        self._cluster_words = cluster_words
        self._point_cluster_map = point_cluster_map

        # Create clusterIdMap: maps dendrogram sequential indices to original cluster IDs
        # old_new_id_map is {original_cluster_id: sequential_index}
        # We need the reverse: {sequential_index: original_cluster_id}
        clusterIdMap = {v: k for k, v in old_new_id_map.items()} if old_new_id_map else {}
        print(f"DEBUG: old_new_id_map size: {len(old_new_id_map)}")
        print(f"DEBUG: clusterIdMap (reversed) size: {len(clusterIdMap)}")
        if clusterIdMap:
            sample = list(clusterIdMap.items())[:5]
            print(f"DEBUG: Sample clusterIdMap entries: {sample}")
        
        # Extract cluster similarities for dendrogram sorting from already-loaded similarity dictionary
        cluster_similarities = None
        if self._similarity_dict and 'mahalanobis_distance' in self._similarity_dict:
            try:
                metric = 'mahalanobis_distance'
                similarity_dict = self._similarity_dict[metric]
                # Convert (cluster_id, cluster_id) -> distance dict to [[id1, id2, distance], ...] format
                cluster_similarities = [
                    [id1, id2, dist]
                    for (id1, id2), dist in similarity_dict.items()
                ]
                print(f"✓ Extracted cluster similarities: {len(cluster_similarities)} pairs from '{metric}'")
            except Exception as e:
                print(f"⚠️  Could not extract cluster similarities: {e}")
        
        result = {
            "points": points,
            "zMatrix": z_matrix,
            "clusterMeta": cluster_meta_json,
            "clusterNames": cluster_names,
            "clusterWords": cluster_words,
            "clusterIdMap": clusterIdMap,  # Map dendrogram sequential index -> original cluster ID
            "clusterSimilarities": cluster_similarities,  # [[id1, id2, distance], ...] for sorting
            "datasetInfo": {
                "name": config["name"],
                "pointCount": point_count,
                "clusterCount": len(cluster_names),
                "description": config["description"]
            }
        }
        
        # Cache the result
        self.cached_data[cache_key] = result
        return result
    
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
        
        filtered_metadata = {
            cid: meta for cid, meta in full_data["clusterMeta"].items()
            if int(cid) in non_noise_cluster_ids
        }
        
        # Update point count in datasetInfo
        result = {
            "points": new_points,
            "zMatrix": full_data["zMatrix"],  # Keep unchanged
            "clusterMeta": filtered_metadata,
            "clusterNames": {
                k: v for k, v in full_data["clusterNames"].items()
                if int(k) in non_noise_cluster_ids
            },
            "clusterWords": {
                k: v for k, v in full_data["clusterWords"].items()
                if int(k) in non_noise_cluster_ids
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
