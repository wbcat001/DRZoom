"""
Color Space Embedding - Converts cluster similarity to RGB colors via HSV mapping
Implements the color space embedding pipeline from color_by_similarity.ipynb
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.manifold import MDS
from scipy.stats import scoreatpercentile
from matplotlib.colors import hsv_to_rgb
import warnings


def create_similarity_matrix_from_dict(similarity_dict: Dict[Tuple[int, int], float]) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Convert (cluster_id, cluster_id) -> similarity dictionary into N×N matrix
    
    Args:
        similarity_dict: Dict with (id1, id2) as keys and similarity as values
    
    Returns:
        tuple: (similarity_matrix, id_to_index mapping)
    """
    cluster_ids = set()
    for id1, id2 in similarity_dict.keys():
        cluster_ids.add(id1)
        cluster_ids.add(id2)
    
    sorted_ids = sorted(list(cluster_ids))
    N = len(sorted_ids)
    
    id_to_index = {id: i for i, id in enumerate(sorted_ids)}
    similarity_matrix = np.zeros((N, N), dtype=float)
    
    for (id1, id2), similarity in similarity_dict.items():
        i = id_to_index.get(id1)
        j = id_to_index.get(id2)
        
        if i is None or j is None:
            continue
        
        similarity_matrix[i, j] = similarity
        if i != j:
            similarity_matrix[j, i] = similarity
    
    np.fill_diagonal(similarity_matrix, 1.0)
    
    return similarity_matrix, id_to_index


def apply_mds_projection(
    similarity_matrix: np.ndarray,
    n_components: int = 3
) -> Tuple[np.ndarray, float]:
    """
    Convert similarity matrix to distance and apply MDS projection
    
    Args:
        similarity_matrix: N×N similarity matrix (0=dissimilar, 1=similar)
        n_components: Number of dimensions for projection
    
    Returns:
        tuple: (3D coordinates, MDS stress value)
    """
    distance_matrix = 1 - similarity_matrix
    mds = MDS(
        n_components=n_components,
        dissimilarity='precomputed',
        random_state=42,
        normalized_stress='auto'
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = mds.fit_transform(distance_matrix)
    
    return coords, mds.stress_


def apply_scaling(
    coords: np.ndarray,
    scaling_type: str = 'robust'
) -> np.ndarray:
    """
    Apply scaling to coordinates (normalize to [0, 1])
    
    Args:
        coords: 1D array of coordinates
        scaling_type: 'linear' or 'robust' (P5-P95 based)
    
    Returns:
        Scaled coordinates in [0, 1] range
    """
    if scaling_type == 'linear':
        min_val = coords.min()
        max_val = coords.max()
        return (coords - min_val) / (max_val - min_val)
    
    elif scaling_type == 'robust':
        p5 = scoreatpercentile(coords, 5)
        p95 = scoreatpercentile(coords, 95)
        scaled = (coords - p5) / (p95 - p5)
        return np.clip(scaled, 0, 1)
    
    else:
        raise ValueError("scaling_type must be 'linear' or 'robust'")


def map_coords_to_hsv(
    coords_3d: np.ndarray,
    scaling_type: str = 'robust',
    hsv_mapping: Tuple[str, str, str] = ('H', 'S', 'V'),
    value_range: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map 3D MDS coordinates to HSV color space
    
    Args:
        coords_3d: (N, 3) 3D coordinates from MDS
        scaling_type: 'linear' or 'robust' scaling
        hsv_mapping: Tuple specifying which MDS axis maps to (H, S, V)
        value_range: Tuple (min, max) specifying the range for V (明度)
    
    Returns:
        tuple: (H, S, V) arrays in [0, 1] range
    """
    hsv_inputs = {}
    
    for i, hsv_comp in enumerate(hsv_mapping):
        mds_coord = coords_3d[:, i]
        scaled = apply_scaling(mds_coord, scaling_type)
        hsv_inputs[hsv_comp] = scaled
    
    H = hsv_inputs['H']
    S = hsv_inputs['S']
    V = hsv_inputs['V']
    
    # V (明度) の範囲を調整
    v_min, v_max = value_range
    V = v_min + V * (v_max - v_min)
    
    return H, S, V


def convert_hsv_to_rgb(
    H: np.ndarray,
    S: np.ndarray,
    V: np.ndarray
) -> list:
    """
    Convert HSV arrays to RGB color strings for web frontend
    
    Args:
        H, S, V: Arrays of HSV values in [0, 1]
    
    Returns:
        List of RGB color strings like 'rgb(255, 128, 0)'
    """
    hsv_array = np.stack([H, S, V], axis=1)
    rgb_array = hsv_to_rgb(hsv_array)
    
    hex_colors = [
        f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
        for r, g, b in rgb_array
    ]
    
    return hex_colors


def compute_cluster_colors_from_similarity(
    similarity_dict: Dict[Tuple[int, int], float],
    cluster_ids: Optional[list] = None,
    scaling_type: str = 'robust',
    hsv_mapping: Tuple[str, str, str] = ('H', 'S', 'V'),
    value_range: Tuple[float, float] = (0, 1)
) -> Dict[int, str]:
    """
    メイン統合関数：類似度辞書からクラスタID→RGB色の対応を計算
    
    Args:
        similarity_dict (dict): 
            形式: {(cluster_id1, cluster_id2): similarity_value}
        
        cluster_ids (list, set, or None):
            色を計算対象にするクラスタIDの集合
            None の場合は全クラスタを対象
        
        scaling_type (str): 
            'linear' - 標準的な正規化
            'robust' - P5-P95パーセンタイル基準（外れ値に強い）
        
        hsv_mapping (tuple):
            MDS座標をどのHSV要素に割り当てるか
        
        value_range (tuple):
            V（明度）の出力範囲を指定
            デフォルト: (0, 1) → 全範囲 [0.0, 1.0]
            例: (0.5, 1) → 暗い色を避ける [0.5, 1.0]
    
    Returns:
        dict: {cluster_id: 'rgb(r, g, b)', ...}
    """
    
    # Step 0: cluster_ids が指定されている場合、類似度辞書をフィルタリング
    if cluster_ids is not None:
        cluster_ids_set = set(cluster_ids)
        
        filtered_dict = {
            (c1, c2): sim 
            for (c1, c2), sim in similarity_dict.items()
            if c1 in cluster_ids_set and c2 in cluster_ids_set
        }
        
        if len(filtered_dict) == 0:
            return {cid: 'rgb(200, 200, 200)' for cid in cluster_ids_set}
        
        working_dict = filtered_dict
    else:
        working_dict = similarity_dict
    
    # Step 1: 類似度行列を作成
    similarity_matrix, id_to_index = create_similarity_matrix_from_dict(working_dict)
    
    # Step 2: MDS投影
    coords_3d, stress = apply_mds_projection(similarity_matrix, n_components=3)
    
    # Step 3: HSV空間にマッピング（value_range を指定）
    H, S, V = map_coords_to_hsv(coords_3d, scaling_type, hsv_mapping, value_range)
    
    # Step 4: RGB色に変換
    colors = convert_hsv_to_rgb(H, S, V)
    
    # Step 5: クラスタID → 色の対応を作成
    cluster_id_to_color = {}
    for cluster_id, index in id_to_index.items():
        if index < len(colors):
            cluster_id_to_color[cluster_id] = colors[index]
    
    return cluster_id_to_color
