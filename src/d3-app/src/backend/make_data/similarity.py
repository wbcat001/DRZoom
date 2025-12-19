"""
Cluster Similarity Calculator

クラスタ間の類似度を計算して保存するスクリプト
HDBSCANの結果からクラスタ間の類似度行列を生成

Usage:
    python similarity.py --data_dir <path> --output_dir <path>
    
Required files in data_dir:
    - vector.npy: (N, 300) high-dimensional vectors
    - hdbscan_label.npy: (N,) cluster labels

Example:
    cd d:\Work_Program\DRZoom\src\d3-app\src\backend\make_data
    
    # 基本的な使用
    python similarity.py --data_dir ../../../../experiments/18_rapids/result/20251203_053328 --output_dir ../../../../experiments/18_rapids/result/20251203_053328
    
    # メトリックを指定
    python similarity.py --data_dir path/to/data --output_dir path/to/output --metrics kl_divergence bhattacharyya_coefficient
    
    # 最小クラスタサイズを指定
    python similarity.py --data_dir path/to/data --output_dir path/to/output --min_cluster_size 20
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
from scipy.linalg import inv, det
from sklearn.covariance import LedoitWolf


# ============================================================
# 1. 分布のパラメータ推定
# ============================================================

def estimate_single_gaussian_params(X_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    単一の高次元データセットから、多変量正規分布の平均ベクトルと共分散行列を推定
    
    サンプル数が次元数より少ない場合、Ledoit-Wolf収縮法を用いて
    共分散行列を頑健に推定する
    
    Args:
        X_data: 単一のクラスタに属するデータ (N, D)
        
    Returns:
        {'mu': 平均ベクトル, 'Sigma': 共分散行列}
    """
    N, D = X_data.shape
    
    if N == 0:
        raise ValueError("Input data array must not be empty.")
    
    # 平均ベクトルの推定
    mu = np.mean(X_data, axis=0)
    
    # 共分散行列の推定
    if N == 1:
        warnings.warn("N=1. Covariance matrix is set to zero (plus regularization).")
        Sigma = np.eye(D) * 1e-6
        
    elif N < D + 1:
        # 特異行列になるリスクが高いため、Ledoit-Wolf収縮推定を使用
        warnings.warn(f"N={N} < D+1={D+1}. Using Ledoit-Wolf shrinkage.")
        lw = LedoitWolf()
        lw.fit(X_data)
        Sigma = lw.covariance_
        
    else:
        # 標準的な最尤推定
        Sigma = np.cov(X_data, rowvar=False)
    
    # 正則化チェック
    if np.linalg.cond(Sigma) > 1e15:
        warnings.warn("Covariance matrix highly ill-conditioned. Applying regularization.")
        Sigma += np.eye(D) * 1e-6
    
    return {'mu': mu, 'Sigma': Sigma}


def estimate_gaussian_params_for_clusters(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_ids: List[int]
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    各クラスタのガウス分布パラメータを推定
    
    Args:
        X: 全データ (N, D)
        cluster_labels: 各点のクラスタラベル (N,)
        cluster_ids: 対象クラスタIDのリスト
        
    Returns:
        {cluster_id: {'mu': 平均, 'Sigma': 共分散}}
    """
    params = {}
    
    for cid in cluster_ids:
        mask = cluster_labels == cid
        X_cluster = X[mask]
        
        if len(X_cluster) == 0:
            warnings.warn(f"Cluster {cid} has no points. Skipping.")
            continue
        
        params[cid] = estimate_single_gaussian_params(X_cluster)
    
    return params


# ============================================================
# 2. 類似度測定（KL情報量）
# ============================================================

def kl_divergence_gaussian(mu1, Sigma1, mu2, Sigma2) -> float:
    """
    多変量ガウス分布 N1 から N2 へのKL情報量 D_KL(N1 || N2)
    
    Args:
        mu1, Sigma1: 分布1の平均と共分散
        mu2, Sigma2: 分布2の平均と共分散
        
    Returns:
        KL divergence (非類似度、値が大きいほど非類似)
    """
    D = mu1.shape[0]
    
    try:
        Sigma2_inv = inv(Sigma2)
    except np.linalg.LinAlgError:
        warnings.warn("Sigma2 is singular. KL divergence is undefined.")
        return np.nan
    
    # ログデターミナント項
    log_det_term = np.log(det(Sigma2) / det(Sigma1))
    
    # トレース項
    trace_term = np.trace(Sigma2_inv @ Sigma1)
    
    # マハラノビス距離項
    diff_mu = mu2 - mu1
    mahalanobis_term = diff_mu.T @ Sigma2_inv @ diff_mu
    
    kl_div = 0.5 * (log_det_term + trace_term + mahalanobis_term - D)
    
    return kl_div


def symmetric_kl_divergence(mu1, Sigma1, mu2, Sigma2) -> float:
    """
    対称化KL情報量 = 0.5 * (D_KL(N1||N2) + D_KL(N2||N1))
    """
    kl_12 = kl_divergence_gaussian(mu1, Sigma1, mu2, Sigma2)
    kl_21 = kl_divergence_gaussian(mu2, Sigma2, mu1, Sigma1)
    
    if np.isnan(kl_12) or np.isnan(kl_21):
        return np.nan
    
    return 0.5 * (kl_12 + kl_21)


# ============================================================
# 3. 類似度測定（バタチャリヤ係数）
# ============================================================

def bhattacharyya_coefficient_gaussian(mu1, Sigma1, mu2, Sigma2) -> float:
    """
    多変量ガウス分布間のバタチャリヤ係数 BC
    
    Args:
        mu1, Sigma1: 分布1の平均と共分散
        mu2, Sigma2: 分布2の平均と共分散
        
    Returns:
        Bhattacharyya Coefficient (重なり、1に近いほど類似)
    """
    D = mu1.shape[0]
    
    # 共分散行列の平均
    Sigma = 0.5 * (Sigma1 + Sigma2)
    
    try:
        Sigma_inv = inv(Sigma)
    except np.linalg.LinAlgError:
        warnings.warn("Sigma (mean covariance) is singular.")
        return np.nan
    
    # バタチャリヤ距離の平均項
    diff_mu = mu1 - mu2
    db_mu_term = 0.125 * diff_mu.T @ Sigma_inv @ diff_mu
    
    # バタチャリヤ距離の共分散項
    db_cov_term = 0.5 * np.log(det(Sigma) / np.sqrt(det(Sigma1) * det(Sigma2)))
    
    # バタチャリヤ距離
    db_distance = db_mu_term + db_cov_term
    
    # バタチャリヤ係数
    bc = np.exp(-db_distance)
    
    return bc


# ============================================================
# 4. マハラノビス距離
# ============================================================

def mahalanobis_distance(mu1, mu2, Sigma_pooled) -> float:
    """
    プールされた共分散行列を用いた2つの平均間のマハラノビス距離
    
    Args:
        mu1, mu2: 平均ベクトル
        Sigma_pooled: プールされた共分散行列
        
    Returns:
        Mahalanobis distance
    """
    try:
        Sigma_inv = inv(Sigma_pooled)
    except np.linalg.LinAlgError:
        warnings.warn("Sigma_pooled is singular.")
        return np.nan
    
    diff_mu = mu1 - mu2
    dist_sq = diff_mu.T @ Sigma_inv @ diff_mu
    
    return np.sqrt(dist_sq)


# ============================================================
# 5. クラスタ間類似度行列の計算
# ============================================================

def compute_similarity_matrix(
    params: Dict[int, Dict[str, np.ndarray]],
    cluster_ids: List[int],
    metric: str = "kl_divergence"
) -> np.ndarray:
    """
    クラスタ間の類似度行列を計算
    
    Args:
        params: 各クラスタのガウス分布パラメータ
        cluster_ids: クラスタIDのリスト
        metric: 類似度メトリック
            - "kl_divergence": 対称化KL情報量（非類似度）
            - "bhattacharyya_coefficient": バタチャリヤ係数（類似度）
            - "mahalanobis_distance": マハラノビス距離（非類似度）
            
    Returns:
        類似度行列 (n_clusters, n_clusters)
    """
    n = len(cluster_ids)
    similarity_matrix = np.zeros((n, n))
    
    for i, cid1 in enumerate(cluster_ids):
        for j, cid2 in enumerate(cluster_ids):
            if cid1 not in params or cid2 not in params:
                similarity_matrix[i, j] = np.nan
                continue
            
            mu1 = params[cid1]['mu']
            Sigma1 = params[cid1]['Sigma']
            mu2 = params[cid2]['mu']
            Sigma2 = params[cid2]['Sigma']
            
            if i == j:
                # 自己類似度
                if metric == "bhattacharyya_coefficient":
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = 0.0
            else:
                if metric == "kl_divergence":
                    similarity_matrix[i, j] = symmetric_kl_divergence(mu1, Sigma1, mu2, Sigma2)
                elif metric == "bhattacharyya_coefficient":
                    similarity_matrix[i, j] = bhattacharyya_coefficient_gaussian(
                        mu1, Sigma1, mu2, Sigma2
                    )
                elif metric == "mahalanobis_distance":
                    Sigma_pooled = 0.5 * (Sigma1 + Sigma2)
                    similarity_matrix[i, j] = mahalanobis_distance(mu1, mu2, Sigma_pooled)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
    
    return similarity_matrix


# ============================================================
# 6. メイン処理
# ============================================================

def load_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    データファイルを読み込み
    
    Args:
        data_dir: データディレクトリへのパス
        
    Returns:
        (embedding, cluster_labels)
    """
    # 高次元ベクトルを読み込み (N, 300)
    vector_file = data_dir / "vector.npy"
    if not vector_file.exists():
        raise FileNotFoundError(f"vector.npy not found in {data_dir}")
    
    embedding = np.load(vector_file)
    print(f"✓ Loaded vectors: {embedding.shape}")
    
    # クラスタラベルを読み込み (N,)
    label_file = data_dir / "hdbscan_label.npy"
    if not label_file.exists():
        raise FileNotFoundError(f"hdbscan_label.npy not found in {data_dir}")
    
    labels = np.load(label_file)
    print(f"✓ Loaded cluster labels: {labels.shape}")
    
    return embedding, labels


def save_similarity_matrices(
    output_dir: Path,
    similarity_matrices: Dict[str, np.ndarray],
    cluster_ids: List[int]
):
    """
    類似度行列をpickle形式で保存
    
    Args:
        output_dir: 出力ディレクトリ
        similarity_matrices: {metric_name: similarity_matrix}
        cluster_ids: クラスタIDのリスト
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for metric, matrix in similarity_matrices.items():
        output_file = output_dir / f"similarity_{metric}.pkl"
        
        data = {
            'matrix': matrix,
            'cluster_ids': cluster_ids,
            'metric': metric
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Saved {metric} similarity matrix to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute cluster similarity matrices")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory containing vector.npy and hdbscan_label.npy')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for similarity matrices')
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=['kl_divergence', 'bhattacharyya_coefficient', 'mahalanobis_distance'],
                       help='Similarity metrics to compute')
    parser.add_argument('--min_cluster_size', type=int, default=10,
                       help='Minimum cluster size to include')
    
    args = parser.parse_args()
    
    # データ読み込み
    print(f"Loading data from {args.data_dir}...")
    data_dir = Path(args.data_dir)
    embedding, labels = load_data(data_dir)
    
    print(f"Data shape: {embedding.shape}")
    print(f"Number of points: {len(labels)}")
    
    # クラスタIDの抽出（ノイズ=-1を除外）
    unique_labels = np.unique(labels)
    cluster_ids = [int(cid) for cid in unique_labels if cid >= 0]
    
    # 最小サイズでフィルタ
    filtered_cluster_ids = []
    for cid in cluster_ids:
        cluster_size = np.sum(labels == cid)
        if cluster_size >= args.min_cluster_size:
            filtered_cluster_ids.append(cid)
        else:
            print(f"  Skipping cluster {cid} (size={cluster_size} < {args.min_cluster_size})")
    
    cluster_ids = filtered_cluster_ids
    print(f"Number of clusters: {len(cluster_ids)}")
    
    # ガウス分布パラメータの推定
    print("\nEstimating Gaussian parameters for each cluster...")
    params = estimate_gaussian_params_for_clusters(embedding, labels, cluster_ids)
    
    # 類似度行列の計算
    print("\nComputing similarity matrices...")
    similarity_matrices = {}
    
    for metric in args.metrics:
        print(f"  Computing {metric}...")
        try:
            matrix = compute_similarity_matrix(params, cluster_ids, metric)
            similarity_matrices[metric] = matrix
            
            # 統計情報
            valid_values = matrix[~np.isnan(matrix)]
            if len(valid_values) > 0:
                print(f"    Min: {np.min(valid_values):.4f}, "
                      f"Max: {np.max(valid_values):.4f}, "
                      f"Mean: {np.mean(valid_values):.4f}")
        except Exception as e:
            print(f"    Error computing {metric}: {e}")
    
    # 保存
    output_dir = Path(args.output_dir)
    save_similarity_matrices(output_dir, similarity_matrices, cluster_ids)
    
    print(f"\n✓ All similarity matrices saved to {output_dir}")


if __name__ == "__main__":
    main()
