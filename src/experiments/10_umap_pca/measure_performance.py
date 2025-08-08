"""
重み付きPCAの計算処理の実行時間を計測するスクリプト
主要な処理部分の実行時間を個別に測定します
"""

import os
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_mnist_data(sample_size=5000):
    """MNISTデータをロード"""
    print("MNISTデータをロード中...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    if sample_size and sample_size < len(mnist.data):
        # サンプルサイズが指定されている場合はランダムサンプリング
        indices = np.random.choice(len(mnist.data), sample_size, replace=False)
        data = mnist.data[indices]
        labels = mnist.target[indices]
    else:
        data = mnist.data
        labels = mnist.target
    
    print(f"データロード完了: {data.shape}")
    return data, labels


def generate_weighted_pca_original(data, points_indices, scaler, weights_exp=4.0):
    """オリジナルの重み付きPCA計算関数 - 計測用"""
    # 標準化データを取得
    start_time = time.time()
    scaled_data = scaler.transform(data)
    scaling_time = time.time() - start_time
    
    # 重みベクトルを生成（選択された点は重み1.0、その他は小さい値）
    start_time = time.time()
    weights = np.ones(len(data)) * 0.01  # 基本重みは0.01
    weights[points_indices] = 1.0  # 選択された点の重みは1.0
    
    # 重みを指数関数的に適用（コントラストを高める）
    if weights_exp != 1.0:
        weights = weights ** weights_exp
    weights_time = time.time() - start_time
    
    # 中心化
    start_time = time.time()
    weighted_mean = np.average(scaled_data, axis=0, weights=weights)
    centered_data = scaled_data - weighted_mean
    centering_time = time.time() - start_time
    
    # 重み付き共分散行列を計算
    start_time = time.time()
    weighted_cov = np.zeros((scaled_data.shape[1], scaled_data.shape[1]))
    total_weight = weights.sum()
    
    # 重み付き共分散行列 - ループ処理（最も時間がかかる部分）
    for i in range(len(data)):
        x = centered_data[i].reshape(-1, 1)
        weighted_cov += weights[i] * (x @ x.T)
    weighted_cov /= total_weight
    cov_time = time.time() - start_time
    
    # 固有値分解
    start_time = time.time()
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    
    # 固有値を降順にソート
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 固有ベクトルを重み付きPCAの主成分とする
    weighted_pca_components = eigvecs
    eig_time = time.time() - start_time
    
    # 結果を投影
    start_time = time.time()
    projection = centered_data @ weighted_pca_components[:, :2]
    projection_time = time.time() - start_time
    
    # 各処理の実行時間を含めて結果を返す
    timing_info = {
        'scaling_time': scaling_time,
        'weights_time': weights_time,
        'centering_time': centering_time,
        'cov_time': cov_time,
        'eig_time': eig_time,
        'projection_time': projection_time,
        'total_time': scaling_time + weights_time + centering_time + cov_time + eig_time + projection_time
    }
    
    return projection, weighted_pca_components, eigvals, weighted_mean, timing_info


def generate_weighted_pca_optimized(data, points_indices, scaler, weights_exp=4.0):
    """最適化版の重み付きPCA計算関数 - ベクトル化による高速化"""
    # 標準化データを取得
    start_time = time.time()
    scaled_data = scaler.transform(data)
    scaling_time = time.time() - start_time
    
    # 重みベクトルを生成（選択された点は重み1.0、その他は小さい値）
    start_time = time.time()
    weights = np.ones(len(data)) * 0.01  # 基本重みは0.01
    weights[points_indices] = 1.0  # 選択された点の重みは1.0
    
    # 重みを指数関数的に適用（コントラストを高める）
    if weights_exp != 1.0:
        weights = weights ** weights_exp
    weights_time = time.time() - start_time
    
    # 中心化
    start_time = time.time()
    weighted_mean = np.average(scaled_data, axis=0, weights=weights)
    centered_data = scaled_data - weighted_mean
    centering_time = time.time() - start_time
    
    # 重み付き共分散行列を計算 - ベクトル化バージョン
    start_time = time.time()
    # 重みの平方根を計算
    sqrt_weights = np.sqrt(weights).reshape(-1, 1)
    
    # 重み付きデータを計算
    weighted_data = centered_data * sqrt_weights
    
    # 共分散行列を一度の行列計算で算出
    weighted_cov = weighted_data.T @ weighted_data
    weighted_cov /= weights.sum()
    cov_time = time.time() - start_time
    
    # 固有値分解
    start_time = time.time()
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    
    # 固有値を降順にソート
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 固有ベクトルを重み付きPCAの主成分とする
    weighted_pca_components = eigvecs
    eig_time = time.time() - start_time
    
    # 結果を投影
    start_time = time.time()
    projection = centered_data @ weighted_pca_components[:, :2]
    projection_time = time.time() - start_time
    
    # 各処理の実行時間を含めて結果を返す
    timing_info = {
        'scaling_time': scaling_time,
        'weights_time': weights_time,
        'centering_time': centering_time,
        'cov_time': cov_time,
        'eig_time': eig_time,
        'projection_time': projection_time,
        'total_time': scaling_time + weights_time + centering_time + cov_time + eig_time + projection_time
    }
    
    return projection, weighted_pca_components, eigvals, weighted_mean, timing_info


def create_weight_animation_frames(data, selected_indices, scaler, method="original", num_frames=30, max_weight_exp=4.0):
    """重みのアニメーションフレームを生成し、実行時間を計測"""
    weight_exps = np.linspace(1.0, max_weight_exp, num_frames)
    frames = []
    
    total_start_time = time.time()
    print(f"アニメーションフレームを計算中... ({method})")
    
    # 各フレームの計算時間を記録
    frame_times = []
    cov_matrix_times = []
    eig_decomp_times = []
    
    for weight_exp in tqdm(weight_exps, desc="フレーム生成"):
        frame_start = time.time()
        
        if method == "original":
            projection, components, eigvals, _, timing_info = generate_weighted_pca_original(
                data, selected_indices, scaler, weight_exp)
        else:  # optimized
            projection, components, eigvals, _, timing_info = generate_weighted_pca_optimized(
                data, selected_indices, scaler, weight_exp)
        
        frames.append({
            'projection': projection,
            'components': components[:2],  # 最初の2つの主成分を保存
            'eigenvalues': eigvals[:2].tolist()  # 固有値も保存
        })
        
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        cov_matrix_times.append(timing_info['cov_time'])
        eig_decomp_times.append(timing_info['eig_time'])
    
    total_time = time.time() - total_start_time
    
    timing_stats = {
        'total_time': total_time,
        'avg_frame_time': np.mean(frame_times),
        'max_frame_time': np.max(frame_times),
        'min_frame_time': np.min(frame_times),
        'avg_cov_time': np.mean(cov_matrix_times),
        'avg_eig_time': np.mean(eig_decomp_times),
        'frames_per_sec': num_frames / total_time
    }
    
    print(f"アニメーションフレーム計算完了！時間: {total_time:.2f}秒")
    print(f"フレームあたり平均時間: {timing_stats['avg_frame_time']:.4f}秒")
    print(f"共分散行列計算平均時間: {timing_stats['avg_cov_time']:.4f}秒")
    print(f"固有値分解平均時間: {timing_stats['avg_eig_time']:.4f}秒")
    
    return frames, timing_stats


def run_benchmark(sample_size=5000, num_selected=100, num_frames=30):
    """ベンチマークを実行し、オリジナル版と最適化版の実行時間を比較"""
    # データのロードと前処理
    data, labels = load_mnist_data(sample_size=sample_size)
    
    # 標準化
    scaler = StandardScaler()
    _ = scaler.fit_transform(data)
    
    # ランダムに点を選択
    selected_indices = np.random.choice(len(data), num_selected, replace=False)
    
    print(f"\n==== ベンチマーク開始: サンプル {sample_size}, 選択点 {num_selected}, フレーム数 {num_frames} ====\n")
    
    # オリジナルの実装でフレーム生成
    print("\n--- オリジナル実装 ---")
    _, original_stats = create_weight_animation_frames(
        data, selected_indices, scaler, method="original", num_frames=num_frames)
    
    # 最適化版の実装でフレーム生成
    print("\n--- 最適化実装 ---")
    _, optimized_stats = create_weight_animation_frames(
        data, selected_indices, scaler, method="optimized", num_frames=num_frames)
    
    # 結果の比較
    speedup = original_stats['total_time'] / optimized_stats['total_time']
    cov_speedup = original_stats['avg_cov_time'] / optimized_stats['avg_cov_time']
    
    print("\n==== 結果比較 ====")
    print(f"オリジナル実装の合計時間: {original_stats['total_time']:.2f}秒")
    print(f"最適化実装の合計時間: {optimized_stats['total_time']:.2f}秒")
    print(f"高速化倍率: {speedup:.2f}倍")
    print(f"共分散行列計算の高速化倍率: {cov_speedup:.2f}倍")
    
    # 詳細なタイミング情報
    print("\n==== 詳細タイミング ====")
    print(f"{'処理':20s} {'オリジナル (秒)':15s} {'最適化版 (秒)':15s} {'高速化倍率':10s}")
    print("-" * 60)
    print(f"{'フレーム平均':20s} {original_stats['avg_frame_time']:.4f}{'':<10s} {optimized_stats['avg_frame_time']:.4f}{'':<10s} {original_stats['avg_frame_time'] / optimized_stats['avg_frame_time']:.2f}倍")
    print(f"{'共分散行列計算':20s} {original_stats['avg_cov_time']:.4f}{'':<10s} {optimized_stats['avg_cov_time']:.4f}{'':<10s} {original_stats['avg_cov_time'] / optimized_stats['avg_cov_time']:.2f}倍")
    print(f"{'固有値分解':20s} {original_stats['avg_eig_time']:.4f}{'':<10s} {optimized_stats['avg_eig_time']:.4f}{'':<10s} {original_stats['avg_eig_time'] / optimized_stats['avg_eig_time']:.2f}倍")
    
    return {
        'original': original_stats,
        'optimized': optimized_stats,
        'speedup': speedup
    }


def run_scaling_benchmark():
    """データサイズを変えて実行時間の変化を確認するベンチマーク"""
    sample_sizes = [1000, 2000, 5000, 10000]
    results = {}
    
    for size in sample_sizes:
        print(f"\n\n====== サンプルサイズ: {size} ======\n")
        results[size] = run_benchmark(sample_size=size, num_selected=100, num_frames=10)
    
    print("\n====== スケーリング結果まとめ ======")
    print(f"{'サンプルサイズ':15s} {'オリジナル時間 (秒)':20s} {'最適化時間 (秒)':20s} {'高速化倍率':10s}")
    print("-" * 70)
    
    for size, result in results.items():
        print(f"{size:<15d} {result['original']['total_time']:<20.2f} {result['optimized']['total_time']:<20.2f} {result['speedup']:<10.2f}")


if __name__ == "__main__":
    # 標準的なベンチマークを実行
    run_benchmark(sample_size=5000, num_selected=100, num_frames=30)
    
    # コメントアウトを外すとスケーリングベンチマークを実行
    # run_scaling_benchmark()
