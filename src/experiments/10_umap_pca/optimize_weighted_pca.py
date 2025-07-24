"""
重み付きPCA計算の最適化方法の検証と比較

異なる最適化アプローチを比較し、詳細な実行時間計測を行います
"""
"""
==== 実装比較 ====
実装              合計時間 (秒)        フレーム平均 (秒)           高速化倍率
-----------------------------------------------------------------
オリジナル                  119.4378              11.9433       1.00倍
ベクトル化                    2.2875               0.2284      52.21倍
バッチ処理                    2.6055               0.2599      45.84倍
Numba                   98.7850               9.8779       1.21倍
"""
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numba


def load_mnist_data(sample_size=5000):
    """MNISTデータをロード"""
    print("MNISTデータをロード中...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    if sample_size and sample_size < len(mnist.data):
        indices = np.random.choice(len(mnist.data), sample_size, replace=False)
        data = mnist.data[indices]
        labels = mnist.target[indices]
    else:
        data = mnist.data
        labels = mnist.target
    
    print(f"データロード完了: {data.shape}")
    return data, labels


def generate_weighted_pca_original(data, points_indices, scaler, weights_exp=4.0):
    """オリジナルの重み付きPCA計算関数"""
    # 標準化データを取得
    start_time = time.time()
    scaled_data = scaler.transform(data)
    scaling_time = time.time() - start_time
    
    # 重みベクトルを生成
    start_time = time.time()
    weights = np.ones(len(data)) * 0.01
    weights[points_indices] = 1.0
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
    
    for i in range(len(data)):
        x = centered_data[i].reshape(-1, 1)
        weighted_cov += weights[i] * (x @ x.T)
    weighted_cov /= total_weight
    cov_time = time.time() - start_time
    
    # 固有値分解
    start_time = time.time()
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
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


def generate_weighted_pca_vectorized(data, points_indices, scaler, weights_exp=4.0):
    """ベクトル化による最適化版のPCA計算"""
    # 標準化データを取得
    start_time = time.time()
    scaled_data = scaler.transform(data)
    scaling_time = time.time() - start_time
    
    # 重みベクトルを生成
    start_time = time.time()
    weights = np.ones(len(data)) * 0.01
    weights[points_indices] = 1.0
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
    # 重みの平方根を使った方法
    sqrt_weights = np.sqrt(weights).reshape(-1, 1)
    weighted_data = centered_data * sqrt_weights
    weighted_cov = weighted_data.T @ weighted_data / weights.sum()
    cov_time = time.time() - start_time
    
    # 固有値分解
    start_time = time.time()
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    weighted_pca_components = eigvecs
    eig_time = time.time() - start_time
    
    # 結果を投影
    start_time = time.time()
    projection = centered_data @ weighted_pca_components[:, :2]
    projection_time = time.time() - start_time
    
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


def generate_weighted_pca_batched(data, points_indices, scaler, weights_exp=4.0, batch_size=500):
    """バッチ処理による最適化版のPCA計算"""
    # 標準化データを取得
    start_time = time.time()
    scaled_data = scaler.transform(data)
    scaling_time = time.time() - start_time
    
    # 重みベクトルを生成
    start_time = time.time()
    weights = np.ones(len(data)) * 0.01
    weights[points_indices] = 1.0
    if weights_exp != 1.0:
        weights = weights ** weights_exp
    weights_time = time.time() - start_time
    
    # 中心化
    start_time = time.time()
    weighted_mean = np.average(scaled_data, axis=0, weights=weights)
    centered_data = scaled_data - weighted_mean
    centering_time = time.time() - start_time
    
    # 重み付き共分散行列を計算 - バッチ処理バージョン
    start_time = time.time()
    n_samples = len(data)
    n_features = scaled_data.shape[1]
    weighted_cov = np.zeros((n_features, n_features))
    total_weight = weights.sum()
    
    # バッチ処理
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_data = centered_data[start_idx:end_idx]
        batch_weights = weights[start_idx:end_idx].reshape(-1, 1)
        
        # バッチデータを使って効率的に計算
        weighted_batch = batch_data * batch_weights
        batch_cov = weighted_batch.T @ batch_data
        weighted_cov += batch_cov
    
    weighted_cov /= total_weight
    cov_time = time.time() - start_time
    
    # 固有値分解
    start_time = time.time()
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    weighted_pca_components = eigvecs
    eig_time = time.time() - start_time
    
    # 結果を投影
    start_time = time.time()
    projection = centered_data @ weighted_pca_components[:, :2]
    projection_time = time.time() - start_time
    
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


# Numbaによる最適化を試みる関数
try:
    @numba.jit(nopython=True)
    def _compute_weighted_cov_numba(centered_data, weights, n_features):
        n_samples = centered_data.shape[0]
        weighted_cov = np.zeros((n_features, n_features))
        total_weight = np.sum(weights)
        
        for i in range(n_samples):
            x = centered_data[i].reshape(-1, 1)
            weighted_cov += weights[i] * np.outer(x, x)
        
        weighted_cov /= total_weight
        return weighted_cov
    
    def generate_weighted_pca_numba(data, points_indices, scaler, weights_exp=4.0):
        """Numba JITコンパイルによる最適化版のPCA計算"""
        # 標準化データを取得
        start_time = time.time()
        scaled_data = scaler.transform(data)
        scaling_time = time.time() - start_time
        
        # 重みベクトルを生成
        start_time = time.time()
        weights = np.ones(len(data)) * 0.01
        weights[points_indices] = 1.0
        if weights_exp != 1.0:
            weights = weights ** weights_exp
        weights_time = time.time() - start_time
        
        # 中心化
        start_time = time.time()
        weighted_mean = np.average(scaled_data, axis=0, weights=weights)
        centered_data = scaled_data - weighted_mean
        centering_time = time.time() - start_time
        
        # 重み付き共分散行列を計算 - Numba版
        start_time = time.time()
        n_features = scaled_data.shape[1]
        weighted_cov = _compute_weighted_cov_numba(centered_data, weights, n_features)
        cov_time = time.time() - start_time
        
        # 固有値分解
        start_time = time.time()
        eigvals, eigvecs = np.linalg.eigh(weighted_cov)
        
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        weighted_pca_components = eigvecs
        eig_time = time.time() - start_time
        
        # 結果を投影
        start_time = time.time()
        projection = centered_data @ weighted_pca_components[:, :2]
        projection_time = time.time() - start_time
        
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

except ImportError:
    def generate_weighted_pca_numba(data, points_indices, scaler, weights_exp=4.0):
        print("Numba is not installed. Using original implementation instead.")
        return generate_weighted_pca_original(data, points_indices, scaler, weights_exp)


def run_comprehensive_benchmark(sample_size=5000, num_selected=100, num_frames=10):
    """各最適化手法を比較するベンチマーク"""
    # データのロードと前処理
    data, labels = load_mnist_data(sample_size=sample_size)
    
    # 標準化
    scaler = StandardScaler()
    _ = scaler.fit_transform(data)
    
    # ランダムに点を選択
    selected_indices = np.random.choice(len(data), num_selected, replace=False)
    
    print(f"\n==== 総合ベンチマーク: サンプル {sample_size}, 選択点 {num_selected}, フレーム数 {num_frames} ====\n")
    
    # 各実装方法の名前とその関数
    implementations = {
        "オリジナル": generate_weighted_pca_original,
        "ベクトル化": generate_weighted_pca_vectorized,
        "バッチ処理": lambda d, p, s, w: generate_weighted_pca_batched(d, p, s, w, batch_size=500),
        "Numba": generate_weighted_pca_numba
    }
    
    # 結果を格納する辞書
    results = {}
    
    # 各実装方法でテスト実行
    for name, impl_func in implementations.items():
        print(f"\n--- {name}実装 ---")
        
        # 1つのフレームの計算時間を測定
        weight_exp = 4.0  # 最終的な重み
        frame_start = time.time()
        _, _, _, _, timing_info = impl_func(data, selected_indices, scaler, weight_exp)
        frame_time = time.time() - frame_start
        
        print(f"単一フレーム計算時間: {frame_time:.4f}秒")
        print(f"共分散行列計算時間: {timing_info['cov_time']:.4f}秒")
        
        # 全フレームのアニメーション計算を測定
        total_start_time = time.time()
        weight_exps = np.linspace(1.0, 4.0, num_frames)
        frame_times = []
        cov_times = []
        
        for w_exp in tqdm(weight_exps, desc=f"{name}フレーム生成"):
            frame_start = time.time()
            _, _, _, _, timing = impl_func(data, selected_indices, scaler, w_exp)
            frame_times.append(time.time() - frame_start)
            cov_times.append(timing['cov_time'])
        
        total_time = time.time() - total_start_time
        
        # 結果を保存
        results[name] = {
            'total_time': total_time,
            'avg_frame_time': np.mean(frame_times),
            'max_frame_time': np.max(frame_times),
            'min_frame_time': np.min(frame_times),
            'avg_cov_time': np.mean(cov_times),
            'frames_per_sec': num_frames / total_time,
            'all_frame_times': frame_times,
            'all_cov_times': cov_times
        }
        
        print(f"合計時間: {total_time:.2f}秒, フレームあたり平均: {results[name]['avg_frame_time']:.4f}秒")
    
    # 結果を比較
    print("\n==== 実装比較 ====")
    baseline = results["オリジナル"]["total_time"]
    
    print(f"{'実装':15s} {'合計時間 (秒)':15s} {'フレーム平均 (秒)':20s} {'高速化倍率':10s}")
    print("-" * 65)
    
    for name, result in results.items():
        speedup = baseline / result["total_time"]
        print(f"{name:15s} {result['total_time']:15.4f} {result['avg_frame_time']:20.4f} {speedup:10.2f}倍")
    
    # グラフ作成
    plt.figure(figsize=(12, 10))
    
    # 1. 総実行時間の比較
    plt.subplot(2, 2, 1)
    names = list(results.keys())
    times = [results[name]['total_time'] for name in names]
    plt.bar(names, times)
    plt.title('総実行時間の比較')
    plt.ylabel('時間 (秒)')
    plt.xticks(rotation=45)
    
    # 2. フレームあたりの平均時間
    plt.subplot(2, 2, 2)
    avg_times = [results[name]['avg_frame_time'] for name in names]
    plt.bar(names, avg_times)
    plt.title('フレームあたりの平均計算時間')
    plt.ylabel('時間 (秒)')
    plt.xticks(rotation=45)
    
    # 3. 共分散行列計算時間
    plt.subplot(2, 2, 3)
    cov_times = [results[name]['avg_cov_time'] for name in names]
    plt.bar(names, cov_times)
    plt.title('共分散行列計算時間')
    plt.ylabel('時間 (秒)')
    plt.xticks(rotation=45)
    
    # 4. フレーム計算時間の推移
    plt.subplot(2, 2, 4)
    for name in names:
        plt.plot(results[name]['all_frame_times'], label=name)
    plt.title('各フレームの計算時間')
    plt.xlabel('フレーム番号')
    plt.ylabel('時間 (秒)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pca_optimization_comparison.png')
    plt.close()
    
    return results


def test_different_sample_sizes():
    """異なるデータサイズでの実行時間比較"""
    sample_sizes = [1000, 2000, 5000, 10000]
    implementations = ["オリジナル", "ベクトル化", "バッチ処理", "Numba"]
    
    results = {}
    
    for size in sample_sizes:
        print(f"\n==== サンプルサイズ {size} のテスト ====")
        results[size] = run_comprehensive_benchmark(sample_size=size, num_selected=100, num_frames=5)
    
    # サンプルサイズによる変化のグラフ
    plt.figure(figsize=(12, 8))
    
    # 1. 総実行時間の変化
    plt.subplot(2, 1, 1)
    for impl in implementations:
        sizes = []
        times = []
        for size in sample_sizes:
            if impl in results[size]:  # 実装が存在する場合
                sizes.append(size)
                times.append(results[size][impl]['total_time'])
        plt.plot(sizes, times, marker='o', label=impl)
    
    plt.title('データサイズと総実行時間の関係')
    plt.xlabel('サンプルサイズ')
    plt.ylabel('時間 (秒)')
    plt.legend()
    plt.grid(True)
    
    # 2. 高速化倍率の変化
    plt.subplot(2, 1, 2)
    for impl in implementations:
        if impl == "オリジナル":
            continue  # ベースラインはスキップ
        
        sizes = []
        speedups = []
        for size in sample_sizes:
            if impl in results[size]:  # 実装が存在する場合
                sizes.append(size)
                baseline_time = results[size]["オリジナル"]['total_time']
                impl_time = results[size][impl]['total_time']
                speedup = baseline_time / impl_time
                speedups.append(speedup)
                
        plt.plot(sizes, speedups, marker='o', label=impl)
    
    plt.title('データサイズと高速化倍率の関係')
    plt.xlabel('サンプルサイズ')
    plt.ylabel('高速化倍率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('pca_scaling_comparison.png')
    plt.close()


if __name__ == "__main__":
    # 単一サイズの総合ベンチマーク
    run_comprehensive_benchmark(sample_size=5000, num_selected=100, num_frames=10)
    
    # コメントアウトを外すと異なるサンプルサイズでのテストを実行
    # test_different_sample_sizes()
