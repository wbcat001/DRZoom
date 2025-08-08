import os
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

def run_original_weighted_pca(data, selected_indices, scaler, weight_exp=2.0):
    """オリジナルの重み付きPCA実装"""
    # データの前処理
    scaled_data = scaler.transform(data)
    n_samples, n_features = scaled_data.shape
    
    # 重みを初期化（選択された点は1.0、それ以外は0.01）
    weights = np.ones(n_samples) * 0.01
    weights[selected_indices] = 1.0
    
    # 指定された指数で重みを調整
    weights = weights ** weight_exp
    
    # 重み付き平均と中心化
    weighted_mean = np.zeros(n_features)
    total_weight = 0
    
    for i in range(n_samples):
        weighted_mean += weights[i] * scaled_data[i]
        total_weight += weights[i]
    
    weighted_mean /= total_weight
    centered_data = scaled_data - weighted_mean
    
    # 重み付き共分散行列の計算
    weighted_cov = np.zeros((n_features, n_features))
    
    for i in range(n_samples):
        weighted_cov += weights[i] * np.outer(centered_data[i], centered_data[i])
    
    weighted_cov /= total_weight
    
    # 固有値分解
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    
    # 固有値を降順にソート
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 投影を計算
    projection = centered_data @ eigvecs
    
    return projection, eigvecs.T, eigvals, centered_data

def run_optimized_weighted_pca(data, selected_indices, scaler, weight_exp=2.0):
    """最適化された重み付きPCA実装"""
    # データの前処理
    scaled_data = scaler.transform(data)
    
    # 重みを初期化（選択された点は1.0、それ以外は0.01）
    weights = np.ones(len(data)) * 0.01
    weights[selected_indices] = 1.0
    
    # 指定された指数で重みを調整
    weights = weights ** weight_exp
    total_weight = weights.sum()
    
    # 重み付き平均と中心化（ベクトル化）
    weighted_mean = np.average(scaled_data, axis=0, weights=weights)
    centered_data = scaled_data - weighted_mean
    
    # 重み付き共分散行列（ベクトル化）
    sqrt_weights = np.sqrt(weights).reshape(-1, 1)
    weighted_data = centered_data * sqrt_weights
    weighted_cov = weighted_data.T @ weighted_data / total_weight
    
    # 固有値分解
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    
    # 固有値を降順にソート
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 投影を計算
    projection = centered_data @ eigvecs
    
    return projection, eigvecs.T, eigvals, centered_data

def analyze_performance(data_size=5000, selected_ratio=0.05, weight_exps=[1.0, 2.0, 3.0, 4.0], n_runs=5):
    """パフォーマンスを測定し比較"""
    print(f"データサイズ: {data_size}, 選択比率: {selected_ratio:.2f}, 実行回数: {n_runs}")
    
    # ダミーデータの生成
    np.random.seed(42)
    data = np.random.randn(data_size, 100)  # 100次元のランダムデータ
    
    # 選択インデックスの作成
    n_selected = int(data_size * selected_ratio)
    selected_indices = np.random.choice(data_size, n_selected, replace=False)
    
    # データの標準化
    scaler = StandardScaler()
    scaler.fit(data)
    
    results = {
        'weight_exp': [],
        'original_time': [],
        'optimized_time': [],
        'speedup': []
    }
    
    for weight_exp in weight_exps:
        print(f"\n重み指数: {weight_exp}")
        
        # 元の実装での測定
        original_times = []
        for i in range(n_runs):
            start_time = time.time()
            run_original_weighted_pca(data, selected_indices, scaler, weight_exp)
            original_times.append(time.time() - start_time)
        avg_original_time = np.mean(original_times)
        
        # 最適化実装での測定
        optimized_times = []
        for i in range(n_runs):
            start_time = time.time()
            run_optimized_weighted_pca(data, selected_indices, scaler, weight_exp)
            optimized_times.append(time.time() - start_time)
        avg_optimized_time = np.mean(optimized_times)
        
        # 高速化の倍率
        speedup = avg_original_time / avg_optimized_time
        
        print(f"  オリジナル実装: {avg_original_time:.4f}秒 (最小: {min(original_times):.4f}秒, 最大: {max(original_times):.4f}秒)")
        print(f"  最適化実装: {avg_optimized_time:.4f}秒 (最小: {min(optimized_times):.4f}秒, 最大: {max(optimized_times):.4f}秒)")
        print(f"  高速化倍率: {speedup:.2f}倍")
        
        results['weight_exp'].append(weight_exp)
        results['original_time'].append(avg_original_time)
        results['optimized_time'].append(avg_optimized_time)
        results['speedup'].append(speedup)
    
    return results

def plot_results(results):
    """結果をグラフ化"""
    plt.figure(figsize=(12, 10))
    
    # 時間のグラフ
    plt.subplot(2, 1, 1)
    plt.plot(results['weight_exp'], results['original_time'], 'o-', label='オリジナル実装')
    plt.plot(results['weight_exp'], results['optimized_time'], 's-', label='最適化実装')
    plt.xlabel('重み指数')
    plt.ylabel('実行時間 (秒)')
    plt.title('重み付きPCA実装の性能比較')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 高速化倍率のグラフ
    plt.subplot(2, 1, 2)
    plt.bar(results['weight_exp'], results['speedup'], width=0.2)
    plt.xlabel('重み指数')
    plt.ylabel('高速化倍率')
    plt.title('最適化による高速化倍率')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 高速化倍率の数値を表示
    for i, v in enumerate(results['speedup']):
        plt.text(results['weight_exp'][i], v + 0.2, f"{v:.2f}倍", ha='center')
    
    plt.tight_layout()
    plt.savefig('pca_optimization_comparison.png', dpi=300)
    plt.show()

def main():
    """メイン関数"""
    print("重み付きPCAパフォーマンス比較ツール")
    print("=====================================")
    
    # 様々なデータサイズでテスト
    data_sizes = [1000, 5000, 10000]
    all_results = []
    
    for data_size in data_sizes:
        print(f"\nデータサイズ: {data_size}のテスト開始")
        results = analyze_performance(data_size=data_size)
        all_results.append(results)
        
        # 結果の保存と表示
        df = pd.DataFrame(results)
        df.to_csv(f'weighted_pca_comparison_{data_size}.csv', index=False)
        
        # 結果をプロット
        plot_results(results)

if __name__ == "__main__":
    main()
