import os
import sys
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 10_umap_pcaディレクトリのappモジュールをインポート
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, '..')
sys.path.append(app_dir)
from app import generate_weighted_pca, create_weight_animation_frames

def test_weighted_pca():
    """最適化された重み付きPCA関数のテスト"""
    print("重み付きPCAのテスト開始...")
    
    # テストデータの生成
    n_samples = 5000
    n_features = 100
    np.random.seed(42)
    data = np.random.randn(n_samples, n_features)
    
    # 選択インデックスをランダムに生成
    selected_indices = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
    
    # スケーラーの準備
    scaler = StandardScaler()
    scaler.fit(data)
    
    # 異なる重み指数でテスト
    for weight_exp in [1.0, 2.0, 4.0]:
        start_time = time.time()
        projection, components, eigvals, _ = generate_weighted_pca(data, selected_indices, scaler, weight_exp)
        elapsed = time.time() - start_time
        
        print(f"重み指数 {weight_exp}: 計算時間 = {elapsed:.4f}秒")
        print(f"  投影データの形状: {projection.shape}")
        print(f"  第1固有値: {eigvals[0]:.4f}, 第2固有値: {eigvals[1]:.4f}")
        
    print("重み付きPCAのテスト完了\n")

def test_animation_frames():
    """アニメーションフレーム生成のテスト"""
    print("アニメーションフレーム生成のテスト開始...")
    
    # テストデータの生成
    n_samples = 5000
    n_features = 100
    np.random.seed(42)
    data = np.random.randn(n_samples, n_features)
    
    # 選択インデックスをランダムに生成（異なるサイズでテスト）
    for select_ratio in [0.05, 0.1]:
        selected_indices = np.random.choice(n_samples, int(n_samples * select_ratio), replace=False)
        
        # スケーラーの準備
        scaler = StandardScaler()
        scaler.fit(data)
        
        print(f"\n選択率 {select_ratio*100:.1f}%のテスト (選択点: {len(selected_indices)}):")
        
        # フレーム数を変えてテスト
        for num_frames in [10, 20]:
            start_time = time.time()
            frames = create_weight_animation_frames(data, selected_indices, scaler, num_frames=num_frames)
            elapsed = time.time() - start_time
            
            print(f"  フレーム数 {num_frames}: 計算時間 = {elapsed:.2f}秒 (フレームあたり {elapsed/num_frames:.4f}秒)")
            print(f"  先頭フレームの固有値: {frames[0]['eigenvalues']}")
            print(f"  最終フレームの固有値: {frames[-1]['eigenvalues']}")
    
    print("アニメーションフレーム生成のテスト完了")

if __name__ == "__main__":
    print("===== 最適化されたPCA実装のテスト =====")
    test_weighted_pca()
    test_animation_frames()
    print("\nすべてのテストが正常に完了しました！")
