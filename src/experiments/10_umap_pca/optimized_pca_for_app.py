"""
最適化した重み付きPCA計算をアプリケーションに組み込むための実装例
"""

import numpy as np
import time
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_weighted_pca(data, points_indices, scaler, weights_exp=4.0):
    """
    最適化された重み付きPCA計算の実装
    
    Parameters:
    -----------
    data : ndarray
        元の特徴量データ
    points_indices : ndarray
        強調するデータポイントのインデックス
    scaler : StandardScaler
        スケーリング済みのスケーラーオブジェクト
    weights_exp : float
        重みの指数値 (デフォルト: 4.0)
    
    Returns:
    --------
    projection : ndarray
        2次元に投影されたデータ
    weighted_pca_components : ndarray
        重み付きPCAの主成分ベクトル
    eigvals : ndarray
        固有値
    weighted_mean : ndarray
        重み付き平均ベクトル
    """
    start_time = time.time()
    
    # 標準化データを取得
    scaled_data = scaler.transform(data)
    
    # 重みベクトルを生成（選択された点は重み1.0、その他は小さい値）
    weights = np.ones(len(data)) * 0.01  # 基本重みは0.01
    weights[points_indices] = 1.0  # 選択された点の重みは1.0
    
    # 重みを指数関数的に適用（コントラストを高める）
    if weights_exp != 1.0:
        weights = weights ** weights_exp
    
    # 中心化 - 重み付き平均の計算
    weighted_mean = np.average(scaled_data, axis=0, weights=weights)
    centered_data = scaled_data - weighted_mean
    
    # 重み付き共分散行列を計算 - ベクトル化バージョン
    sqrt_weights = np.sqrt(weights).reshape(-1, 1)
    weighted_data = centered_data * sqrt_weights
    weighted_cov = weighted_data.T @ weighted_data / weights.sum()
    
    # 固有値分解
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    
    # 固有値を降順にソート
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 固有ベクトルを重み付きPCAの主成分とする
    weighted_pca_components = eigvecs
    
    # 結果を投影
    projection = centered_data @ weighted_pca_components[:, :2]
    
    elapsed = time.time() - start_time
    logger.info(f"重み付きPCA計算完了: {elapsed:.4f}秒")
    
    return projection, weighted_pca_components, eigvals, weighted_mean


def create_weight_animation_frames(data, selected_indices, scaler, num_frames=30, max_weight_exp=4.0):
    """
    アニメーションフレーム生成の最適化実装
    
    Parameters:
    -----------
    data : ndarray
        元の特徴量データ
    selected_indices : ndarray
        強調するデータポイントのインデックス
    scaler : StandardScaler
        スケーリング済みのスケーラーオブジェクト
    num_frames : int
        生成するフレーム数 (デフォルト: 30)
    max_weight_exp : float
        最大の重み指数値 (デフォルト: 4.0)
    
    Returns:
    --------
    frames : list
        計算されたフレームのリスト
    """
    weight_exps = np.linspace(1.0, max_weight_exp, num_frames)
    frames = []
    
    # プログレス表示のためのカウンター
    start_time = time.time()
    logger.info(f"アニメーションフレーム計算開始: {len(selected_indices)}点, {num_frames}フレーム")
    
    # 標準化データを前計算（全フレームで共通）
    scaled_data = scaler.transform(data)
    
    # 基本重みベクトルも前計算
    base_weights = np.ones(len(data)) * 0.01
    base_weights[selected_indices] = 1.0
    
    for i, weight_exp in enumerate(weight_exps):
        # 重みベクトルを計算
        weights = base_weights ** weight_exp
        
        # 中心化
        weighted_mean = np.average(scaled_data, axis=0, weights=weights)
        centered_data = scaled_data - weighted_mean
        
        # 最適化された共分散行列計算
        sqrt_weights = np.sqrt(weights).reshape(-1, 1)
        weighted_data = centered_data * sqrt_weights
        weighted_cov = weighted_data.T @ weighted_data / weights.sum()
        
        # 固有値分解
        eigvals, eigvecs = np.linalg.eigh(weighted_cov)
        
        # 固有値を降順にソート
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # 投影
        projection = centered_data @ eigvecs[:, :2]
        
        # 結果を保存
        frames.append({
            'projection': projection,
            'components': eigvecs[:, :2],  # 最初の2つの主成分を保存
            'eigenvalues': eigvals[:2].tolist()  # 固有値も保存
        })
        
        # 進捗状況をログに出力（たまに）
        if (i + 1) % 10 == 0 or i == 0 or i == len(weight_exps) - 1:
            elapsed = time.time() - start_time
            logger.info(f"フレーム {i+1}/{num_frames} 完了 ({elapsed:.2f}秒経過)")
    
    total_time = time.time() - start_time
    logger.info(f"アニメーションフレーム計算完了: {total_time:.2f}秒, フレームあたり平均 {total_time/num_frames:.4f}秒")
    
    return frames


# フレーム数の動的調整機能
def suggest_frame_count(data_size, selected_points_count):
    """
    データサイズと選択ポイント数に基づいて適切なフレーム数を提案
    
    Parameters:
    -----------
    data_size : int
        データのサンプル数
    selected_points_count : int
        選択された点の数
    
    Returns:
    --------
    frame_count : int
        推奨されるフレーム数
    """
    # 基本フレーム数
    base_frames = 30
    
    # データサイズが大きい場合はフレーム数を調整
    if data_size > 10000:
        # 大きなデータセットの場合、フレーム数を少なめに
        base_frames = 15
    elif data_size > 5000:
        base_frames = 20
    
    # 選択点数が多い場合も計算負荷が上がるため調整
    if selected_points_count > 300:
        # さらにフレーム数を減らす
        base_frames = max(10, base_frames - 10)
    
    logger.info(f"データサイズ {data_size}、選択点数 {selected_points_count} に基づく推奨フレーム数: {base_frames}")
    return base_frames


# アプリケーションに組み込む場合のサンプルコード
def app_integration_example():
    """
    Dashアプリケーションに組み込む場合のコード例
    """
    # 擬似的なアプリケーションコンテキスト
    class AppContext:
        def __init__(self):
            self.data = None
            self.scaler = None
            self.animation_frames = None
            self.selected_indices = None
    
    app_ctx = AppContext()
    
    # Dashアプリケーションのコールバック関数の例
    def prepare_animation_frames_callback(selected_data, animation_state):
        """
        選択データからアニメーションフレームを生成するコールバック
        """
        if not selected_data or not selected_data.get('points'):
            return None, None, animation_state, True
        
        # 選択された点のインデックスを取得
        selected_indices = [p['pointIndex'] for p in selected_data['points']]
        app_ctx.selected_indices = selected_indices
        
        logger.info(f"選択された点: {len(selected_indices)}")
        
        # アニメーション再生を停止
        animation_state = animation_state or {"playing": False, "frame": 0}
        animation_state["playing"] = False
        animation_state["frame"] = 0
        
        # データサイズに応じてフレーム数を調整
        num_frames = suggest_frame_count(len(app_ctx.data), len(selected_indices))
        
        # 最適化された関数でアニメーションフレームを生成
        app_ctx.animation_frames = create_weight_animation_frames(
            app_ctx.data, selected_indices, app_ctx.scaler, num_frames=num_frames)
        
        # フレームデータを構造化して保存
        # 最初と最後のフレームのみをJSONに保存し、中間フレームはグローバル変数に保存
        frame_data = {
            'first': {
                'projection': app_ctx.animation_frames[0]['projection'].tolist(),
                'eigenvalues': app_ctx.animation_frames[0]['eigenvalues']
            },
            'last': {
                'projection': app_ctx.animation_frames[-1]['projection'].tolist(),
                'eigenvalues': app_ctx.animation_frames[-1]['eigenvalues']
            },
            'num_frames': num_frames  # フレーム数も送信
        }
        
        return frame_data, selected_indices, animation_state, True
    
    # 使用例を示すだけのコード
    print("これはDashアプリケーションに組み込むためのサンプルコードです")
    print("実際には以下のような形でapp.pyに最適化された関数を組み込みます:")
    print("1. generate_weighted_pca関数をオリジナルの実装と置き換える")
    print("2. create_weight_animation_frames関数をオリジナルの実装と置き換える")
    print("3. suggest_frame_count関数を追加して、データサイズに応じたフレーム数調整を実装")


if __name__ == "__main__":
    # 使用例
    app_integration_example()
