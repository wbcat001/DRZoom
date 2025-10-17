import numpy as np
import hdbscan
import pandas as pd
from typing import List, Dict, Any, Tuple

class DataManager:
    """高次元データ、クラスタリング結果、埋め込み座標を管理するシングルトン的なクラス"""
    
    def __init__(self, data_vectors: np.ndarray):
        self.data_vectors = data_vectors # N x D の高次元データ
        self.N = data_vectors.shape[0]
        self.hdbscan_tree: hdbscan.HDBSCAN = None
        self.current_embeddings: Dict[str, np.ndarray] = {} # {level_id: coordinates}
        self.level_lambdas: List[float] = [] # 意味のあるズームレベル（ラムダ値）

   

    def get_cluster_members(self, cluster_id: int, cluster_labels: np.ndarray = None) -> np.ndarray:
        """指定されたクラスタIDに属するデータのインデックスを取得"""
        if cluster_labels is None:
            cluster_labels = self.current_cluster_ids
        return np.where(cluster_labels == cluster_id)[0]
    
    

    def initialize_hdbscan(self, min_cluster_size: int = 10, min_samples: int = 5):
        """HDBSCANを実行し、階層構造を構築する"""
        print("HDBSCANを初期化中...")
        self.hdbscan_tree = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, 
            min_samples=min_samples, 
            prediction_data=True
        ).fit(self.data_vectors)
        self.extract_significant_lambdas()
        print(f"HDBSCAN完了。有意なズームレベル数: {len(self.level_lambdas)}")

    def extract_significant_lambdas(self):
        """デンドログラムから「意味のあるズームレベル」となるラムダ値を抽出する"""
        # HDBSCANのcondensed_treeからstabilityが大きく変化するノードのlambda値を抽出
        
        # 凝縮された木のDataFrameを取得
        condensed_tree = pd.DataFrame(self.hdbscan_tree.condensed_tree_.copy())
        
        # 結合が発生するノードのみに限定
        merge_nodes = condensed_tree[condensed_tree['parent'] != -1].copy()
        
        # Stability (存続期間) の差分が大きなノードを「重要」と見なす
        # ここでは簡略化のため、クラスタの数が大きく変わるλ値を選ぶ、または一定間隔で抽出
        
        # Option 1: 一定間隔でλを抽出 (最も単純)
        min_lambda = condensed_tree['lambda_val'].min()
        max_lambda = condensed_tree['lambda_val'].max()
        # 5段階のズームレベルを設定
        self.level_lambdas = np.linspace(min_lambda, max_lambda, 5).tolist()
        
        # Option 2: 安定度の高いクラスタの分割点を利用するロジック（厳密な実装で必要）
        # 例: parentノードのλ値が0.5以上変化する場所など
        
        # Level 1 (Overview First)を常に含める
        if min_lambda not in self.level_lambdas:
             self.level_lambdas.insert(0, min_lambda) 
        
        self.level_lambdas = sorted(list(set(self.level_lambdas)))

# グローバルなデータマネージャーインスタンス（FastAPIで依存性注入を想定）
DATA_MANAGER = None 

# 例：ダミーデータで初期化
def init_data_manager(num_points=1000, dim=768):
    global DATA_MANAGER
    dummy_data = np.random.rand(num_points, dim)
    DATA_MANAGER = DataManager(dummy_data)
    DATA_MANAGER.initialize_hdbscan()
    return DATA_MANAGER

# FastAPIの依存性注入で使うための関数
def get_data_manager() -> DataManager:
    return DATA_MANAGER