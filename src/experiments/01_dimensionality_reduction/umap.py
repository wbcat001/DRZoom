"""
UMAP（Uniform Manifold Approximation and Projection）の実装
"""
import numpy as np
from umap import UMAP as UMAPModel
from . import DimensionalityReductionBase

class UMAP(DimensionalityReductionBase):
    """
    UMAPを用いた次元削減
    """
    def __init__(self, output_dim=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
        """
        初期化
        
        Args:
            output_dim (int): 出力次元数（通常は2または3）
            n_neighbors (int): 近傍点の数
            min_dist (float): 埋め込み点間の最小距離
            metric (str): 距離メトリック
            random_state (int): 乱数シード
        """
        super().__init__(output_dim, random_state)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.model = UMAPModel(
            n_components=output_dim,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        
    def fit_transform(self, data):
        """
        UMAPモデルを学習し、データを変換する
        
        Args:
            data (numpy.ndarray): 高次元データ
            
        Returns:
            numpy.ndarray: 次元削減された低次元データ
        """
        return self.model.fit_transform(data)
        
    def set_params(self, n_neighbors=None, min_dist=None, metric=None):
        """
        パラメータを設定する
        
        Args:
            n_neighbors (int): 近傍点の数
            min_dist (float): 埋め込み点間の最小距離
            metric (str): 距離メトリック
        """
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors
        if min_dist is not None:
            self.min_dist = min_dist
        if metric is not None:
            self.metric = metric
            
        self.model = UMAPModel(
            n_components=self.output_dim,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state
        )