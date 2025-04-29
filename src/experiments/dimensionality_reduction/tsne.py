"""
t-SNE（t-distributed Stochastic Neighbor Embedding）の実装
"""
import numpy as np
from sklearn.manifold import TSNE as SklearnTSNE
from . import DimensionalityReductionBase

class TSNE(DimensionalityReductionBase):
    """
    t-SNEを用いた次元削減
    """
    def __init__(self, output_dim=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, random_state=42):
        """
        初期化
        
        Args:
            output_dim (int): 出力次元数（通常は2または3）
            perplexity (float): パープレキシティ（近傍点の数に関連するパラメータ）
            learning_rate (float): 学習率
            n_iter (int): 最大反復回数
            random_state (int): 乱数シード
        """
        super().__init__(output_dim, random_state)
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.model = SklearnTSNE(
            n_components=output_dim,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=random_state
        )
        
    def fit_transform(self, data):
        """
        t-SNEモデルを学習し、データを変換する
        
        Args:
            data (numpy.ndarray): 高次元データ
            
        Returns:
            numpy.ndarray: 次元削減された低次元データ
        """
        return self.model.fit_transform(data)
        
    def set_params(self, perplexity=None, learning_rate=None, n_iter=None):
        """
        パラメータを設定する
        
        Args:
            perplexity (float): パープレキシティ
            learning_rate (float): 学習率
            n_iter (int): 最大反復回数
        """
        if perplexity is not None:
            self.perplexity = perplexity
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if n_iter is not None:
            self.n_iter = n_iter
            
        self.model = SklearnTSNE(
            n_components=self.output_dim,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            random_state=self.random_state
        )