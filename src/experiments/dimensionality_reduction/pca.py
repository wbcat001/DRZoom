"""
主成分分析（PCA）の実装
"""
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from . import DimensionalityReductionBase

class PCA(DimensionalityReductionBase):
    """
    主成分分析（PCA）を用いた次元削減
    """
    def __init__(self, output_dim=2, random_state=42):
        """
        初期化
        
        Args:
            output_dim (int): 出力次元数（通常は2または3）
            random_state (int): 乱数シード
        """
        super().__init__(output_dim, random_state)
        self.model = SklearnPCA(n_components=output_dim, random_state=random_state)
        
    def fit_transform(self, data):
        """
        PCAモデルを学習し、データを変換する
        
        Args:
            data (numpy.ndarray): 高次元データ
            
        Returns:
            numpy.ndarray: 次元削減された低次元データ
        """
        return self.model.fit_transform(data)
        
    def get_explained_variance_ratio(self):
        """
        各主成分の説明分散比を取得する
        
        Returns:
            numpy.ndarray: 説明分散比
        """
        return self.model.explained_variance_ratio_