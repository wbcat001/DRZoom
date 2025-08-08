"""
次元削減アルゴリズムのモジュール
"""
from abc import ABC, abstractmethod
import numpy as np

class DimensionalityReductionBase(ABC):
    """
    次元削減アルゴリズムのための基底クラス
    """
    def __init__(self, output_dim=2, random_state=42):
        """
        初期化
        
        Args:
            output_dim (int): 出力次元数（通常は2または3）
            random_state (int): 乱数シード
        """
        self.output_dim = output_dim
        self.random_state = random_state
        
    @abstractmethod
    def fit_transform(self, data):
        """
        モデルを学習し、データを変換する
        
        Args:
            data (numpy.ndarray): 高次元データ
            
        Returns:
            numpy.ndarray: 次元削減された低次元データ
        """
        pass