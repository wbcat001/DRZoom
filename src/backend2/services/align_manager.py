import numpy as np
from abc import ABC, abstractmethod
from process_manager import Processor
from scipy.spatial import procrustes
from config_manager import AlignmentType


class Aligner(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def align(self, X:np.ndarray, Y:np.ndarray ) -> np.ndarray:
        """
        YをXにアライメントする
        """
        pass


class ProcrustesAligner(Aligner):
    def __init__(self):
        super().__init__()

    def align(self, X:np.ndarray, Y:np.ndarray) -> np.ndarray:
        """
        Procrustes法によるアライメント
        """
        mean_X = np.mean(X, axis=0)
        normalized_X, aligned_Y, d = procrustes(X, Y)
        scale_factor = np.linalg.norm(Y - np.mean(Y, axis=0)) 
        aligned_Y *= scale_factor # scaleを元に戻す
        aligned_Y += mean_X   # 平行移動を元に戻す
        
        return aligned_Y

class NoAligner(Aligner):
    def __init__(self):
        super().__init__()

    def align(self, X:np.ndarray, Y:np.ndarray) -> np.ndarray:
        return Y


# Aliger: 手法を選択することで、alignerを選択する
class AlignManager(Processor):
    def __init__(self, method: AlignmentType = "procrustes"):
        super().__init__()
        self.method = method
        self.aligner = self.get_aligner(method)
    def get_aligner(self, method: AlignmentType) -> Aligner:
        if method == "procrustes":
            return ProcrustesAligner()
        elif method == "none":
            return NoAligner()
        else:
            raise ValueError(f"invalid method: {method}")

    def process(self, X:np.ndarray, Y:np.ndarray) -> np.ndarray:
        return self.aligner.align(X, Y)

    


