import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial import procrustes
from services.core import Processor
from services.config import AlignmentType

class Aligner(ABC):
    X: np.ndarray
    @abstractmethod
    def __init__(self, X: np.ndarray):
        self.X = X 

    @abstractmethod
    def align(self, Y:np.ndarray ) -> np.ndarray:
        """
        YをXにアライメントする
        """
        pass

class ProcrustesAligner(Aligner):
    def __init__(self, X: np.ndarray):
        super().__init__(X)

    def align(self, Y:np.ndarray) -> np.ndarray:
        """
        Procrustes法によるアライメント
        """
        mean_X = np.mean(self.X, axis=0)
        normalized_X, aligned_Y, d = procrustes(self.X, Y)
        scale_factor = np.linalg.norm(Y - np.mean(Y, axis=0)) 
        aligned_Y *= scale_factor # scaleを元に戻す
        aligned_Y += mean_X   # 平行移動を元に戻す
        
        return aligned_Y

class NoAligner(Aligner):
    def __init__(self, X: np.ndarray):
        super().__init__(X)

    def align(self, Y:np.ndarray) -> np.ndarray:
        return Y



class AlignManager(Processor):
    def __init__(self, X: np.ndarray, method: AlignmentType = "procrustes" ):
        super().__init__()
        self.method = method
        self.aligner = self.get_aligner(method, X)

    def get_aligner(self, method: AlignmentType, X: np.ndarray) -> Aligner:
        if X is None:
            return NoAligner(X)
        else:
            if method == "procrustes":
                return ProcrustesAligner(X)
            elif method == "none":
                return NoAligner(X)
            else:
                raise ValueError(f"invalid method: {method}")

    def process(self, X:np.ndarray) -> np.ndarray:
        return self.aligner.align(X)


if __name__ == "__main__":
    # Example usage
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Y = np.array([[7, 8], [9, 10], [11, 12]])
    
    aligner: AlignManager = AlignManager(method="procrustes", X=X)
    aligned_Y = aligner.process(Y)
    print(aligned_Y)