from typing import Literal
import numpy as np
from abc import ABC, abstractmethod


from scipy.spatial import procrustes
class Aligner(ABC): 
    def __init__(self):
        pass

    @abstractmethod
    def align(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pass

    def no_align(self, X: np.ndarray) -> np.ndarray:
        return X

class ProcrustesAligner(Aligner):
    def __init__(self):
        super().__init__()

    def align(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Procrustes法によるアライメント
        """
        mean_X = np.mean(X, axis=0)
        normalized_X, aligned_Y, d = procrustes(X, Y)

        scale_factor = np.linalg.norm(X) / np.linalg.norm(Y)
        aligned_Y = aligned_Y * scale_factor

        aligned_Y += mean_X   

        return aligned_Y   

AlignerType = Literal["procrustes"]

class AlignmentHandler:
    def __init__(self, method: AlignerType = "procrustes"):
        self.method = method
        self.aligner = self.get_aligner(method)

    def get_aligner(self, method: AlignerType):
        if method == "procrustes":
            return ProcrustesAligner()
        else:
            print(f"invalid method: {method}, use procrustes instead")
            return ProcrustesAligner()
            raise ValueError(f"invalid method: {method}")
        
    def align(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.aligner.align(X, Y)
    



if __name__ == "__main__":
    handler = AlignmentHandler()
    X = np.random.rand(1000, 2)
    Y = np.random.rand(1000, 2)
    aligned_Y = handler.align(X, Y)
    print(aligned_Y[:10])