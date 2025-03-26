from typing import Literal
import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import svd

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


    # Align Y to X
    def align(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Procrustes法によるアライメント
        """
        # rangeの出力 (Xは2次元の座標列)
        print(f"X_0_range" , [np.min(X[:, 0]), np.max(X[:, 0])])
        print(f"X_1_range" , [np.min(X[:, 1]), np.max(X[:, 1])])
        print(f"Y_0_range" , [np.min(Y[:, 0]), np.max(Y[:, 0])])
        print(f"Y_1_range" , [np.min(Y[:, 1]), np.max(Y[:, 1])])
       
        mean_X = np.mean(X, axis=0)
        normalized_X, aligned_Y, d = procrustes(X, Y)

        scale_factor = np.linalg.norm(Y - np.mean(Y, axis=0)) 
        aligned_Y *= scale_factor

        aligned_Y += mean_X   
        
    
        

        return aligned_Y   
    # def align(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    #     return Y

    # def align(self, data1, data2):
    #     """
    #     Procrustes解析（回転のみ）を行い、data2 を data1 に最適な回転でフィットさせる
    #     """
    #     # 中心化（平行移動を除去）
    #     mu1 = np.mean(data1, axis=0)
    #     mu2 = np.mean(data2, axis=0)
    #     data1_centered = data1 - mu1
    #     data2_centered = data2 - mu2
        
    #     # 最適な回転行列を求める（SVDを用いる）
    #     U, _, Vt = svd(data1_centered.T @ data2_centered)
    #     R = U @ Vt  # 回転行列
        
    #     # data2 に回転を適用
    #     data2_rotated = data2_centered @ R.T + mu1  # 回転後、data1 の重心位置に戻す
        
        
    #     return data2_rotated # 回転後のデータと回転行列を返す

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
        
    # Align Y to X
    def align(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.aligner.align(X, Y)
    



if __name__ == "__main__":
    handler = AlignmentHandler()
    X = np.random.rand(1000, 2)
    Y = np.random.rand(1000, 2)
    aligned_Y = handler.align(X, Y)
    print(aligned_Y[:10])