import numpy as np
from abc import ABC, abstractmethod
from process_manager import Processor

class Aligner(ABC):
    def __init__(self):
        super().__init__()

    def align(self, X:np.ndarray, Y:np.ndarray ) -> np.ndarray:
        pass


# Aliger: 手法を選択することで、alignerを選択する
class AlignManager(Processor):
    def __init__(self, method:str):
        super().__init__()
        self.method = method

    def align(self, X:np.ndarray) -> np.ndarray:
        if self.method == "procrustes":
            return self.procrustes(X)
        elif self.method == "none":
            return X
        else:
            raise ValueError(f"invalid method: {self.method}")

    def procrustes(self, X:np.ndarray) -> np.ndarray:
        # Implement Procrustes alignment here
        pass