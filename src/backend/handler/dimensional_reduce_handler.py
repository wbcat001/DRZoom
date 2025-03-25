import numpy as np
import pandera as pa
from pandera.typing import DataFrame
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Literal, Union
from abc import ABC, abstractmethod
# vector列-> 2次元のベクトルを出力する?
DimensionalReduceMethodType = Literal["pca", "tsne"]

class DimensionalReducer(ABC):
    def __init__(self, n_components:int):
        self.n_components = n_components

    @abstractmethod 
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pass

class PCADimensionalReducer(DimensionalReducer):
    def __init__(self, n_components:int):
        super().__init__(n_components)
        self.reducer = PCA(n_components=n_components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.reducer.fit_transform(X)
    

class CustomPCSDimensionalReducer(DimensionalReducer):
    def __init__(self, n_components:int=2):
        super().__init__(n_components)
        self.reducer = PCA(n_components=n_components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.reducer.fit_transform(X)

class TSNEDimensionalReducer(DimensionalReducer):
    def __init__(self, n_components:int):
        super().__init__(n_components)
        self.reducer = TSNE(n_components=n_components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.reducer.fit_transform(X)
    

class DimensionalReduceHandler:
    def __init__(self, method: DimensionalReduceMethodType = "pca", n_components:int = 2):
        self.method:DimensionalReduceMethodType = method
        self.n_components = n_components
        self.reducer = self.get_reducer(method)

    def get_reducer(self, method: DimensionalReduceMethodType):
        if method == "pca":
            return  PCADimensionalReducer(self.n_components)
        elif method == "tsne":
            return TSNEDimensionalReducer(self.n_components)
        else:
            print(f"invalid method: {method}, use pca instead")
            return PCADimensionalReducer(self.n_components)
            raise ValueError(f"invalid method: {method}")


    def reduce(self, X: np.ndarray) -> np.ndarray:
        if self.reducer is None:
            raise ValueError("reducer is not set")
        
        return self.reducer.fit_transform(X)
    
if __name__ == "__main__":
    handler = DimensionalReduceHandler()
    data = np.random.rand(1000, 10)
    reduced_data = handler.reduce(data)
    print(reduced_data[:10])





