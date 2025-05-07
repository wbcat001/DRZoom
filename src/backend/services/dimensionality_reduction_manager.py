import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)
from abc import ABC, abstractmethod
from services.core import Processor
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from services.config import DimensionalityReductionType


#### Dimensionality Reducer ####
class DimensionalityReducer(ABC):
    n_components: int

    @abstractmethod
    def __init__(self, n_components: int):
        self.n_components = n_components

    @abstractmethod
    def reduce(self, X: np.ndarray) -> np.ndarray:
        pass


class PCADimensionalityReducer(DimensionalityReducer):
    def __init__(self, n_components: int):
        super().__init__(n_components)

    def reduce(self, X: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=self.n_components)
        return pca.fit_transform(X)


class CustomPCSDimensionalReducer(DimensionalityReducer):
    def __init__(self, n_components: int = 2):
        super().__init__(n_components)
        # self.reducer = PCA(n_components=n_components)

    def reduce(self, X: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=self.n_components)
        return pca.fit_transform(X)


class TSNEDimensionalReducer(DimensionalityReducer):
    def __init__(self, n_components: int):
        super().__init__(n_components)

    def reduce(self, X: np.ndarray) -> np.ndarray:
        tsne = TSNE(n_components=self.n_components)
        return tsne.fit_transform(X)


class DimensionalityReductionManager(Processor):
    def __init__(
        self, method: DimensionalityReductionType = "pca", n_components: int = 2
    ):
        self.method: DimensionalityReductionType = method
        self.n_components = n_components
        self.reducer = self.get_reducer(method)

    def get_reducer(self, method: DimensionalityReductionType) -> DimensionalityReducer:
        if method == "pca":
            return PCADimensionalityReducer(self.n_components)
        elif method == "tsne":
            return TSNEDimensionalReducer(self.n_components)
        else:
            print(f"invalid method: {method}, use pca instead")
            raise ValueError(f"invalid method: {method}")

    def process(self, X: np.ndarray) -> np.ndarray:
        if self.reducer is None:
            raise ValueError("reducer is not set")

        return self.reducer.reduce(X)


if __name__ == "__main__":
    # Example usage
    reducer = DimensionalityReductionManager(method="pca")
    data = np.random.rand(100, 10)  # Example data
    reduced_data = reducer.process(data)
    print(reduced_data.shape)
