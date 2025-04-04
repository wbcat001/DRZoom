from core import BaseProcessManager
from abc import ABC, abstractmethod
import numpy as np

class ProcessManager(BaseProcessManager):
    def __init__(self):
        super().__init__()

    def process(self, data):
        # ここを変更することで、パイプラインの構築を変えたり色々
        return data
    


## 設定配列を元に、パイプラインを構築する
## (process_type -> 選択)[]
class ProcessPipeline:
    def __init__(self): 
    
    def generate(self, process_types):
        # process_typesを元に、Pipelineを生成する
        pass

    def execute(self, data):
        # Implement your pipeline execution logic here
        return data
    

# PipelineのProcessの中でつなぎ合わせるもの
class Processor(ABC):
    def __init__(self):
        pass
        
    def process(self, data):
        pass
        
class DimensionalityReducer(ABC):
    def __init__(self, n_components:int):
        super().__init__()
        self.n_compoenents = n_components
    
    def reduce(self, X: np.ndarray) -> np.ndarray:
        pass

    

class Aligner(ABC):
    def __init__(self):
        super().__init__()

    def align(self, X:np.ndarray ) -> np.ndarray:
        pass


# Aliger: 手法を選択することで、alignerを選択する
class AlignManaager(Processor):
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

    

#### dimensinality reducer
from sklearn.decomposition import PCA

class PCADimensionalityReducer(DimensionalityReducer):
    def __init__(self, n_components:int):
        super().__init__(n_components)
        
    def reduce(self, X: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=self.n_components)
        return pca.fit_transform(X)
    
#### aligner
