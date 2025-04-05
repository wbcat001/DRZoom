from core import BaseProcessManager
from abc import ABC, abstractmethod
import numpy as np
from typing import List

from dimensionality_reduction_manager import DimensionalityReductionManager
from align_manager import AlignManager
from config_manager import PipelineConfig

"""
dimentionality_reductionだと、
method: pca, tsne,
n_components: number

alignmentだと
method: procrustes, none

"""
#


class ProcessManager(BaseProcessManager):
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config

    def process(self, data):
        # ここを変更することで、パイプラインの構築を変えたり色々
        process = 

        return data
    

# PipelineのProcessの中でつなぎ合わせるもの
class Processor(ABC):
    def __init__(self):
        pass

    @abstractmethod    
    def process(self):
        pass
    


## 設定配列を元に、パイプラインを構築する
## (process_type -> 選択)[] {"dimensionality_reduction": "pca", "alignment": "procrustes"}
class ProcessPipeline:
    pipeline: List[Processor] = []
    def __init__(self, config: PipelineConfig):     
        self.generate(config)
    
    def generate(self, config: PipelineConfig):
        # process_typesを元に、Pipelineを生成する
        for process_config in config:
            process_type = process_config.type
            if process_type == "dimensionality_reduction":
                method = process_config.method
                n_components = 2 # hardcoded
                self.pipeline.append(DimensionalityReductionManager(method, n_components))
            elif process_type == "alignment":
                method = process_config.method
                # X = np.random.rand(100, 2) # dummy data
                # align元を事前に関数に組み込む
                # カリー化https://qiita.com/ytaki0801/items/a0c18a78f6f8f5c8fe2a
                self.pipeline.append(AlignManager(method))
            else:
                raise ValueError(f"Invalid process type: {process_type}")
    
    def update(self, config: PipelineConfig):
        self.pipeline = []
        self.generate(config)

    def execute(self, X: np.ndarray) -> np.ndarray:
        # Implement your pipeline execution logic here
        for processor in self.pipeline:
            X = processor.process(X)
        
        return X
    


        


    



    


    
#### aligner
