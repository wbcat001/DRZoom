import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import BaseProcessManager, Processor
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional

from dimensionality_reduction_manager import DimensionalityReductionManager
from align_manager import AlignManager
from config_manager import PipelineConfig
from model import PositionData, HighDimensionalData

#


class ProcessManager(BaseProcessManager[PositionData]):
    """
    パイプラインの構築、inputのデータ形式を決定する
    """
    def __init__(self):
        super().__init__()
    
    def process(self,
                data: HighDimensionalData, 
                prev_layout: PositionData, 
                config: PipelineConfig) -> PositionData:
        self.process_pipeline = ProcessPipeline(config, prev_layout)

        return self.process_pipeline.execute(data)
    


    


## 設定配列を元に、パイプラインを構築する
## (process_type -> 選択)[] {"dimensionality_reduction": "pca", "alignment": "procrustes"}
class ProcessPipeline:
    pipeline: List[Processor] = []
    def __init__(self, config: PipelineConfig, prev_layout: PositionData):     
        self.generate(config, prev_layout)
    
    def generate(self, config: PipelineConfig, prev_layout: Optional[PositionData]):
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
                self.pipeline.append(AlignManager(prev_layout, method))
            else:
                raise ValueError(f"Invalid process type: {process_type}")
    
    def update(self, config: PipelineConfig):
        self.pipeline = []
        prev_layout = None
        self.generate(config, prev_layout)

    def execute(self, X: np.ndarray) -> np.ndarray:
        # Implement your pipeline execution logic here
        for processor in self.pipeline:
            X = processor.process(X)
        
        return X


if __name__ == "__main__":
    # Example usage
    from config_manager import DimensionalityReductionConfig, AlignmentConfig
    config = [
        DimensionalityReductionConfig(type="dimensionality_reduction", method="pca"),
        AlignmentConfig(type="alignment", method="procrustes")
    ]
    prev_layout = np.random.rand(1000, 2)  # Dummy previous layout
    data = np.random.rand(1000, 700)  # Dummy data
    import time
    start = time.time()
    process_manager = ProcessManager()
    print(f"time: {time.time() - start} seconds")

    start = time.time()
    processed_data = process_manager.process(data, prev_layout, config)
    end = time.time()
    print(f"Processing time: {end - start} seconds")
    print(processed_data.shape)