import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.core import BaseProcessManager, Processor
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional

from services.dimensionality_reduction_manager import DimensionalityReductionManager
from services.align_manager import AlignManager
from services.config import PipelineConfig
from services.model import PositionData, HighDimensionalData

#

class ProcessManager(BaseProcessManager[PositionData]):
    """
    パイプラインの構築、inputのデータ形式を決定する
    """

    def __init__(self):
        super().__init__()

    def process(
        self,
        data: HighDimensionalData,
        prev_layout: PositionData,
        config: PipelineConfig,
    ) -> PositionData:
        self.data = data
        self.process_pipeline = ProcessPipeline(config, prev_layout)
        print(
            f"len(self.process_pipeline.pipeline): {len(self.process_pipeline.pipeline)}"
        )

        return self.process_pipeline.execute(self.data)


## 設定配列を元に、パイプラインを構築する
## (process_type -> 選択)[] {"dimensionality_reduction": "pca", "alignment": "procrustes"}
class ProcessPipeline:
    pipeline: List[Processor] = []

    def __init__(self, config: PipelineConfig, prev_layout: PositionData):
        self.generate(config, prev_layout)

    def generate(self, config: PipelineConfig, prev_layout: Optional[PositionData]):
        self.pipeline = []
        print("config", config)

        # process_typesを元に、Pipelineを生成する
        for process_config in config:
            process_type = process_config.type
            if process_type == "dimensionality_reduction":
                method = process_config.method
                self.pipeline.append(DimensionalityReductionManager(method))
            elif process_type == "alignment":
                method = process_config.method
                self.pipeline.append(AlignManager(prev_layout, method))
            else:
                raise ValueError(f"Invalid process type: {process_type}")

    def execute(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for processor in self.pipeline:
            X = processor.process(X)
            print(f"Processed data shape: {X.shape}")
        print(f"Final processed data shape: {X.shape}")
        return X


if __name__ == "__main__":
    # Example usage
    from services.config import DimensionalityReductionConfig, AlignmentConfig

    config = [
        DimensionalityReductionConfig(type="dimensionality_reduction", method="pca"),
        AlignmentConfig(type="alignment", method="procrustes"),
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
