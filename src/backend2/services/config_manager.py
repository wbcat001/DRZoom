from abc import ABC, abstractmethod
from typing import Literal, Union, List
from core import BaseConfig, BaseConfigManager


# 処理の種類
ProcessType = Literal["dimensionality_reduction", "alignment"]

# 各処理の手法の種類
DimensionalityReductionType = Literal["pca", "tsne", "custom_pca"]
AlignmentType = Literal["procrustes", "none"]

# Configの具体的な型定義
class DimensionalityReductionConfig():
    type: Literal["dimensionality_reduction"]
    method: DimensionalityReductionType
    # オプションの追加設定もできる
    # n_components: int
    def __init__(self, type: Literal["dimensionality_reduction"], method: DimensionalityReductionType):
        self.type = type
        self.method = method
    
class AlignmentConfig():
    type: Literal["alignment"]
    method: AlignmentType
    # optional parameters
    # normalize: bool
    def __init__(self, type: Literal["alignment"], method: AlignmentType):
        self.type = type
        self.method = method


ProcessConfig = Union[DimensionalityReductionConfig, AlignmentConfig]
PipelineConfig = List[ProcessConfig]

#### アプリ全体の設定 ####
class Config(BaseConfig):
    data: str
    dimensionality_reduction_config: DimensionalityReductionConfig
    alignment_config: AlignmentConfig

    def __init__(self, data: str, dimensionality_reduction_config: DimensionalityReductionConfig, alignment_config: AlignmentConfig):
        self.data = data
        self.dimensionality_reduction_config = dimensionality_reduction_config
        self.alignment_config = alignment_config

    

class ConfigManager(BaseConfigManager):
    config: Config
    def __init__(self, config: Config):
        self.config = config

    def get_config(self) -> Config:
        return self.config
    
    def set_config(self, config: Config) -> None:
        self.config = config

"""
data_path
process_type dimensionality_reduction, alignment
"""


# test code
if __name__ == "__main__":
    config = Config(
        data="data.csv",
        dimensionality_reduction_config=DimensionalityReductionConfig(
            type="dimensionality_reduction",
            method="pca"
        ),
        alignment_config=AlignmentConfig(
            type="alignment",
            method="procrustes"
        )
    )
    config_manager = ConfigManager(config)
    print(config_manager.get_config().data)
    print(config_manager.get_config().dimensionality_reduction_config)
    print(config_manager.get_config().alignment_config.method)
    
