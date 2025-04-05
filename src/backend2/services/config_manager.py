from abc import ABC, abstractmethod
from typing import Literal, TypedDict, Union, List
from core import BaseConfig, BaseConfigManager


# 処理の種類
ProcessType = Literal["dimensionality_reduction", "alignment"]

# 各処理の手法の種類
DimensionalityReductionType = Literal["pca", "tsne", "custom_pca"]
AlignmentType = Literal["procrustes", "none"]

# Configの具体的な型定義
class DimensionalityReductionConfig(TypedDict):
    type: Literal["dimensionality_reduction"]
    method: DimensionalityReductionType
    # オプションの追加設定もできる
    # n_components: int
    
class AlignmentConfig(TypedDict):
    type: Literal["alignment"]
    method: AlignmentType
    # optional parameters
    # normalize: bool

ProcessConfig = Union[DimensionalityReductionConfig, AlignmentConfig]
PipelineConfig = List[ProcessConfig]


class Config(BaseConfig):
    dimensionality_reduction: DimensionalityReductionType
    alignment: AlignmentType
    

class ConfigManager(BaseConfigManager):
    def __init__(self, config: Config):
        self.config = config

    def get_config(self) -> Config:
        return self.config
    
    def set_config(self, config: Config) -> None:
        self.config = config
    

# test code
if __name__ == "__main__":
    config = Config(
        dimensionality_reduction="pca",
        alignment="procrustes"
    )
    config_manager = ConfigManager(config)
    print(config_manager.get_config())
    config_manager.set_config(Config(dimensionality_reduction="tsne", alignment="none"))
    print(config_manager.get_config())

    
