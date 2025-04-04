from abc import ABC, abstractmethod
from typing import Literal, TypedDict
from core import BaseConfig, BaseConfigManager


DimensionalityReductionType = Literal["pca", "tsne", "custom_pca"]
AlignmentType = Literal["procrustes", "none"]
    

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

    
