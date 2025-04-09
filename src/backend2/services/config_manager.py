import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from abc import ABC, abstractmethod
from typing import Literal, Union, List
from services.core import BaseConfig, BaseConfigManager
from services.config import Config
    
class ConfigManager(BaseConfigManager):
    config: Config
    def __init__(self, config: Config):
        self.config = config

    def get_config(self) -> Config:
        return self.config
    
    def set_config(self, config: Config) -> None:
        self.config = config


# test code
if __name__ == "__main__":
    from services.config import DimensionalityReductionConfig, AlignmentConfig
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
    
