import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Generic, TypeVar, Tuple, List
from abc import ABC, abstractmethod
from services.config import Config, DimensionalityReductionConfig, AlignmentConfig
from services.model import PositionData

T = TypeVar("T")


class Repository(Generic[T], ABC):
    cache: T
    def __init__(self):
        pass

    @abstractmethod
    def set_data(self, data: T)-> None:
        pass

    @abstractmethod
    def get_data(self) -> T:
        pass

# TODO: ユーザー固有のデータ保持をする(Redisとか)
class LayoutStateRepository(Repository[Tuple[PositionData, List[int]]]):
    cache: Tuple[PositionData, List[int]] 
    def __init__(self):
        pass
    
    def set_data(self, data: Tuple[PositionData, List[int]]) -> None:
        self.cache = data

    def get_data(self) -> Tuple[PositionData, List[int]]:
        if self.cache is None:
            raise ValueError("Cache is not set yet.")
        return self.cache
    

class ConfigRepository(Repository[Config]):
    cache: Config = Config(
        data="",
        dimensionality_reduction_config=DimensionalityReductionConfig(
            type="dimensionality_reduction",
            method="pca",
        ),
        alignment_config=AlignmentConfig(
            type="alignment",
            method="procrustes",
        )
    )
    def __init__(self):
        pass
    
    def set_data(self, data: Config) -> None:
        self.cache = data

    def get_data(self) -> Config:
        if self.cache is None:
            raise ValueError("Cache is not set yet.")
        return self.cache

"""
class DataResistory(Resistory[Tuple[HighDimensionalData, pd.DataFrame]]):
    def __init__(self, high_dim_data: HighDimensionalData, df: pd.DataFrame):
        self.high_dim_data = high_dim_data
        self.df = df
    
    def set_data(self, data: Tuple[HighDimensionalData, pd.DataFrame]) -> None:
        high_dim_data, df = data
        self.high_dim_data = high_dim_data
        self.df = df

    def get_data(self) -> Tuple[HighDimensionalData, pd.DataFrame]:
        return self.high_dim_data, self.df
"""

    

