import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
from core import BaseMainManager
from config_manager import AlignmentConfig, Config, DimensionalityReductionConfig, PipelineConfig

from data_manager import DataManager
from process_manager import ProcessManager
from layout_manager import LayoutManager
from filter_manager import FilterManager
from model import PositionData
from config_manager import ConfigManager

from typing import Generic, TypeVar, Tuple
from abc import ABC, abstractmethod
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
        data = "",
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

    



class MainManager(BaseMainManager):
    
    process_manager: ProcessManager
    layout_manager: LayoutManager
    layout_state_resistory: LayoutStateRepository = LayoutStateRepository()
    config_repository: ConfigRepository = ConfigRepository()
    data_manager: DataManager

    
    
    def __init__(self):
        # self.config_manager = ConfigManager() # config_managerをもつかconfigをもつか
        
        self.data_manager = DataManager(dir_path="")
        self.process_manager = ProcessManager() 
        self.layout_manager = LayoutManager(filter_manager=FilterManager())

    
    def _config_to_pipeline(self, config: Config) -> PipelineConfig:
        """
        ConfigをPipelineConfigに変換する
        """
        pipeline_config = [config.dimensionality_reduction_config, config.alignment_config]
        return pipeline_config
 
        
    def load_data(self) -> None:
        self.data_manager.load()

    def init_layout(self) -> PositionData:

        # ズームされていない状態でのレイアウト
        indecies = list(range(len(self.data_manager.get_data().high_dim_data)))
        output = self.layout_manager.init_layout(
            data_manager=self.data_manager,
            process_manager=self.process_manager,
            config=self._config_to_pipeline(self.config_repository.get_data())
        )
        # レイアウトのキャッシュを保存する
        self.layout_state_resistory.set_data((output, indecies))
        
        return output


    def update_layout(self, indecies: List[int]) -> PositionData:
        prev_layout, prev_indecies = self.layout_state_resistory.get_data()

        output = self.layout_manager.update_layout(
            indecies=indecies, 
            prev_layout=prev_layout,
            prev_indecies=prev_indecies,
            data_manager=self.data_manager,
            process_manager=self.process_manager,
            config=self._config_to_pipeline(self.config_repository.get_data())
        )

        # レイアウトのキャッシュを保存する
        self.layout_state_resistory.set_data((output, indecies))

        return output

    def update_config(self, config: Config) -> PositionData:
        
        self.config_repository.set_data(config) # set_configが非同期になったときに支障が出るかも
        # data_manager.load() # データの初期化
        output = self.init_layout() # レイアウトの初期化を行う
        return output
    
    def get_config(self) -> Config:
        return self.config_manager.get_config()
        
        
"""
prev_layoutの保持が必要
update_layout, update_configで初期化や使用する
"""



if __name__ == "__main__":
    main_manager = MainManager()
    main_manager.load_data()

    import time
    start = time.time()
    initial_layout = main_manager.init_layout()
    print(f"time: {time.time() - start} seconds")

    start = time.time()
    updated_layout = main_manager.update_layout(list(range(700)))
    print(f"time: {time.time() - start} seconds")
    print(initial_layout.shape)
    print(updated_layout.shape)
    # main_manager.update_config(config)
    # main_manager.update_layout(indecies)


