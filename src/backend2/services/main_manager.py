import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..",   )))

from typing import List
from services.core import BaseMainManager
from services.config import PipelineConfig, Config
from services.data_manager import DataManager
from services.process_manager import ProcessManager
from services.layout_manager import LayoutManager
from services.model import PositionData
from abc import ABC, abstractmethod
from services.repositories import LayoutStateRepository, ConfigRepository



class MainManager(BaseMainManager):
    
    process_manager: ProcessManager
    layout_manager: LayoutManager
    data_manager: DataManager
    layout_state_repository: LayoutStateRepository = LayoutStateRepository()
    config_repository: ConfigRepository = ConfigRepository()
    

    def __init__(self):
        # self.config_manager = ConfigManager() # config_managerをもつかconfigをもつか
        
        self.data_manager = DataManager(dir_path="")
        self.process_manager = ProcessManager() 
        self.layout_manager = LayoutManager()
        

    def _config_to_pipeline(self, config: Config) -> PipelineConfig:
        """
        ConfigをPipelineConfigに変換する
        """
        pipeline_config: PipelineConfig = [config.dimensionality_reduction_config, config.alignment_config]

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
        self.layout_state_repository.set_data((output, indecies))
        
        return output


    def update_layout(self, indecies: List[int]) -> PositionData:
        prev_layout, prev_indecies = self.layout_state_repository.get_data()
        output = self.layout_manager.update_layout(
            indecies=indecies, 
            prev_layout=prev_layout,
            prev_indecies=prev_indecies,
            data_manager=self.data_manager,
            process_manager=self.process_manager,
            config=self._config_to_pipeline(self.config_repository.get_data())
        )

        # レイアウトのキャッシュを保存する
        prev_layout[indecies] = output
        self.layout_state_repository.set_data((prev_layout, indecies))

        return output

    def update_config(self, config: Config) -> PositionData:
        
        self.config_repository.set_data(config) # set_configが非同期になったときに支障が出るかも
        # data_manager.load() # データの初期化
        output = self.init_layout() # レイアウトの初期化を行う
        return output
    
    def get_config(self) -> Config:
        return self.config_repository.get_data()
        



if __name__ == "__main__":
    main_manager = MainManager()
    main_manager.load_data()

    import time
    initial_layout = main_manager.init_layout()
    updated_layout = main_manager.update_layout(list(range(700)))
    updated_layout = main_manager.update_layout(list(range(300)))

    print(initial_layout.shape)
    print(updated_layout.shape)



