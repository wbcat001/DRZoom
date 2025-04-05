import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
from core import BaseMainManager
from config_manager import Config

from data_manager import DataManager
from process_manager import ProcessManager
from layout_manager import LayoutManager
from filter_manager import FilterManager
from model import PositionData
from config_manager import ConfigManager

class MainManager(BaseMainManager):
    config_manager: ConfigManager
    data_manager: DataManager
    process_manager: ProcessManager
    layout_manager: LayoutManager
    
    def __init__(self):
        self.config_manager = Config() # config_managerをもつかconfigをもつか
        self.data_manager = DataManager(self.config["dir_path"])
        self.process_manager = ProcessManager(self.config["process"])
        self.layout_manager = LayoutManager(process_manager=self.process_manager, filter_manager=FilterManager())

        # super().__init__(data_manager=data_manager, process_manager=process_manager, layout_manager=layout_manager, config_manager=config_manager)
        
    def load_data(self) -> None:
        self.data_manager.load(self.config["dir_path"])

    def init_layout(self) -> PositionData:
        # ズームされていない状態でのレイアウト
        indecies = list(range(self.data_manager.get_data().high_dim_data))
        output = self.update_layout(indecies)
        return output


    def update_layout(self, indecies: List[int]) -> PositionData:

        output = self.layout_manager.update_layout(
            indecies=indecies, 
            prev_layout=,
            data_manager=self.data_manager,
            process_manager=self.process_manager
        )
        return output

    def update_config(self, config: Config) -> PositionData:
        
        self.config_manager.set_config(config)
        self.data_manager.load()
        self.process_manager.update(config["process"])
        # self.process_manager2.update(config["process"])
        
        output = self.init_layout()
        return output
    
    def get_config(self) -> Config:
        return self.config_manager.get_config()
        
        
"""
prev_layoutの保持が必要
update_layout, update_configで初期化や使用する
"""


    



