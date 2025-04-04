import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List

from core import BaseMainManager

from backend2.services.config_manager import Config


class MainManager(BaseMainManager):
    
    def __init__(self, config: Config):
        self.config = config

    def update_config(self, config: Config) -> None:
        self.config = config

    def load_data(self):
        pass

        
        
    


    



