from abc import ABC, abstractmethod
from typing import TypedDict, Literal, List, TypeVar, Generic

#### Config ####
class BaseConfig(TypedDict):
    pass

class BaseConfigManager(ABC):
    @abstractmethod
    def get_config(self) -> BaseConfig:
        pass

    @abstractmethod
    def set_config(self, config: BaseConfig) -> None:
        pass

    


#### Data ####
T = TypeVar('T')

class BaseDataManager(Generic[T], ABC):
    data: T
    @abstractmethod
    def __init__(self):
        self.data = self.load()

    @abstractmethod
    def load(self) -> T:
        pass


#### Process ####
# define output, input type
class BaseProcessManager(Generic[T], ABC):
    @abstractmethod
    def process(self, data: T) -> T:
        pass

#### 

#### Main ####
class BaseMainManager(ABC):
    data_manager: BaseDataManager
    config_manager: BaseConfigManager
    process_manager: BaseProcessManager

    @abstractmethod
    def __init__(self, config: BaseConfig):
        self.config = config

    @abstractmethod
    def load_data(self) -> None:
        
        pass

    @abstractmethod
    def init_layout(self) -> None:
        pass

    @abstractmethod
    def update_layout(self) -> None:
        pass

    @abstractmethod
    def update_config(self):
        pass

