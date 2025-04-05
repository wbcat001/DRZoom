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
    def process(self) -> T:
        pass

class BaseLayoutManager(Generic[T], ABC):
    @abstractmethod
    def init_layout(self) -> T:
        """
        Initialize the layout.
        """
        pass

    @abstractmethod
    def update_layout(self) -> T:
        """
        Update the layout with new data.
        """
        pass

    


#### Main ####
class BaseMainManager(ABC):
    data_manager: BaseDataManager
    config_manager: BaseConfigManager
    process_manager: BaseProcessManager
    layout_manager: BaseLayoutManager

    def __init__(self, data_manager: BaseDataManager, config_manager: BaseConfigManager, process_manager: BaseProcessManager, layout_manager: BaseLayoutManager) -> None:
        self.data_manager = data_manager
        self.config_manager = config_manager
        self.process_manager = process_manager
        self.layout_manager = layout_manager

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

