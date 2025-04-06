from abc import ABC, abstractmethod
from typing import TypedDict, Literal, List, TypeVar, Generic

#### Config ####
class BaseConfig():
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
        レイアウトの初期化を行う.初期化が特殊な場合には定義する
        """
        pass

    @abstractmethod
    def update_layout(self) -> T:
        """
        レイアウトの更新を行う
        """
        pass

    
# PipelineのProcessの中でつなぎ合わせるもの
class Processor(ABC):
    def __init__(self):
        pass

    @abstractmethod    
    def process(self):
        pass

#### Main ####
class BaseMainManager(ABC):
    data_manager: BaseDataManager
    process_manager: BaseProcessManager
    layout_manager: BaseLayoutManager
    config: BaseConfig

    def __init__(self, data_manager: BaseDataManager, process_manager: BaseProcessManager, layout_manager: BaseLayoutManager, config: BaseConfig) -> None:
        self.data_manager = data_manager
        self.process_manager = process_manager
        self.layout_manager = layout_manager
        self.config = config

    @abstractmethod
    def load_data(self) -> None:
        """
        DataManagerにてデータを読み込む
        """
        pass

    @abstractmethod
    def init_layout(self) -> None:
        """
        LayoutManagerにてレイアウトを初期化する
        """
        pass

    @abstractmethod
    def update_layout(self) -> None:
        """
        LayoutManagerにてレイアウトを更新する
        """
        pass

    @abstractmethod
    def get_config(self) -> BaseConfig:
        """
        Configを取得する
        """
        pass

    @abstractmethod
    def update_config(self) -> None:
        """
        Configの変更を反映する
        - 処理のプロセスの変更
        - 使用するデータの変更
        - レイアウトの初期化
        """
        pass

