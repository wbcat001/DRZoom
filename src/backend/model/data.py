from .metadata import MetaData
from .high_dimensional_data import HighDimensionalData


# Dataクラスの定義: high_dimentional_data と metadata を含む
class Data:
    def __init__(self, data: HighDimensionalData, metadata: MetaData):
        self.high_dim_data:HighDimensionalData = data
        self.metadata: MetaData = metadata