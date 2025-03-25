"""
インデックスとDataを受け取り、フィルタリングする
"""
from typing import List
from model.metadata import MetaData
from model.high_dimensional_data import HighDimensionalData
from model.position_data import PositionData

class FilterHandler:
    def __init__(self):
        pass

    def filter_metadata(self, indexies: List[int], data: MetaData)->MetaData:
        """
        メタデータをフィルタリングする
        """
        return data.iloc[indexies]

    def filter_high_dim_data(self, indexies: List[int], data: HighDimensionalData) -> HighDimensionalData:
        """
        高次元データをフィルタリングする
        """
        return data[indexies] 
    
    def filter_position_data(self, indexies: List[int], data: PositionData) -> PositionData:
        """
        位置データをフィルタリングする
        """
        return data[indexies]