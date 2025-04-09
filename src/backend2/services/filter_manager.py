"""
インデックスとDataを受け取り、フィルタリングする
"""
from typing import List
from services.model import MetaData, HighDimensionalData, PositionData

class FilterManager:
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
    
"""
- データの行数に対するスケーリング
高次元データからのサンプリング処理を追加する
example
def filter_high_dim_data(self, indexies: List[int], data: HighDimensionalData) -> HighDimensionalData:
    high_dim_data = data[indexies]
    if high_dim_data.shape[0] > 1000:
        high_dim_data = high_dim_data.sample(frac=1, random_state=42).reset_index(drop=True)
    return high_dim_data


"""