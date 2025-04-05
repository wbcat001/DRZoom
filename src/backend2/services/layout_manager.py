

from core import BaseLayoutManager
from process_manager import ProcessManager
from data_manager import DataManager
from filter_manager import FilterManager
from model import PositionData

from typing import List


class LayoutManager(BaseLayoutManager[PositionData]):
    """
    1. レイアウトの初期化
    2. 一つ前のレイアウトと現在のデータを元に、レイアウトを更新
    の動作に関するクラス 
    """
    filter_manager: FilterManager

    def __init__(self, filter_manager:FilterManager) -> None:
        self.filter_manager = filter_manager   
    
    def initialize_layout(self) -> PositionData:
        """
        レイアウトの初期化を行う.初期化が特殊な場合には定義する
        """
        # Initialize the layout with the processed data
        pass

    def update_layout(self, indecies: List[int], prev_indecies: List[int], prev_layout: PositionData, data_manager: DataManager, process_manager:ProcessManager) -> PositionData:
        """
        レイアウトの更新を行う
        """
        # 多段階のレイアウト更新を行う場合はoutputを配列にする
        # frame: List[PositionData] = []
        

        high_dim_data = data_manager.get_data().high_dim_data
        common_indecies = list(set(indecies) & set(prev_indecies))

        # データのフィルタ: レイアウト計算に必要なデータを取得
        prev_filtered_position_data = self.filter_manager.filter_position_data(common_indecies, prev_layout)
        filtered_high_dim_data = self.filter_manager.filter_high_dim_data(common_indecies, high_dim_data)

        # レイアウトの計算
        new_layout = process_manager.process(filtered_high_dim_data, prev_filtered_position_data)
        

        return new_layout


    

if __name__ == "__main__":
    # Example usage
    layout_manager = LayoutManager(process_manager=ProcessManager())
    layout_manager.initialize_layout()
    import numpy as np
    sample_prev_layout = np.random.rand(100, 2)  # Example data
    sample_data = np.random.rand(100, 100)  # Example data
    sample_indecies = list(range(100))  # Example indecies
    sample_prev_indecies = list(range(100))  # Example indecies
    layout_manager.update_layout(sample_indecies, sample_prev_indecies, sample_prev_layout, sample_data)
    