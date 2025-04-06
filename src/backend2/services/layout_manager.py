

from core import BaseLayoutManager
from process_manager import ProcessManager
from data_manager import DataManager
from filter_manager import FilterManager
from model import PositionData
from config_manager import PipelineConfig
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
    
    def init_layout(self,
                    data_manager: DataManager, process_manager:ProcessManager,
                    config: PipelineConfig) -> PositionData:
            
        """
        レイアウトの初期化を行う.初期化が特殊な場合には定義する
        """
        high_dim_data = data_manager.get_data().high_dim_data
        layout = process_manager.process(
            data=high_dim_data, 
            prev_layout=None, 
            config=config
        )
        
        
        return layout

    def update_layout(self, 
                      indecies: List[int], 
                      prev_indecies: List[int], 
                      prev_layout: PositionData, 
                      data_manager: DataManager, process_manager:ProcessManager,
                      config: PipelineConfig
                      ) -> PositionData:
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
        new_layout = process_manager.process(filtered_high_dim_data, prev_filtered_position_data, config=config)
        

        return new_layout


    

if __name__ == "__main__":
    # Example usage
    layout_manager = LayoutManager(filter_manager=FilterManager())
    data_manager = DataManager(dir_path="")
    process_manager = ProcessManager()
    from config_manager import AlignmentConfig, DimensionalityReductionConfig

    pipeline_config:PipelineConfig = [
        DimensionalityReductionConfig(type="dimensionality_reduction", method="pca"),
        AlignmentConfig(type="alignment", method="procrustes")
    ]
    # layout_manager.init_layout()
    import numpy as np
    sample_prev_layout = np.random.rand(100, 2)  # Example data
    sample_data = np.random.rand(100, 100)  # Example data
    sample_indecies = list(range(100))  # Example indecies
    sample_prev_indecies = list(range(100))  # Example indecies
    print(layout_manager.update_layout(
        indecies=sample_indecies,
        prev_indecies=sample_prev_indecies,
        prev_layout=sample_prev_layout,
        data_manager=data_manager,
        process_manager=process_manager,
        config=pipeline_config
    ).shape)
    