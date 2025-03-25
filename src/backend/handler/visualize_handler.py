"""
描画のための座標列を生成する

インデックスを受取、
- データをフィルタ(n)
- 次元削減(n)
- アライメント(一つ前の結果から?)
- (クラスタリング)
を行う

"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List
from model.high_dimensional_data import HighDimensionalData
from model.position_data import PositionData

from handler.align_handler import AlignmentHandler
from handler.dimensional_reduce_handler import DimensionalReduceHandler
from handler.filter_handler import FilterHandler
from handler.data_handler import DataHandler


class TransitionData:
    def __init__(self, position_data: PositionData, frame:int=2):
        self.data_list = [position_data for _ in range(frame)]
        self.frame = frame

    def update(self, data:List[HighDimensionalData]):
        self.data_list = data


# 描画のための座標列を生成する
class VisualizeHandler:
    def __init__(self, data_handler: DataHandler, filter_handler: FilterHandler, reduce_handler:DimensionalReduceHandler, align_handler: AlignmentHandler):
        
        self.filter_handler = filter_handler
        self.reduce_handler = reduce_handler
        self.align_handler = align_handler
        self.data_handler = data_handler

        high_dim_data = self.data_handler.get_data().high_dim_data
        reduced_data = self.reduce_handler.reduce(high_dim_data)
        # aligned_data = self.align_handler.align(reduced_data, None)

        self.transition_data = TransitionData(reduced_data)
        print(f"initialized: len {len(reduced_data)}")

    def update(self, indexies: List[int]):
        frame: List[PositionData] = []

        
        prev_high_dim_data = self.data_handler.get_data().high_dim_data
        prev_position_data = self.transition_data.data_list[-1] # 一つ前のフレームのデータを取得
        prev_filtered_position_data = self.filter_handler.filter_position_data(indexies, prev_position_data)

        # filter
        filtered_data = self.filter_handler.filter_high_dim_data(indexies, prev_high_dim_data)

        # reduce
        reduced_data = self.reduce_handler.reduce(filtered_data)

        # align
        aligned_data = self.align_handler.align(reduced_data, prev_filtered_position_data)

        frame.append(aligned_data)

        print(f"updated: len {len(aligned_data)}")

        return frame

if __name__ == "__main__":
    data_handler = DataHandler("data/text/alice/")
    filter_handler = FilterHandler()
    reduce_handler = DimensionalReduceHandler()
    align_handler = AlignmentHandler()


    handler = VisualizeHandler(data_handler=data_handler, filter_handler=filter_handler, reduce_handler=reduce_handler, align_handler=align_handler)
    import numpy as np
    import time
    

    start_time = time.time()
     # random 500 indexies
    new_position_data = handler.update(np.random.choice(800, 500))
    print(f"time: {time.time() - start_time}")

    import plotly.express as px
    for i in range(10):
        start_time = time.time()
        new_position_data = handler.update(np.random.choice(800, 500))
        print(f"time: {time.time() - start_time}")

        print(f"shape{new_position_data[0].shape}")
        # line plot
        fig = px.line(x=new_position_data[0][:, 0], y=new_position_data[0][:, 1])
        fig.show()

    # print(f"new_position_data: {new_position_data}")

        
        
        
        