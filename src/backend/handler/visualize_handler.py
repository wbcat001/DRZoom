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
import time

class TransitionData:
    def __init__(self, position_data: PositionData, frame:int=1):
        # position dataはすべての点のデータを保存するもの
        self.data_list = position_data
        self.frame = frame
        self.indecies = list(range(len(position_data)))

    def update(self, data:List[HighDimensionalData], indecies: List[int]):
   
        self.data_list[indecies] = data[0]
        self.indecies = indecies


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

        high_dim_data = self.data_handler.get_data().high_dim_data
        prev_position_data = self.transition_data.data_list# 一つ前のフレームのデータを取得

        start = time.time()
        # ２つのリストから、共通のインデックスを作る
        prev_indecies = self.transition_data.indecies
        common_indecies = list(set(indexies) & set(prev_indecies))

        print(f"indexeis time: {time.time() - start}")

        print(f"common_indecies: {len(common_indecies)}")   
        print(f"prev position data: {len(prev_position_data)}") 
        # インデックスに対応する位置を取得

        start = time.time()

        prev_filtered_position_data = self.filter_handler.filter_position_data(common_indecies, prev_position_data)

        # filter
        filtered_high_dim_data = self.filter_handler.filter_high_dim_data(common_indecies, high_dim_data)

        print(f"filter time: {time.time() - start}")
        # reduce
        start = time.time()
        reduced_data = self.reduce_handler.reduce(filtered_high_dim_data)
        print(f"reduce time: {time.time() - start}")
        # align
        start = time.time()
        aligned_data = self.align_handler.align(prev_filtered_position_data, reduced_data)
        print(f"align time: {time.time() - start}")
        frame.append(aligned_data)
        # self.transition_data.update(frame, indexies)

        start = time.time()
        self.transition_data.update(frame, indexies)
        print(f"update time: {time.time() - start}")
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
    random_indexies = [np.random.choice(800, 500)]
    print(len(random_indexies))
    new_position_data = handler.update(random_indexies)
    print(f"time: {time.time() - start_time}")

    # import plotly.express as px
    # for i in range(10):
    #     start_time = time.time()
    #     new_position_data = handler.update(np.random.choice(800, 500))
    #     print(f"time: {time.time() - start_time}")

    #     print(f"shape{new_position_data[0].shape}")
    #     # line plot
    #     fig = px.line(x=new_position_data[0][:, 0], y=new_position_data[0][:, 1])
    #     fig.show()

    # print(f"new_position_data: {new_position_data}")

        
        
        
        