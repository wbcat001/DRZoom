import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List
from handler.data_handler import DataHandler
from handler.align_handler import AlignmentHandler
from handler.dimensional_reduce_handler import DimensionalReduceHandler
from handler.visualize_handler import VisualizeHandler
from handler.filter_handler import FilterHandler

from model.position_data import PositionData

class MainHandler:
    def __init__(self):
        self.config = {
            "dir_path": "data\\text\\alice\\",
            "reduce": "pca",
            "align": "procrustes",
        } # configファイルからの読み込み、データ型作る
        
        self.data_handler = DataHandler(self.config["dir_path"])
        self.alignment_handler = AlignmentHandler(method=self.config["align"])
        self.dimensinal_reduce_handler = DimensionalReduceHandler(method=self.config["reduce"])
        self.filter_handler = FilterHandler()

        self.visualize_handler = VisualizeHandler(self.data_handler, self.filter_handler, self.dimensinal_reduce_handler, self.alignment_handler)
    
    def reset(self):
        self.visualize_handler = VisualizeHandler(self.data_handler, self.filter_handler, self.dimensinal_reduce_handler, self.alignment_handler
    )

    def get_initial_data(self) -> PositionData:
        return self.visualize_handler.transition_data.data_list


    def update(self, indeices: List[int]):
        return self.visualize_handler.update(indeices)

    def get_config(self):
        return self.config
    
    

if __name__ == "__main__":
    handler = MainHandler()
    data = handler.get_initial_data()
    print(data[:10])
    print(handler.update([1, 2, 3, 4, 5])[:10])
    
    