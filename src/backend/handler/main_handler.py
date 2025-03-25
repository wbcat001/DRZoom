from align_handler import AlignmentHandler

from dimensinal_reduce_handler import DimensionalReduceHandler
from data_handler import DataHandler

class MainHandler:
    def __init__(self):
        self.config = {
            "dir_path": "./data/",
            "reduce": "pca",
            "align": "procrustes",
        } # configファイルからの読み込み、データ型作る
        
        self.data_handler = DataHandler(self.config["dir_path"])
        self.alignment_handler = AlignmentHandler()
        self.dimensinal_reduce_handler = DimensionalReduceHandler("pca", 2)


    def update():
        pass

    def get_config(self):
        return self.config
    
    



if __name__ == "__main__":

        