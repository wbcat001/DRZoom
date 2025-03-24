from align_handler import AlignmentHandler

from dimensinal_reduce_handler import DimensinalReduceHandler
from data_handler import DataHandler

class MainHandler:
    def __init__(self, dir_path: str):
        
        self.data_handler = DataHandler(dir_path)
        self.alignment_handler = AlignmentHandler()
        self.dimensinal_reduce_handler = DimensinalReduceHandler("pca", 2)
        