
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.high_dimensional_data import HighDimensionalData
from model.data import Data
from model.metadata import MetaData

from typing import List
import numpy as np
import pandas as pd

class DataHandler:
    def __init__(self, dir_path: str):
        metadata = self.load_metadata(os.path.join(dir_path, "metadata.csv"))
        high_dim_data = self.load_high_dimensinal_data(os.path.join(dir_path, "high_dim_data.csv"))

        self.data = Data(high_dim_data, metadata)
        

    def get_data(self) -> List[Data]:
        return self.data

    def load_high_dimensinal_data(self, file_path: str) -> HighDimensionalData:
        """
        高次元データをnumpy形式で読み込む
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"file not found: {file_path}")
        try:
            high_dim_data = np.load(file_path)
            return high_dim_data
        except Exception as e:
            raise Exception(f"failed to load high dimensional data: {e}")

        
    
    def load_metadata(self, file_path: str) -> MetaData:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"file not found: {file_path}")
        try:
            metadata = pd.read_csv(file_path)
            return metadata
        except Exception as e:
            raise Exception(f"failed to load metadata: {e}")
        


if __name__ == "__main__":
    handler = DataHandler("data")
    data = handler.get_data()
    print(data[:10])