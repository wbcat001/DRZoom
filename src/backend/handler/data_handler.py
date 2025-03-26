
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print(f"Current working directory: {os.getcwd()}")
script_dir = os.path.dirname(os.path.abspath(__file__))

from model.high_dimensional_data import HighDimensionalData
from model.data import Data
from model.metadata import MetaData

from typing import List
import numpy as np
import pandas as pd
import pickle

class DataHandler:
    def __init__(self, dir_path: str):
        # dir_path = os.path.join(script_dir, dir_path)
        # 上はC:\Users\acero\Work_Research\DRZoom\src\backend\handler\data/text/alice/metadata.csvになってしまうので
     
        # get root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        dir_path = os.path.join(root_dir, dir_path)
        print(f"DataHandler: dir_path: {dir_path}")
        metadata = self.load_metadata(os.path.join(dir_path, "metadata.csv"))
        high_dim_data = self.load_high_dimensinal_data(os.path.join(dir_path, "high_dim_data.pkl"))

        self.data = Data(high_dim_data, metadata)
        

    def get_data(self) -> Data:
        return self.data

    def load_high_dimensinal_data(self, file_path: str) -> HighDimensionalData:
        """
        高次元データをnumpy形式で読み込む
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"file not found: {file_path}")
        try:
            with open(file_path, "rb") as f:
                high_dim_data = np.array(pickle.load(f))

            # high_dim_dataは(N, D)のndarray stride=1, window=50のwindowed_dataを作成 -> (N-50, D)を作って
            window = 50
            # windowed_data = np.array([high_dim_data[i:i+window] for i in range(len(high_dim_data)-window)]) これだと(752, 50, 768)になってしまう
            windowed_data = np.array([high_dim_data[i:i+window].flatten() for i in range(len(high_dim_data)-window)])
            # print(f"high_dim_data: {high_dim_data.shape}")



            print(f"windowed_data: {windowed_data.shape}")
            return windowed_data
            

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
    handler = DataHandler("data/text/alice/")
    data = handler.get_data()
    print(data.metadata[:10])