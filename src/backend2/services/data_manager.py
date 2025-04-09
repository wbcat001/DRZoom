import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.model import HighDimensionalData, MetaData, Data
from services.core import BaseDataManager
import numpy as np
import pandas as pd
import pickle



class DataManager(BaseDataManager):
    """
    データを管理するクラス
    """
    data: Data

    def __init__(self, dir_path: str):
        dir_path = "data\\text\\alice\\"
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        dir_path = os.path.join(root_dir, dir_path)

        self.dir_path = dir_path
        self.data = self.load()

    def load(self) -> Data:
        high_dim_data = self.load_high_dimensional_data(os.path.join(self.dir_path, "high_dim_data.pkl"))
        metadata = self.load_metadata(os.path.join(self.dir_path, "metadata.csv"))
        return Data(high_dim_data, metadata)
        

    def load_high_dimensional_data(self, file_path: str) -> HighDimensionalData:
        """
        高次元データをnumpy形式で読み込む.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"file not found: {file_path}")
        try:
            with open(file_path, "rb") as f:
                high_dim_data = np.array(pickle.load(f))

            # N行D列のベクトルデータをwindowで平均することで (N-window行)D列のデータに変換する
            window = 50
            windowed_data = np.zeros((high_dim_data.shape[0] - window, high_dim_data.shape[1]))
            for i in range(high_dim_data.shape[0] - window):
                windowed_data[i] = np.mean(high_dim_data[i:i+window], axis=0)

            print(f"windowed_data: {windowed_data.shape}")
            return windowed_data

        except Exception as e:
            raise Exception(f"failed to load high dimensional data: {e}")
        
    def load_metadata(self, file_path: str) -> MetaData:
        """
        メタデータをDataFrame形式で読み込む.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"file not found: {file_path}")
        try:
            metadata = pd.read_csv(file_path)
            return metadata
        except Exception as e:
            raise Exception(f"failed to load metadata: {e}")
        
    def get_data(self) -> Data:
        return self.data

    
   
if __name__ == "__main__":
    data_manager = DataManager("data\\text\\alice\\")
    data = data_manager.get_data()
    print(data.high_dim_data.shape)
    print(data.metadata.head())