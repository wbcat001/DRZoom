# src/my_module/data_processor.py

from .high_dimensional_data import HighDimensionalData
import numpy as np

def process_data(n_samples, n_features):
    data = HighDimensionalData(n_samples, n_features)
    # 例えばデータの平均を計算する
    mean = np.mean(data.get_data(), axis=0)
    return mean
