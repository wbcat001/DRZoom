import numpy as np
import pandas as pd

PositionData = np.ndarray

MetaData = pd.DataFrame
HighDimensionalData = np.ndarray 

class Data:
    def __init__(self, high_dim_data: HighDimensionalData, metadata: MetaData):
        self.high_dim_data:HighDimensionalData = high_dim_data
        self.metadata: MetaData = metadata