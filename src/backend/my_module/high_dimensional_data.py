# src/my_module/high_dimensional_data.py

import numpy as np

class HighDimensionalData:
    def __init__(self, n_samples, n_features):
        self.data = np.random.rand(n_samples, n_features)
    
    def get_data(self):
        return self.data
