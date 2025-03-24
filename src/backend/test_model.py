from model.data import Data
from model.high_dimensional_data import HighDimensionalData
from model.metadata import MetaData
import pandas as pd
import numpy as np

# test
def test_data():
    # テストデータの作成
    metadata = pd.DataFrame({
        "Index": [1],
        "Chapter": [1],
        "Content": ["test"],
        "Word_Count": [3],
        "Text_Length": [3],
        "Location": ["test"],
        "LocationType": ["test"],
        "Time": ["test"],
        "Event": ["test"],
        "ESummary": ["test"],
        "EImportance": [1.0],
        "CTag": ["test"],
        "CText": ["test"],
        "Character": ["test"],
        "ERole": ["test"],
    })
    metadata = MetaData(metadata)
    print(metadata)
    high_dim_data = np.array([[1, 2, 3], [4, 5, 6]])
    data = Data(high_dim_data, metadata)
    assert data.metadata["Index"][0] == 1
    assert data.metadata["Chapter"][0] == 1
    assert data.metadata["Content"][0] == "test"
    assert data.metadata["Word_Count"][0] == 3



if __name__ == "__main__":
    test_data()



