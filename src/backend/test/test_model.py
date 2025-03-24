from model.high_dimensional_data import HighDimensionalData
from model.metadata import MetaData
from model.data import Data
import unittest
import numpy as np
import pandas as pd
class TestData(unittest.TestCase):
    
   

    def test_metadata(self):
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
        self.assertEqual(metadata.metadata["Index"][0], 1)
        self.assertEqual(metadata.metadata["Chapter"][0], 1)
        self.assertEqual(metadata.metadata["Content"][0], "test")
        self.assertEqual(metadata.metadata["Word_Count"][0], 3)
        self.assertEqual(metadata.metadata["Text_Length"][0], 3)
        self.assertEqual(metadata.metadata["Location"][0], "test")
        self.assertEqual(metadata.metadata["LocationType"][0], "test")
        self.assertEqual(metadata.metadata["Time"][0], "test")
        self.assertEqual(metadata.metadata["Event"][0], "test")
        self.assertEqual(metadata.metadata["ESummary"][0], "test")
        self.assertEqual(metadata.metadata["EImportance"][0], 1.0)
        self.assertEqual(metadata.metadata["CTag"][0], "test")
        self.assertEqual(metadata.metadata["CText"][0], "test")
        self.assertEqual(metadata.metadata["Character"][0], "test")
        self.assertEqual(metadata.metadata["ERole"][0], "test")

    def test_high_dimensional_data(self):
        high_dim_data = np.array([[1, 2, 3], [4, 5, 6]])
        data = HighDimensionalData(high_dim_data)
        self.assertEqual(data.get_data().shape, (2, 3))
        self.assertEqual(data.get_data()[0][0], 1)
        self.assertEqual(data.get_data()[0][1], 2)
        self.assertEqual(data.get_data()[0][2], 3)


if __name__ == '__main__':
    unittest.main()
