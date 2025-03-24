# tests/test_high_dimensional_data.py

import unittest
import numpy as np
from my_module.high_dimensional_data import HighDimensionalData
from my_module.data_processor import process_data

class TestHighDimensionalData(unittest.TestCase):

    def test_high_dimensional_data_creation(self):
        data = HighDimensionalData(100, 1000)
        self.assertEqual(data.get_data().shape, (100, 1000))

    def test_process_data(self):
        mean = process_data(100, 1000)
        self.assertEqual(len(mean), 1000)  # 平均値の長さは1000であるべき
        self.assertTrue(np.all(mean >= 0) and np.all(mean <= 1))  # 平均が0〜1の範囲内であるべき

if __name__ == '__main__':
    unittest.main()
