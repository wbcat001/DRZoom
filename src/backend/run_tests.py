# run_tests.py

import unittest

if __name__ == "__main__":
    # テストディレクトリのパスを指定してテストを実行
    test_loader = unittest.defaultTestLoader
    test_suite = test_loader.discover('test')
    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)
