import time
import random

# インメモリキャッシュ（辞書）
cache = {}

# データの登録
def set_data(key, value):
    cache[key] = value

# データの取得
def get_data(key):
    return cache.get(key)

# data = pd.DataFrame({
#     "key": [f"key_{i}" for i in range(1000)],  # 1百万件のデータ
#     "value": [random.random() for _ in range(1000000)]
# })
import numpy as np
import pandas as pd
start_time = time.time()
data = {"data": np.random.rand(1000, 1000)  # 1百万件のデータ
        ,"df": pd.DataFrame({
            "key": [f"key_{i}" for i in range(1001)],  # 1百万件のデータ
            "value": [random.random() for _ in range(1001)]
        })}
print(f"データ生成にかかった時間: {time.time() - start_time:.4f}秒")
# キャッシュへのデータ登録
def benchmark_set():
    set_data("id", data)
    

# キャッシュからデータ取得
def benchmark_get():
    return get_data("id")


# 計測：登録
start_time = time.time()
benchmark_set()
end_time = time.time()
print(f"データ登録にかかった時間: {end_time - start_time:.4f}秒")

# 計測：取得
start_time = time.time()
print(benchmark_get())
end_time = time.time()
print(f"データ取得にかかった時間: {end_time - start_time:.4f}秒")
