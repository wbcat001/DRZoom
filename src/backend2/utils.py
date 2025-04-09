# 全体で使用する関数を定義
from functools import wraps
import time 
import functools
import abc

# 関数の実行時間を計測するデコレータを適用するデコレータ
# def decorate_calc_time(cls):
#     for attr_name, attr_value in cls.__dict__.items():
#         if callable(attr_value) and not attr_name.startswith("__"):
#             setattr(cls, attr_name, calc_time(attr_value))
#     return cls


# 関数の実行時間を計測するデコレータ
def calc_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken by {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper

