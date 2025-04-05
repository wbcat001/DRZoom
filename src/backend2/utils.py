# 全体で使用する関数を定義
from functools import wraps
import time 


# 関数の実行時間を計測するデコレータ
def calc_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken by {func.__name__}: {end - start:.4f} seconds")
        return result
    return wrapper