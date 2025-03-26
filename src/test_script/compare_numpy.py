import numpy as np
import time 


# データをインデックスでフィルタする際の速度を比較する
N = 1000


# サンプルデータの作成: N × 1000の行列
data_list = [np.random.rand(1000) for i in range(N)]
data_array = np.array(data_list)

# インデックス列でのフィルタ
filter_index = np.random.choice(range(N), N//2, replace=False)
print(len(filter_index))


# numpy
start = time.time()
data_filtered = data_array[filter_index]
print(len(data_filtered))
print(time.time() - start)


# list
print("list")
start = time.time()
data_list_filtered = [data_list[i] for i in filter_index]
print(len(data_list_filtered))  
print(time.time() - start)


