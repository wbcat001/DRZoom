import pandas as pd
import random
import time

# データ件数
N = 1000

# サンプルデータの作成: カラム数は2
data_list = [{"id": i, "col1": random.random()} for i in range(N)]
df = pd.DataFrame(data_list)

# # dfに型付けしてデータとして扱いたい
# df["id"] = df["id"].astype(int)
# df["col1"] = df["col1"].astype(float)
# print(df.dtypes)




# インデックス列でのフィルタ
filter_index = random.sample(range(N), N//2)
print(len(filter_index))

# df
start = time.time()
df_filtered = df[df["id"].isin(filter_index)]
print(len(df_filtered))
print(time.time() - start)

# dict
start = time.time()
data_dict = {d["id"]: d for d in data_list}
data_dict_filtered = {k: v for k, v in data_dict.items() if k in filter_index}
print(len(data_dict_filtered))  

print(time.time() - start)
# 500