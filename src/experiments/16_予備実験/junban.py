import random

names = [
    "岩田",
    "渥美",
    "荒牧",
    "吉井",
    "浅井",
    "脇田",
    "松本",
    "足立",
    "ジブ",
    "堀内",
    "田中"
]

# ランダムに順番をシャッフル
random.shuffle(names)

# 結果を出力
for name in names:
    print(name)
