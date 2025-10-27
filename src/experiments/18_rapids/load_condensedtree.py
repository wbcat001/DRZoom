import os
import pickle
import matplotlib.pyplot as plt
from hdbscan.plots import CondensedTree as hdbscan_CondensedTree


dir_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(dir_path, "result", "20251021_181430", "condensed_tree_object.pkl")

# pickleファイルからCondensedTreeオブジェクトを読み込み
with open(file_path, 'rb') as f:
    condensed_tree = pickle.load(f)
print("CondensedTree object loaded from pickle.")


# plot()メソッドで描画、保存

plt.figure(figsize=(15, 20))
condensed_tree.plot(leaf_separation=2,
                    log_size=True, 

                    max_rectangles_per_icicle=2)
ax = plt.gca()
# ax.set_ylim(1.5, 0.95)

plt.show()

