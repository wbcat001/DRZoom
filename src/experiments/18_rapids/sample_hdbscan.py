import cupy as cp
import cudf
from cuml.cluster import HDBSCAN as cuHDBSCAN
import matplotlib.pyplot as plt

# === GPUでHDBSCANを実行 ===
X_gpu = cp.random.rand(300, 5)
clusterer = cuHDBSCAN(min_cluster_size=10)
clusterer.fit(X_gpu)

# === condensed_tree_ を取得（GPU上のcudf.DataFrame） ===
tree_df = clusterer.condensed_tree_

# === CPUに転送 ===
tree_cpu = tree_df.to_pandas()

print(tree_cpu.head())

# 例: カラムは ["parent", "child", "lambda_val", "child_size"]
# lambda_val が「密度階層の高さ」に相当する

# === 独自プロット ===
plt.figure(figsize=(10, 6))

# 各親ノードに対して子ノードを描画
for _, row in tree_cpu.iterrows():
    plt.plot(
        [row["parent"], row["child"]],
        [row["lambda_val"], row["lambda_val"]],
        color="steelblue", alpha=0.6
    )

plt.xlabel("Cluster tree index")
plt.ylabel("Lambda (density level)")
plt.title("Condensed Tree (GPU-based simplified plot)")
plt.tight_layout()
plt.savefig("gpu_condensed_tree.png", dpi=300)
plt.close()

print("✅ GPU condensed tree plot saved (gpu_condensed_tree.png)")
