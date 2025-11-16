import cupy as cp
import cudf
from cuml.cluster import HDBSCAN as cuHDBSCAN
import matplotlib.pyplot as plt
import os
from gensim.models import KeyedVectors
from hdbscan.plots import CondensedTree as hdbscan_CondensedTree
import numpy as np
from sklearn.preprocessing import normalize
from datetime import datetime
from cuml.manifold import UMAP
import plotly.express as px

n_samples = 100000

min_cluster_size = 5
min_samples = 5
max_cluster_size = 1000
cluster_selection_method = 'eom'

n_neighbors = 15
min_dist = 0.1
random_state = 42   

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tmp_date = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir_path = os.path.join(os.path.join(BASE_DIR, 'result'), tmp_date)
output_file_path = os.path.join(output_dir_path, 'cuml_hdbscan_condensed_tree_plot.png')

if os.path.exists(output_dir_path) is False:
    os.makedirs(output_dir_path)

print(f"output path: {output_file_path}")

def load_w2v(n_samples=5000, is_random=True):
    print(os.getcwd())
    print(BASE_DIR)
    file_path = os.path.join(BASE_DIR, "data", "GoogleNews-vectors-negative300.bin")
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)

    words = model.index_to_key
    print(f"Number of words in the model: {len(words)}")

    if is_random:
        np.random.seed(random_state)
        ramdom_indices = np.random.choice(len(words), size=n_samples, replace=False)

    else:
        ramdom_indices = np.arange(n_samples)
    
    selected_words = [words[i] for i in ramdom_indices]
    selected_vectors = model.vectors[ramdom_indices]

    return selected_vectors, selected_words

# word2vec読み込み
X_cpu, words = load_w2v(n_samples=n_samples)
print(f"Loaded {X_cpu.shape[0]} word vectors of dimension {X_cpu.shape[1]}.")
X_cpu = normalize(X_cpu, axis=1)
X_gpu = cp.asarray(X_cpu, dtype=cp.float32)

# cuML HDBSCANの実行
clusterer = cuHDBSCAN(min_cluster_size=min_cluster_size,
                      min_samples=min_samples,
                      max_cluster_size=max_cluster_size,
                      cluster_selection_method=cluster_selection_method).fit(X_gpu)

raw_tree_np = clusterer.condensed_tree_.to_numpy()
labels_np = clusterer.labels_.get()
print("HDBSCAN clustering on GPU completed.")

# ノイズ割合、クラスタ数の表示
n_noise = np.sum(labels_np == -1)
n_clusters = len(set(labels_np)) - (1 if -1 in labels_np else 0)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise} ({n_noise / len(labels_np) * 100:.2f}%)")

# UMAPによる2次元可視化
reducer = UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_components=2,
    random_state=42
)
embeddings = reducer.fit_transform(X_gpu)
fig = px.scatter(
    x=embeddings[:, 0].get(),
    y=embeddings[:, 1].get(),
    color=labels_np.astype(str),
    title="UMAP Projection of HDBSCAN Clusters (cuML)",
    labels={"x": "UMAP 1", "y": "UMAP 2"},
    hover_name=words
)
fig.update_traces(marker=dict(size=2))
fig.update_layout(height=800, width=800)
umap_output_file = os.path.join(output_dir_path, 'umap_plot.html')
fig.write_html(umap_output_file)

# CondensedTreeオブジェクトをcpu側に再構築
hdbscan_tree = hdbscan_CondensedTree(
    raw_tree_np,
    labels_np
)
print("CondensedTree object reconstructed on CPU.")

# plot()メソッドで描画、保存
fig, ax = plt.subplots(figsize=(10, 6))
hdbscan_tree.plot(
)
plt.savefig(output_file_path)
plt.close(fig)
print(f"✅ cuML HDBSCAN condensed tree plot saved ({output_file_path})")

# Condensed Treeオブジェクトを保存
import pickle
output_pickle_path = os.path.join(output_dir_path, 'condensed_tree_object.pkl')
with open(output_pickle_path, 'wb') as f:
    pickle.dump(hdbscan_tree, f)

# X, labelsを保存
output_data_path = os.path.join(output_dir_path, 'data.npz')
np.savez(output_data_path, X=X_cpu, labels=labels_np)
print(f"✅ Data and labels saved ({output_data_path})")


# ハイパーパラメータをテキスト保存
hyperparams = {
    "n_samples": n_samples,
    "min_cluster_size": min_cluster_size,
    "min_samples": min_samples,
    "cluster_selection_method": cluster_selection_method,
    "max_cluster_size": max_cluster_size,
    "n_neighbors": n_neighbors,
    "min_dist": min_dist,
}
hyperparams_path = os.path.join(output_dir_path, 'hyperparams.txt')
with open(hyperparams_path, 'w') as f:
    for key, value in hyperparams.items():
        f.write(f"{key}: {value}\n")
