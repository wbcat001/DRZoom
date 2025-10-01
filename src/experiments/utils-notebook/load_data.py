# fetch 
from sklearn.datasets import fetch_openml
from gensim.models import KeyedVectors
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_mnist(n_samples=None):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target
    X = X.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    y = y.astype('int')
    print(f"MNIST dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.")
    print(f"Classes: {set(y)}")
    return X, y

def load_fmnist(n_samples=None):
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    X, y = fashion_mnist.data, fashion_mnist.target
    X = X.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    y = y.astype('int')
    print(f"Fashion-MNIST dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.")
    print(f"Classes: {set(y)}")
    return X, y

def load_cifar10(n_samples=None):
    cifar10 = fetch_openml('CIFAR_10', version=1, as_frame=False)
    X, y = cifar10.data, cifar10.target
    X = X.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    y = y.astype('int')
    print(f"CIFAR-10 dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.")
    print(f"Classes: {set(y)}")
    return X, y


def load_w2v(n_samples=5000):
    print(os.getcwd())
    print(BASE_DIR)
    file_path = os.path.join(BASE_DIR, "GoogleNews-vectors-negative300.bin")
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)

    words = model.index_to_key
    print(f"Number of words in the model: {len(words)}")

    # top_n words
    top_n = n_samples
    top_words = words[:top_n]
    top_vectors = np.array([model[word] for word in top_words])
    print(f"shape of vector: {top_vectors.shape}")

    return top_vectors, top_words

# ガウスによる人口データ生成

def load_synthetic_gaussian(n_samples=1000, n_features=50, n_clusters=10, cluster_stds=[1.0, 1.0, 1.0, 1.0], random_state=42):
    from sklearn.datasets import make_blobs
    import numpy as np
    
    np.random.seed(random_state)
    
    # 各クラスタのサンプル数を計算（n_samples / n_clusters）
    samples_per_cluster = [n_samples // n_clusters] * n_clusters
    remainder = n_samples % n_clusters
    for i in range(remainder):
        samples_per_cluster[i] += 1
    
    # 標準偏差をクラスタ数に合わせて割り当て
    # cluster_stdsの配列を繰り返し使用してn_clusters分確保
    std_assignments = []
    for i in range(n_clusters):
        std_assignments.append(cluster_stds[i % len(cluster_stds)])
    
    # クラスタ中心をランダムに配置（重複を避けるため適度に離す）
    cluster_centers = np.random.uniform(-10, 10, (n_clusters, n_features))
    
    X_list = []
    y_list = []
    
    for i, (n_samples_cluster, std) in enumerate(zip(samples_per_cluster, std_assignments)):
        # 各クラスタを個別に生成
        X_cluster, _ = make_blobs(
            n_samples=n_samples_cluster,
            n_features=n_features,
            centers=[cluster_centers[i]],  # このクラスタの中心
            cluster_std=std,
            random_state=random_state + i  # 各クラスタで異なるシード
        )
        
        y_cluster = np.full(n_samples_cluster, i)  # クラスタラベル
        
        X_list.append(X_cluster)
        y_list.append(y_cluster)
    
    # 全クラスタを結合
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # データをシャッフル
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    print(f"Synthetic Gaussian dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.")
    print(f"Number of clusters: {n_clusters}")
    print(f"Standard deviation variations: {cluster_stds}")
    print(f"Assigned standards: {std_assignments}")
    print(f"Samples per cluster: {samples_per_cluster}")
    print(f"Actual samples per cluster: {[np.sum(y == i) for i in range(n_clusters)]}")
    
    return X, y