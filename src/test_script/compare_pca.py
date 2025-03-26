"""
pcaのスクラッチ実装と実行時間測定

"""

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        # 平均を引いて中心化
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 共分散行列を計算
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # 固有値と固有ベクトルを計算
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # 固有値の大きい順にソート
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.explained_variance = eigenvalues[sorted_indices]
        self.components = eigenvectors[:, sorted_indices][:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

# カスタムPCA（例えば、特定の主成分を強調するような調整）
class CustomPCA(PCA):
    def __init__(self, n_components, custom_weights=None):
        super().__init__(n_components)
        self.custom_weights = custom_weights
    
    def fit(self, X):
        super().fit(X)
        if self.custom_weights is not None:
            # 指定された重みを主成分に適用
            self.components *= self.custom_weights[:self.n_components]
    
    def transform(self, X):
        return super().transform(X)

# サンプルデータ
np.random.seed(42)
X_sample = np.random.rand(100, 5)  # 100サンプル, 5次元

# 通常のPCA
pca = PCA(n_components=2)
pca.fit(X_sample)
X_pca = pca.transform(X_sample)
print("Standard PCA result:")
print(X_pca[:5])

# カスタムPCA（主成分1に1.5倍、主成分2に0.8倍の重み）
custom_weights = np.array([1.5, 0.8, 1, 1, 1])
custom_pca = CustomPCA(n_components=2, custom_weights=custom_weights)
custom_pca.fit(X_sample)
X_custom_pca = custom_pca.transform(X_sample)
print("Custom PCA result:")
print(X_custom_pca[:5])

# performance test
import time
X_sample = np.random.rand(1000, 1000)  # 1000サンプル, 1000次元
start = time.time()
pca = PCA(n_components=2)
pca.fit(X_sample)
print("Standard PCA performance test:")  # 1000次元から2次元に削減
print(time.time() - start)
