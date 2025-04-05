# pca
from sklearn.decomposition import PCA
import numpy as np

# decorator
# 10回実行して計測する
def timeit(func):
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        for _ in range(10):
            result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


# pca
@timeit
def func(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

@timeit
def func2(pca, X: np.ndarray, n_components: int = 2, ) -> np.ndarray:
    return pca.fit_transform(X)


# pca = PCA(n_components=2)
# data = np.random.rand(1000, 100)  # 1000 samples, 100 features
# func2(pca, data, n_components=2)

data = np.random.rand(1000, 100)  # 1000 samples, 100 features
func(data, n_components=2)
