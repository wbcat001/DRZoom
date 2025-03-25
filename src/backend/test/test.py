import numpy as np
from sklearn.decomposition import PCA

def pca():
    pca = PCA(n_components=2)
    data = np.random.rand(1000, 10)
    filter = [1, 2, 3]
    filter_array = np.array(filter)
    data_filtered = data[filter_array]
    result = pca.fit_transform(data_filtered)
    return result.tolist()


print(pca())