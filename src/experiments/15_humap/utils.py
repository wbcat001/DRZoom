import numpy as np
# pca
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=0)
a = 5

def sample_function(x):
    return x * a

print(sample_function(10))

