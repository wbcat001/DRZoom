from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
# pca
def run_pca(X, n_components=2, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    return X_pca

# tsne
def run_tsne(X, n_components=2, random_state=42):
    tsne = TSNE(n_components=n_components, random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    return X_tsne
# umap
def run_umap(X, n_components=2, random_state=42):
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    X_umap = reducer.fit_transform(X)
    return X_umap

