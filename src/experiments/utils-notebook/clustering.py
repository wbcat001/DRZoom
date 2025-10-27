import umap
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import numpy as np
import hdbscan
from itertools import product

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# dbscanでできた階層的なクラスタリングの変化をumapで可視化
def hierarchical_dbscan_umap(X, eps_list=None, min_samples=5, n_neighbors=15, n_components=2, random_state=42):
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=random_state)
    proj = reducer.fit_transform(X)
    print(f"UMAP projection shape: {proj.shape}")

    # eps_listの自動生成
    if eps_list is None:
        # k-NNの距離を計算
        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(proj)
        distances, indices = nbrs.kneighbors(proj)
        # k-NN距離の平均を計算し、epsの候補とする
        mean_distances = distances.mean(axis=1)
        eps_list = sorted(mean_distances)[::len(mean_distances)//5][:5]  # 5段階に分ける
        print(f"Auto-generated eps_list: {np.round(eps_list, 4)}")

    dfs = []
    for eps in tqdm(eps_list, desc="DBSCAN clustering"):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(proj)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"eps: {eps:.4f}, clusters: {n_clusters}")
        
        df = pd.DataFrame(proj, columns=[f"UMAP_{i+1}" for i in range(n_components)])
        df['cluster'] = labels
        df['eps'] = eps
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    fig = px.scatter(
        df_all, 
        x='UMAP_1', 
        y='UMAP_2', 
        color='cluster',
        facet_col='eps',
        title='Hierarchical DBSCAN Clustering with UMAP Projection',
    )
    n_facets = len(eps_list)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        height=400 * ((n_facets + 1) // 2),
        width=800)
    fig.show()

    return 

def hierarchical_hdbscan_umap():
    pass



# parameter tuning(GridSearch)
# hdbscan tuning
def tune_hdbscan_parameters(X, y=None, metric='euclidean'):
    param_grid = {
        "min_cluster_size":[5, 10, 20],
        "min_samples":[1, 5, 10],
        "cluster_selection_epsilon":[0.0, 0.1, 0.2]
    }
    
    if metric == "euclidean":
        matrix = X
        metric_name = "euclidean"
    if metric == "cosine":
        matrix = cosine_similarity(X)
        print(f"matrix shape: {matrix.shape}")
        metric_name = "precomputed"
    else:
        raise ValueError("metric should be 'euclidean' or 'cosine'")
    

    results = []
    key, values = zip(*param_grid.items())
    for vals in product(*values):
        params = dict(zip(key, vals))
        print(f"Testing parameters: {params}")
        
        clusterer = hdbscan.HDBSCAN(
            **params,
            core_dist_n_jobs=-1,
            metric=metric_name,
        )
        
        cluster_labels = clusterer.fit_predict(matrix)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_ratio = list(cluster_labels).count(-1) / len(cluster_labels)
        
        result = {
            "min_cluster_size": params["min_cluster_size"],
            "min_samples": params["min_samples"],
            "cluster_selection_epsilon": params["cluster_selection_epsilon"],
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "metric": metric
        }
        
        if y is not None and n_clusters > 1:
            ari = adjusted_rand_score(y, cluster_labels)
            nmi = normalized_mutual_info_score(y, cluster_labels)
        else:
            ari, nmi = None, None
        results.append({
            **params,
            "noise_ratio": noise_ratio,
            "n_clusters": n_clusters,
            "ARI": ari,
            "NMI": nmi
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=["noise_ratio", "ARI"], ascending=[True, False])

    return df_results


# dbscan tuning
def tune_dbscan_parameters(X, y=None, metric='euclidean'):
    param_grid = {
        "eps":[0.1, 0.2, 0.3, 0.4, 0.5],
        "min_samples":[5, 10, 20]
    }
    
    if metric == "euclidean":
        matrix = X
        metric_name = "euclidean"
    if metric == "cosine":
        matrix = cosine_similarity(X)
        print(f"matrix shape: {matrix.shape}")
        metric_name = "precomputed"
    else:
        raise ValueError("metric should be 'euclidean' or 'cosine'")
    

    results = []
    key, values = zip(*param_grid.items())
    for vals in product(*values):
        params = dict(zip(key, vals))
        print(f"Testing parameters: {params}")
        
        clusterer = DBSCAN(
            **params,
            metric=metric_name,
            n_jobs=-1
        )
        
        cluster_labels = clusterer.fit_predict(matrix)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_ratio = list(cluster_labels).count(-1) / len(cluster_labels)
        
        result = {
            "eps": params["eps"],
            "min_samples": params["min_samples"],
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "metric": metric
        }
        
        if y is not None and n_clusters > 1:
            ari = adjusted_rand_score(y, cluster_labels)
            nmi = normalized_mutual_info_score(y, cluster_labels)
        else:
            ari, nmi = None, None
        results.append({
            **params,
            "noise_ratio": noise_ratio,
            "n_clusters": n_clusters,
            "ARI": ari,
            "NMI": nmi
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=["noise_ratio", "ARI"], ascending=[True, False])

    return df_results

# 階層的クラスタリング(linkage) tuning
def tune_dbscan_parmeters(X, y=None, metric='euclidean'):
    """
    Z = linkage(X, method='ward', metric='euclidean') 
    """  
    pass



# hdbscan, tree structure

def run_hdbscan(X, min_cluster_size=5, min_samples=5, cluster_selection_epsilon=0.0, metric='euclidean', proj=None):
    if metric == "euclidean":
        matrix = X
        metric_name = "euclidean"
    elif metric == "cosine":
        matrix = cosine_similarity(X)
        print(f"matrix shape: {matrix.shape}")
        metric_name = "precomputed"
    else:
        raise ValueError("metric should be 'euclidean' or 'cosine'")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        core_dist_n_jobs=-1,
        metric=metric_name,
    )
    
    cluster_labels = clusterer.fit_predict(matrix)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_ratio = list(cluster_labels).count(-1) / len(cluster_labels)
    
    print(f"HDBSCAN results: n_clusters={n_clusters}, noise_ratio={noise_ratio:.4f}")

    # report
    clusterer.condensed_tree_.plot()
    clusterer.single_linkage_tree_.plot()

    # umap
    if proj is None:
        reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
        proj = reducer.fit_transform(X)
        print(f"UMAP projection shape: {proj.shape}")
    df = pd.DataFrame(proj, columns=["UMAP_1", "UMAP_2"])
    df['cluster'] = cluster_labels
    fig = px.scatter(
        df, 
        x='UMAP_1', 
        y='UMAP_2', 
        color='cluster',
        title='HDBSCAN Clustering with UMAP Projection',
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        height=600,
        width=600)
    fig.show()

    return clusterer