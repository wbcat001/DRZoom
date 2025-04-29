import os
import pandas as pd
import numpy as np
import torch
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import time

# Start timing
start_time = time.time()

# File paths
metadata_file = 'data/arxiv/metadata_0.json'
embedding_file = 'data/arxiv/abstract_embeddings.pt'

# Check if metadata file exists
if not os.path.exists(metadata_file):
    print(f"File not found: {metadata_file}")
    exit(1)

# Check if embedding file exists
if not os.path.exists(embedding_file):
    print(f"File not found: {embedding_file}")
    exit(1)

# Load metadata
metadata_df = pd.read_json(metadata_file, lines=True)

# Check if 'categories' column exists
if 'categories' not in metadata_df.columns:
    print("Error: 'categories' column not found in metadata file.")
    exit(1)

# Load embeddings
embeddings = torch.load(embedding_file)
ids, vectors = zip(*embeddings.items())
vectors = np.array(vectors)

# Merge metadata with embeddings
metadata_df = metadata_df[metadata_df['id'].isin(ids)]
metadata_df['embedding_index'] = metadata_df['id'].apply(lambda x: ids.index(x))
metadata_df = metadata_df.sort_values('embedding_index')

# Extract categories
metadata_df['categories'] = metadata_df['categories'].str.split().str[0]  # Use the first category for simplicity

# Measure execution time for PCA
start_pca = time.time()
pca = PCA(n_components=2)
pca_result = pca.fit_transform(vectors)
metadata_df['PCA1'] = pca_result[:, 0]
metadata_df['PCA2'] = pca_result[:, 1]
end_pca = time.time()
print(f"PCA execution time: {end_pca - start_pca:.2f} seconds")

# Plot PCA with categories
fig_pca = px.scatter(
    metadata_df,
    x='PCA1',
    y='PCA2',
    color='categories',
    title="PCA Projection with Categories",
    labels={'PCA1': 'PCA1', 'PCA2': 'PCA2'}
)
fig_pca.show()

# Measure execution time for t-SNE
start_tsne = time.time()
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(vectors)
metadata_df['tSNE1'] = tsne_result[:, 0]
metadata_df['tSNE2'] = tsne_result[:, 1]
end_tsne = time.time()
print(f"t-SNE execution time: {end_tsne - start_tsne:.2f} seconds")

# Plot t-SNE with categories
fig_tsne = px.scatter(
    metadata_df,
    x='tSNE1',
    y='tSNE2',
    color='categories',
    title="t-SNE Projection with Categories",
    labels={'tSNE1': 't-SNE1', 'tSNE2': 't-SNE2'}
)
fig_tsne.show()

# Measure execution time for UMAP
start_umap = time.time()
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_model.fit_transform(vectors)
metadata_df['UMAP1'] = umap_result[:, 0]
metadata_df['UMAP2'] = umap_result[:, 1]
end_umap = time.time()
print(f"UMAP execution time: {end_umap - start_umap:.2f} seconds")

# Plot UMAP with categories
fig_umap = px.scatter(
    metadata_df,
    x='UMAP1',
    y='UMAP2',
    color='categories',
    title="UMAP Projection with Categories",
    labels={'UMAP1': 'UMAP1', 'UMAP2': 'UMAP2'}
)
fig_umap.show()

# End timing
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")