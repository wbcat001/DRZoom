
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ペアワイズ距離の関係確認
def plot_distance_relationship(X_high, X_low):
    dist_high = pairwise_distances(X_high)
    dist_low = pairwise_distances(X_low)

    i_upper = np.triu_indices_from(dist_high, k=1)
    dist_high_flat = dist_high[i_upper]
    dist_low_flat = dist_low[i_upper]

    # bin
    n_bins = 30
    bins = np.linspace(0, np.max(dist_high_flat), n_bins + 1)
    bin_indices = np.digitize(dist_high_flat, bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    df = pd.DataFrame({
        'dist_high': dist_high_flat,
        'dist_low': dist_low_flat,
        'bin': bin_indices
    })
    df = df[df["bin"].between(0, n_bins-1)]
    df["bin_center"] = df["bin"].map(lambda x: bin_centers[x])

    # histogram
    fig_hist = px.histogram(
        df, 
        x='dist_high', 
        nbins=n_bins,
        title='Histogram of High-Dimensional Distances',
        labels={'dist_high': 'High-Dimensional Distance', 'count': 'Frequency'}
    ).show()

    # box plot
    fig_box = px.box(
        df, 
        x='bin_center', 
        y='dist_low',
        title='Box Plot of Low-Dimensional Distances by High-Dimensional Distance Bins',
        labels={'bin_center': 'High-Dimensional Distance Bins', 'dist_low': 'Low-Dimensional Distance'}
    ).show()
    fig_shapard = px.scatter(
        df, 
        x='dist_high', 
        y='dist_low',
        title='Shepard Diagram',
        labels={'dist_high': 'High-Dimensional Distance', 'dist_low': 'Low-Dimensional Distance'},
        opacity=0.5
    ).show()
    
    # 相関
    correlation = np.corrcoef(dist_high_flat, dist_low_flat)[0, 1]
    print(f"Correlation between high-dimensional and low-dimensional distances: {correlation:.4f}")