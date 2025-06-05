# mnistデータセットに関して、PCAによる可視化を行う(Plotly)

# code
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import plotly.express as px
from sklearn.datasets import load_digits

def plot_pca_mnist():
    # MNISTデータをダウンロード
    mnist = load_digits()
    X, y = mnist.data, mnist.target

    # データの大きさをプロット

    # データを標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCAによる次元削減
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # データフレームの作成
    df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    df['target'] = y

    # strに
    df['target'] = df['target'].astype(str)

    # Plotlyで散布図を作成
    fig = px.scatter(df, x='PC1', y='PC2', color='target', title='PCA of MNIST Dataset')

    # layoutの設定(正方形、囲う線)
    fig.update_layout(
        width=800,
        height=800,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    fig.show()

if __name__ == "__main__":
    plot_pca_mnist()