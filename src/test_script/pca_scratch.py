import numpy as np

def pca(X, k):
    # 1. データの中心化
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # 2. 共分散行列を計算
    S = np.cov(X_centered, rowvar=False)

    # 3. 固有値と固有ベクトルを計算
    lam, h = np.linalg.eigh(S)

    # 4. 固有値を降順にソートし、対応する固有ベクトルも並べ替える
    index = np.argsort(lam)[::-1]
    sorted_lam = lam[index]
    sorted_h = h[:, index]

    # 5. 最も大きいk個の固有ベクトルを選ぶ
    selected_h = sorted_h[:, :k]         
    
    # データを低次元空間に射影
    X_pca = X_centered @ selected_h
    
    return X_pca
