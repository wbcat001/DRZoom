import numpy as np

def custom_pca(X, k, Z_old=None, lambda_=0.1, max_iter=100, lr=0.01):
    """
    レイアウト変化を抑えつつPCAを実行する関数

    Parameters:
    - X: (n_samples, n_features) 入力データ
    - k: 次元数
    - Z_old: (n_samples, k) 旧レイアウト（Noneなら通常のPCA）
    - lambda_: レイアウト変更抑制の重み
    - max_iter: 最適化のイテレーション数
    - lr: 学習率（射影行列の更新ステップ）

    Returns:
    - Z: (n_samples, k) 低次元データ
    - H: (n_features, k) 射影行列
    """
    # 1. データの中心化
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # 2. 共分散行列を計算
    S = np.cov(X_centered, rowvar=False)

    # 3. 固有値・固有ベクトルを計算
    lam, H = np.linalg.eigh(S)

    # 4. 固有値を降順ソート
    index = np.argsort(lam)[::-1]
    H = H[:, index[:k]]  # k個の固有ベクトルを取得

    # 5. 最適化（レイアウト変化抑制）
    for _ in range(max_iter):
        # 低次元空間への射影
        Z = X_centered @ H

        # PCAの再構成誤差
        X_reconstructed = Z @ H.T
        loss_pca = np.mean((X_centered - X_reconstructed) ** 2)

        # レイアウト変更抑制
        if Z_old is not None:
            loss_layout = np.mean((Z - Z_old) ** 2)
        else:
            loss_layout = 0

        # 総合損失
        loss = loss_pca + lambda_ * loss_layout

        # 勾配計算（Hを更新）
        grad_H = -2 * X_centered.T @ (X_centered @ H)  # PCAの勾配
        if Z_old is not None:
            grad_H += 2 * lambda_ * (X_centered.T @ (Z - Z_old) @ H.T).T  # レイアウト抑制項の勾配

        # 勾配降下法で更新
        H -= lr * grad_H

        # 直交化（射影行列を再正規化）
        U, _, Vt = np.linalg.svd(H, full_matrices=False)
        H = U @ Vt  # 直交行列に修正

    # 最終的な低次元データ
    Z = X_centered @ H
    return Z, H

# データ例
np.random.seed(0)
X = np.random.rand(100, 5)  # 100サンプル, 5次元

# 旧レイアウト（適当にPCAで求めたものを使う）
Z_old, _ = custom_pca(X, k=2, lambda_=0.0)

# カスタムPCAを適用（レイアウト変化を抑制）
Z_new, _ = custom_pca(X, k=2, Z_old=Z_old, lambda_=0.1)

print(Z_new[:5])  # 新しいレイアウトの一部を表示
