import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cuml.cluster import HDBSCAN as cuML_HDBSCAN
from hdbscan.plots import CondensedTree as hdbscan_CondensedTree
from sklearn.datasets import make_blobs
import os
import sys

def plot_cuml_hdbscan_condensed_tree(output_filename='cuml_hdbscan_condensed_tree_plot.png'):
    """
    RAPIDS cuML HDBSCANの結果からCondensed Treeを計算し、
    hdbscanのプロット機能（Matplotlib）を使用して描画・保存する関数。
    """
    print("--- 1. データ準備とcuML HDBSCANの実行 ---")
    
    # データの準備 (CuPyアレイ)
    try:
        data_cpu, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.7, random_state=42)
        # GPUに転送
        data_gpu = cp.asarray(data_cpu, dtype=np.float32)
    except Exception as e:
        print(f"データの準備またはCuPyへの変換中にエラーが発生しました: {e}")
        print("RAPIDS環境が正しくセットアップされているか確認してください。")
        return

    # cuML HDBSCANの実行
    try:
        # min_cluster_sizeを小さくして、より多くの枝分かれを生成
        cuml_clusterer = cuML_HDBSCAN(min_cluster_size=10).fit(data_gpu)
    except Exception as e:
        print(f"cuML HDBSCANの実行中にエラーが発生しました: {e}")
        print("GPUドライバやcuMLの依存関係を確認してください。")
        return

    print("cuML HDBSCAN実行完了。")
    
    # 2. 必要な結果をCPU (NumPy) に取り出す
    # to_numpy() で既に NumPy (CPU) に変換されているため、.get() は不要
    try:
        raw_tree_np = cuml_clusterer.condensed_tree_.to_numpy()
        # ラベルは CuPy アレイなので、.get() で NumPy に変換
        labels_np = cuml_clusterer.labels_.get()
    except Exception as e:
        print(f"結果のCPUへの転送中にエラーが発生しました: {e}")
        return

    print("Condensed TreeデータとラベルをCPUに転送完了。")
    
    # 3. hdbscanライブラリのCondensedTreeオブジェクトを再構築
    try:
        hdbscan_tree = hdbscan_CondensedTree(
            raw_tree_np,
            labels_np
        )
    except Exception as e:
        print(f"hdbscan.plots.CondensedTreeの再構築中にエラーが発生しました: {e}")
        return

    print("hdbscan CondensedTreeオブジェクト再構築完了。")
    
    # 4. デフォルトのplot()メソッドで描画・保存
    print(f"--- 4. デンドログラムの描画とファイル保存 ({output_filename}) ---")
    try:
        # MatplotlibのFigureとAxesを作成
        fig, ax = plt.subplots(figsize=(14, 7))

        # hdbscanのplot()メソッドを使用
        hdbscan_tree.plot(
            # select_clusters=True,   # 選択されたクラスタをハイライト
            # label_clusters=True,    # クラスタ番号を表示
            # log_size=True,          # 幅を対数スケールで表示
            # cmap='viridis',         # カラーマップ
            # axis=ax                 # 作成したAxesオブジェクトに描画
        )
        
        # グラフの調整
        ax.set_title("Condensed Tree from cuML HDBSCAN (Plotted with hdbscan/Matplotlib)", fontsize=14)

        # グラフの保存
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close(fig) # メモリを解放

        print(f"✅ 成功: MatplotlibグラフをPNGファイルに保存しました: {os.path.abspath(output_filename)}")

    except ImportError:
        print("\n⚠️ Matplotlibまたはその依存関係が見つかりません。")
        print("プロット機能には 'matplotlib' が必要です。`pip install matplotlib` を実行してください。")
    except Exception as e:
        print(f"プロットまたはファイル保存中に予期せぬエラーが発生しました: {e}")
        
if __name__ == '__main__':
    # スクリプトの実行
    plot_cuml_hdbscan_condensed_tree()