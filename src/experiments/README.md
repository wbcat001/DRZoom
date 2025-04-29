# DRZoom 実験ディレクトリ

このディレクトリには、DRZoomのさまざまな実験結果や設定が含まれています。

## 実験タイプ

### 1. 次元削減アルゴリズム比較

様々な次元削減アルゴリズムをデータセットに適用し、その結果を比較します。

- **実装ファイル**: [dimension_reduction_comparison.py](./dimension_reduction_comparison.py)
- **使用方法**:

```python
from dimension_reduction_comparison import run_experiment

# データパスと出力次元を指定
data_path = "../../data/text/harrypotter1/paragraph_embedding.pkl"
experiment = run_experiment(data_path, output_dim=2)
```

- **サポートされているアルゴリズム**:
  - PCA
  - Kernel PCA (RBF, Polynomial)
  - Truncated SVD
  - t-SNE
  - MDS
  - Isomap
  - LLE (Locally Linear Embedding)
  - UMAP

- **評価指標**:
  - 実行時間
  - 高次元距離と低次元距離の相関

## 実験結果の保存構造

各実験は以下のディレクトリ構造で保存されます：

```
experiments/
├── results/          # 集計された実験結果
└── snapshots/        # 各実験のスナップショット
    └── YYYYMMDD_HHMMSS/  # タイムスタンプ付きの実験ディレクトリ
        ├── comparison_scatter_*.png  # 散布図比較
        ├── time_comparison_*.png     # 実行時間比較
        ├── results_*.pkl             # 実験結果データ
        ├── metadata_*.pkl            # 実験メタデータ
        └── report_*.md               # 自動生成された実験レポート
```

## 新しい実験を追加する方法

1. 実験用のPythonスクリプトを作成します
2. 適切な結果保存とレポート生成機能を実装します
3. このREADMEファイルに実験の概要を追加します

## 今後の実験アイデア

- 様々なデータセットに対するアルゴリズム性能の比較
- ハイパーパラメータの影響分析
- データサイズとパフォーマンスの関係
- 前処理手法の影響