# D3 App - Data Configuration Guide

このドキュメントでは、実際のデータを使用する際に変更が必要な設定箇所を説明します。

## 現在の状態

現在、アプリケーションは**モックデータモード**で動作しています。実際のデータファイルが存在しないため、開発用の合成データを生成して表示します。

## 実データへの切り替え方法

### 1. データファイルの準備

以下のファイルを指定のディレクトリに配置してください：

```
experiments/<your_data_path>/
├── projection.npy                # 必須: (N, 2) UMAP 2D座標
├── word.npy                      # 必須: (N,) 単語ラベル
├── hdbscan_label.npy             # 必須: (N,) クラスタラベル
├── cluster_to_label.csv          # 必須: クラスタ代表単語
├── vector.npy                    # 必須: (N, 300) 高次元ベクトル
├── condensed_tree_object.pkl     # オプション: HDBSCAN凝縮木
└── similarity_*.pkl              # オプション: クラスタ類似度行列
```

### 2. バックエンド設定の変更

**ファイル**: `src/backend/services/d3_data_manager.py`

#### (1) データパスの設定

```python
def _load_configuration(self):
    # ============================================================
    # DATA PATH CONFIGURATION - CHANGE THIS TO YOUR DATA LOCATION
    # ============================================================
    self.datasets_config = {
        "default": {
            "name": "RAPIDS HDBSCAN Result",
            "description": "HDBSCAN clustering result from RAPIDS GPU acceleration",
            "data_path": "18_rapids/result/20251203_053328",  # ← ここを変更
            "point_count": 100000,
            "cluster_count": 0,
            "dr_methods": ["umap", "tsne", "pca"]
        }
    }
```

`data_path`を実際のデータが格納されているディレクトリに変更してください（`experiments/`からの相対パス）。

#### (2) モックモードの無効化

同じメソッド内で：

```python
# Enable mock data mode when files don't exist
self.use_mock_data = True  # ← これを False に変更
```

### 3. データファイルの詳細仕様

#### projection.npy (必須)

UMAP等で計算された2次元投影座標。

- **形式**: NumPy配列
- **Shape**: (N, 2)
- **Dtype**: float64 or float32

作成例：
```python
import numpy as np
np.save('projection.npy', umap_coords)  # shape: (N, 2)
```

#### word.npy (必須)

各データポイントの単語ラベル。

- **形式**: NumPy配列
- **Shape**: (N,)
- **Dtype**: object (文字列)

作成例：
```python
import numpy as np
np.save('word.npy', word_labels)  # shape: (N,)
```

#### hdbscan_label.npy (必須)

HDBSCANで計算されたクラスタラベル。

- **形式**: NumPy配列
- **Shape**: (N,)
- **Dtype**: int32 or int64
- **値**: -1 (ノイズ), 0以上 (クラスタID)

作成例：
```python
import numpy as np
from hdbscan import HDBSCAN

clusterer = HDBSCAN(min_cluster_size=50)
labels = clusterer.fit_predict(vectors)
np.save('hdbscan_label.npy', labels)  # shape: (N,)
```

#### cluster_to_label.csv (必須)

クラスタの代表単語を定義するCSVファイル。

形式：
```csv
cluster_id,representative_label,word1,word2,word3,word4,word5,word6,word7,word8,word9,word10
0,machine learning,machine,learning,algorithm,neural,network,deep,model,train,data,feature
1,natural language,language,nlp,text,word,sentence,parse,semantic,syntax,corpus,token
2,computer vision,image,vision,pixel,cnn,detection,segmentation,feature,object,recognition,visual
```

フィールド：
- `cluster_id`: int - クラスタID
- `representative_label`: str - クラスタの代表ラベル
- `word1`～`word10`: str - 代表単語10個

#### vector.npy (必須)

高次元ベクトル（類似度計算用）。

- **形式**: NumPy配列
- **Shape**: (N, D) - 例: (N, 300)
- **Dtype**: float32 or float64

作成例：
```python
import numpy as np
np.save('vector.npy', embeddings)  # shape: (N, 300)
```

#### condensed_tree_object.pkl (オプション)

HDBSCANライブラリの`CondensedTree`オブジェクト。デンドログラムの詳細表示に使用。

作成例：
```python
import pickle
from hdbscan import HDBSCAN

clusterer = HDBSCAN(min_cluster_size=50)
clusterer.fit(vectors)

with open('condensed_tree_object.pkl', 'wb') as f:
    pickle.dump(clusterer.condensed_tree_, f)
```

**注意**: このファイルがない場合、クラスタラベルから簡易デンドログラムを生成します。

#### similarity_*.pkl (オプション)

クラスタ間類似度行列を格納したpickleファイル。クラスタヒートマップの表示に使用。

ファイル名パターン：
- `similarity_kl_divergence.pkl`
- `similarity_bhattacharyya_coefficient.pkl`
- `similarity_mahalanobis_distance.pkl`

各ファイルの内容（辞書形式）：
```python
{
    'matrix': np.ndarray,      # (n_clusters, n_clusters) 類似度行列
    'cluster_ids': List[int],  # クラスタIDのリスト
    'metric': str              # メトリック名
}
```

作成方法：

**自動生成（推奨）**:
```bash
cd src/backend/make_data
python similarity.py --data_dir <path_to_data> --output_dir <path_to_data>
```

例：
```bash
python similarity.py \
  --data_dir ../../../../experiments/18_rapids/result/20251203_053328 \
  --output_dir ../../../../experiments/18_rapids/result/20251203_053328 \
  --metrics kl_divergence bhattacharyya_coefficient mahalanobis_distance \
  --min_cluster_size 10
```

**手動作成**:
```python
import pickle
import numpy as np

data = {
    'matrix': similarity_matrix,  # (n_clusters, n_clusters)
    'cluster_ids': cluster_ids,   # [0, 1, 2, ...]
    'metric': 'kl_divergence'
}
with open('similarity_kl_divergence.pkl', 'wb') as f:
    pickle.dump(data, f)
```

### 4. フロントエンド設定の確認

**ファイル**: `src/d3-app/.env.local`

APIエンドポイントの設定を確認：

```env
VITE_API_URL=http://localhost:8000/api
```

開発サーバーのポートが異なる場合は適宜変更してください。

### 5. 動作確認

#### バックエンド起動

```bash
cd src/backend
uvicorn main_d3:app --reload
```

または

```bash
cd src
uvicorn backend.main_d3:app --reload
```

#### APIテスト

```bash
curl http://localhost:8000/api/initial_data?dataset=default&dr_method=umap
```

正常な場合、実データのJSONレスポンスが返ります。

#### フロントエンド起動

```bash
cd src/d3-app
npm install
npm run dev
```

ブラウザで `http://localhost:5173` を開き、データが正しく表示されることを確認してください。

## トラブルシューティング

### データが読み込まれない

1. ファイルパスが正しいか確認
2. `use_mock_data = False`に設定されているか確認
3. バックエンドのログで`FileNotFoundError`がないか確認

### データ形式エラー

1. npzファイルに`embedding`と`words`キーが存在するか確認：
   ```python
   import numpy as np
   data = np.load('data.npz')
   print(data.files)  # ['embedding', 'words']
   ```
2. pickleファイルがHDBSCANの`CondensedTree`オブジェクトか確認

### モックデータが表示され続ける

1. `use_mock_data`フラグを確認
2. ファイルが実際に存在するか確認
3. サーバーを再起動（キャッシュのクリア）

## まとめ：変更が必要なファイルと箇所

| ファイル | 変更箇所 | 内容 |
|---------|---------|------|
| `src/backend/services/d3_data_manager.py` | `_load_configuration()` | `data_path`を実データパスに変更 |
| 同上 | `_load_configuration()` | `use_mock_data = False`に変更 |
| 実験ディレクトリ | 指定パス | 必要なデータファイルを配置 |
| `src/d3-app/.env.local` | `VITE_API_URL` | API URLが正しいか確認（通常は変更不要） |

実データへの切り替え後、`src/backend/services/mock_data_generator.py`は削除しても構いません（ただし、開発時のフォールバックとして残しておくことも可能です）。
