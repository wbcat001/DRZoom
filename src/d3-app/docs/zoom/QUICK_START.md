# ズーム機能 - クイックスタート

## インストール

### GPU サポート（推奨）
```bash
conda create -n drzoom -c rapids -c conda-forge cuml cupy cudatoolkit=11.2
conda activate drzoom
pip install fastapi uvicorn pydantic numpy
```

### CPU のみ
```bash
pip install fastapi uvicorn pydantic numpy
```

---

## テスト実行

### 1. バックエンドを起動
```bash
cd src/d3-app/src/backend
uvicorn main_d3:app --host 0.0.0.0 --port 8000
```

出力例：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. テストスクリプト実行
別のターミナルで：
```bash
cd src/d3-app/src/backend
python test_zoom_api.py
```

出力例：
```
============================================================
Testing Zoom Redraw API
============================================================

[Test 1] Zoom with 50 points
Status Code: 200
Status: success
✓ Received coordinates with shape: (50, 2)
  Coordinates range: x=[-2.34, 3.21], y=[-1.89, 2.45]
```

### 3. 手動テスト（curl）
```bash
curl -X POST http://localhost:8000/api/zoom/redraw \
  -H "Content-Type: application/json" \
  -d '{
    "point_ids": [0, 1, 2, 3, 4],
    "dr_method": "umap"
  }'
```

期待される出力：
```json
{
  "status": "success",
  "coordinates": "gAN9cQA...",
  "shape": [5, 2],
  "point_ids": [0, 1, 2, 3, 4]
}
```

---

## トラブルシューティング

### エラー: "GPU UMAP not available"

**原因**: CuPy または cuML がインストールされていない

**解決策**:
```bash
conda install -c rapids -c conda-forge cuml cupy
```

### エラー: "Vector file not found"

**原因**: `vector.npy` ファイルが存在しない

**解決策**:
```bash
ls -la src/d3-app/data/vector.npy
# ファイルが存在することを確認
```

### エラー: "Connection refused"

**原因**: バックエンドが起動していない

**解決策**:
```bash
cd src/d3-app/src/backend
uvicorn main_d3:app --port 8000
```

---

## API パラメータの意味

| パラメータ | デフォルト | 説明 |
|-----------|---------|------|
| `point_ids` | 必須 | ズーム対象のポイント番号リスト |
| `dr_method` | "umap" | 次元削減手法（現在は umap のみ） |
| `n_neighbors` | 15 | ローカル構造のバランス（小: 局所的, 大: 大域的） |
| `min_dist` | 0.1 | ポイント間の最小距離（小: 密集, 大: 分散） |
| `n_epochs` | 200 | 最適化のイテレーション数（小: 高速, 大: 高品質） |

### パラメータのチューニング

**ポイント数が少ない場合（<100）**:
```json
{
  "n_neighbors": 20,
  "n_epochs": 300
}
```

**ポイント数が多い場合（>1000）**:
```json
{
  "n_neighbors": 10,
  "n_epochs": 100
}
```

**より密集したクラスタ**:
```json
{
  "min_dist": 0.2
}
```

---

## 次のステップ

- **ARCHITECTURE.md** - システムの仕組みを理解したい場合
- **FRONTEND_GUIDE.md** - フロントエンド実装を始めたい場合
- **TROUBLESHOOTING.md** - 問題が発生した場合
