# ズーム機能 - 実装詳細

## バックエンド実装

### 追加されたメソッド

#### d3_data_manager.py

**`zoom_redraw(point_ids, dr_method, n_neighbors, min_dist, n_epochs)`**

位置: `src/d3-app/src/backend/services/d3_data_manager.py` 行 ~1100

```python
def zoom_redraw(
    self,
    point_ids: List[int],
    dr_method: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_epochs: int = 200
) -> Dict[str, Any]:
    """
    GPU UMAPで選択ポイントの2D座標を再計算
    メンタルマップ（空間的位置関係）を保持
    """
```

**ヘルパーメソッド:**

```python
@staticmethod
def _numpy_to_b64(array: np.ndarray) -> str:
    """NumPy配列をBase64文字列に変換"""

@staticmethod
def _b64_to_numpy(data_b64: str) -> np.ndarray:
    """Base64文字列をNumPy配列に変換"""
```

### API エンドポイント

#### main_d3.py

**`POST /api/zoom/redraw`**

位置: `src/d3-app/src/backend/main_d3.py` 行 ~383

```python
@app.post("/api/zoom/redraw", response_model=ZoomRedrawResponse)
async def zoom_redraw(request: ZoomRedrawRequest):
    """GPU UMAPで2D座標を再計算するエンドポイント"""
```

**リクエストモデル:**

```python
class ZoomRedrawRequest(BaseModel):
    point_ids: List[int]
    dr_method: str = "umap"
    n_neighbors: int = 15
    min_dist: float = 0.1
    n_epochs: int = 200
```

**レスポンスモデル:**

```python
class ZoomRedrawResponse(BaseModel):
    status: str  # "success" or "error"
    coordinates: Optional[str] = None  # Base64
    shape: Optional[List[int]] = None
    point_ids: Optional[List[int]] = None
    message: Optional[str] = None
```

---

## 実装の流れ

### 1. リクエスト受け取り
```python
@app.post("/api/zoom/redraw")
async def zoom_redraw(request: ZoomRedrawRequest):
    # point_ids, n_neighbors, etc. を抽出
    result = data_manager.zoom_redraw(...)
    return result
```

### 2. ファイル読込
```python
# 高次元ベクトルを読込
all_vectors = np.load(vectors_file)  # (N, D)
selected_vectors = all_vectors[point_ids]  # (N_selected, D)

# 現在の2D座標を取得
current_coords = self._embedding2d[point_ids]  # (N_selected, 2)
```

### 3. GPU 転送
```python
vectors_gpu = cp.asarray(selected_vectors, dtype=cp.float32)
init_gpu = cp.asarray(current_coords, dtype=cp.float32)
```

### 4. UMAP 実行
```python
umap_model = cuMLUMAP(
    n_components=2,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    metric="euclidean",
    random_state=42,
    init=init_gpu,  # ← メンタルマップ保持
    n_epochs=n_epochs
)
embedding_gpu = umap_model.fit_transform(vectors_gpu)
```

### 5. Base64 エンコード
```python
embedding_cpu = cp.asnumpy(embedding_gpu)
embedding_b64 = self._numpy_to_b64(embedding_cpu)
```

### 6. レスポンス返却
```python
return {
    "status": "success",
    "coordinates": embedding_b64,
    "shape": list(embedding_cpu.shape),
    "point_ids": point_ids
}
```

---

## エラーハンドリング

### GPU チェック
```python
try:
    import cupy as cp
    from cuml.manifold import UMAP as cuMLUMAP
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if not HAS_GPU:
    return {"status": "error", "message": "GPU UMAP not available"}
```

### ポイントID検証
```python
point_ids = [int(p) for p in point_ids]
max_id = self._embedding2d.shape[0] - 1
if any(p < 0 or p > max_id for p in point_ids):
    return {"status": "error", "message": f"Point IDs out of range"}
```

### ファイル検証
```python
vectors_file = self.base_path / ... / "vector.npy"
if not vectors_file.exists():
    return {"status": "error", "message": "Vector file not found"}
```

---

## テストスクリプト

**ファイル**: `src/d3-app/src/backend/test_zoom_api.py`

### テスト 1: 小規模ズーム
```python
point_ids = list(range(0, 50))  # 最初の50ポイント
```

### テスト 2: 中規模ズーム
```python
point_ids = list(range(100, 300))  # 100-299ポイント
```

### テスト 3: エラーハンドリング
```python
point_ids = [999999, 1000000]  # 無効なID
```

### テスト 4: 未サポートメソッド
```python
"dr_method": "tsne"  # サポート外
```

---

## 修正内容の詳細

### d3_data_manager.py の変更

**追加行数**: +226

**変更内容:**
1. インポート追加 (+20行)
   - `base64`, `io`
   - `cupy`, `cuml.manifold.UMAP` (GPU ライブラリ)

2. ヘルパーメソッド (+40行)
   - `_numpy_to_b64()`
   - `_b64_to_numpy()`

3. メインメソッド (+166行)
   - `zoom_redraw()`: GPU UMAP 実装とエラーハンドリング

### main_d3.py の変更

**追加行数**: +43

**変更内容:**
1. Pydantic モデル (+16行)
   - `ZoomRedrawRequest`
   - `ZoomRedrawResponse`

2. API エンドポイント (+27行)
   - `@app.post("/api/zoom/redraw")`
   - エラーハンドリング

---

## 依存関係

### 必須
- `numpy >= 1.19.0`
- `fastapi >= 0.68.0`
- `pydantic >= 1.8.0`

### GPU（オプション但し推奨）
- `cupy >= 9.0.0`
- `cuml >= 21.0.0`
- CUDA >= 11.0

---

## 次のドキュメント

- **FRONTEND_GUIDE.md** - フロントエンド実装
- **CONFIGURATION.md** - パラメータ調整
- **TROUBLESHOOTING.md** - 問題解決
