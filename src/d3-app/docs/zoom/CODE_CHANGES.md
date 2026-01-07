# ズーム機能 - コード変更詳細

## 修正ファイル一覧

| ファイル | 変更 | 行数 |
|---------|------|------|
| `services/d3_data_manager.py` | 追加 | +226 |
| `main_d3.py` | 追加 | +43 |
| `test_zoom_api.py` | 新規 | 200+ |

---

## d3_data_manager.py の変更

### インポート追加（~20行）

```python
import base64
import io

try:
    import cupy as cp
    from cuml.manifold import UMAP as cuMLUMAP
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
```

### ヘルパーメソッド追加（~40行）

```python
@staticmethod
def _b64_to_numpy(data_b64: str) -> np.ndarray:
    """Base64文字列からNumPy配列にデコード"""
    decoded = base64.b64decode(data_b64)
    return np.load(io.BytesIO(decoded))

@staticmethod
def _numpy_to_b64(array: np.ndarray) -> str:
    """NumPy配列をBase64文字列にエンコード"""
    buff = io.BytesIO()
    np.save(buff, array, allow_pickle=False)
    return base64.b64encode(buff.getvalue()).decode('utf-8')
```

### メインメソッド追加（~166行）

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
    選択ポイントの2D座標をGPU UMAPで再計算
    メンタルマップ（空間的位置関係）を保持
    """
    if not HAS_GPU:
        return {
            "status": "error",
            "message": "GPU UMAP not available. Install cupy and cuml."
        }
    
    if dr_method != "umap":
        return {
            "status": "error",
            "message": f"Only 'umap' is supported, got '{dr_method}'"
        }
    
    try:
        # ポイントID検証
        point_ids = [int(p) for p in point_ids]
        if not self._embedding2d is None:
            max_id = self._embedding2d.shape[0] - 1
            if any(p < 0 or p > max_id for p in point_ids):
                return {
                    "status": "error",
                    "message": f"Point IDs out of range [0, {max_id}]"
                }
        
        # ベクトルファイル読込
        vectors_file = self.base_path / self.datasets_config["default"]["data_path"] / "vector.npy"
        if not vectors_file.exists():
            return {
                "status": "error",
                "message": f"Vector file not found at {vectors_file}"
            }
        
        # 高次元ベクトルを読込
        all_vectors = np.load(vectors_file)
        selected_vectors = all_vectors[point_ids]
        
        print(f"✓ Loaded {len(point_ids)} vectors: {selected_vectors.shape}")
        
        # 現在の2D座標を取得（メンタルマップ保持用）
        current_coords = self._embedding2d[point_ids]
        print(f"✓ Extracted initial positions: {current_coords.shape}")
        
        # GPU転送
        vectors_gpu = cp.asarray(selected_vectors, dtype=cp.float32)
        init_gpu = cp.asarray(current_coords, dtype=cp.float32)
        
        print(f"✓ GPU ready: vectors {vectors_gpu.shape}, init {init_gpu.shape}")
        
        # UMAP実行
        umap_model = cuMLUMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(point_ids) - 1),
            min_dist=min_dist,
            metric="euclidean",
            random_state=42,
            init=init_gpu,  # ← メンタルマップ保持の鍵
            n_epochs=n_epochs,
            verbose=True
        )
        
        # GPU計算実行
        embedding_gpu = umap_model.fit_transform(vectors_gpu)
        cp.cuda.runtime.deviceSynchronize()
        
        # CPU転送
        embedding_cpu = cp.asnumpy(embedding_gpu)
        print(f"✓ UMAP complete: {embedding_cpu.shape}")
        
        # Base64エンコード
        embedding_b64 = self._numpy_to_b64(embedding_cpu)
        
        return {
            "status": "success",
            "coordinates": embedding_b64,
            "shape": list(embedding_cpu.shape),
            "point_ids": point_ids
        }
    
    except ValueError as e:
        print(f"ValueError: {e}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Internal error: {str(e)}"}
```

---

## main_d3.py の変更

### Pydanticモデル追加（~16行）

```python
class ZoomRedrawRequest(BaseModel):
    """ズーム再描画のリクエストモデル"""
    point_ids: List[int]
    dr_method: str = "umap"
    n_neighbors: int = 15
    min_dist: float = 0.1
    n_epochs: int = 200


class ZoomRedrawResponse(BaseModel):
    """ズーム再描画のレスポンスモデル"""
    status: str
    coordinates: Optional[str] = None
    shape: Optional[List[int]] = None
    point_ids: Optional[List[int]] = None
    message: Optional[str] = None
```

### APIエンドポイント追加（~27行）

```python
@app.post("/api/zoom/redraw", response_model=ZoomRedrawResponse)
async def zoom_redraw(request: ZoomRedrawRequest):
    """
    GPU UMAPで2D座標を再計算
    メンタルマップ（空間的位置関係）を保持
    """
    try:
        result = data_manager.zoom_redraw(
            point_ids=request.point_ids,
            dr_method=request.dr_method,
            n_neighbors=request.n_neighbors,
            min_dist=request.min_dist,
            n_epochs=request.n_epochs
        )
        return result
    except Exception as e:
        print(f"Error in zoom_redraw endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
```

---

## test_zoom_api.py（新規ファイル）

### テストケース

**テスト 1: 小規模ズーム（50ポイント）**
- 概要: 最初の50ポイントをズーム
- 期待結果: 5-8秒で成功

**テスト 2: 中規模ズーム（200ポイント）**
- 概要: ポイント 100-299 をズーム
- 期待結果: 15-25秒で成功

**テスト 3: エラーハンドリング（無効なID）**
- 概要: 存在しないポイントID
- 期待結果: エラーメッセージ

**テスト 4: 未サポートメソッド**
- 概要: dr_method = "tsne"
- 期待結果: エラーメッセージ

---

## API仕様

### Request
```
POST /api/zoom/redraw
Content-Type: application/json

{
  "point_ids": [integer],
  "dr_method": "umap",
  "n_neighbors": 15,
  "min_dist": 0.1,
  "n_epochs": 200
}
```

### Response (Success)
```json
{
  "status": "success",
  "coordinates": "gAN9cQA...",
  "shape": [N, 2],
  "point_ids": [0, 1, 2, ...]
}
```

### Response (Error)
```json
{
  "status": "error",
  "message": "Error description",
  "coordinates": null,
  "shape": null,
  "point_ids": null
}
```

---

## 実装のポイント

### 1. メンタルマップ保持
```python
init=init_gpu,  # ← これが重要
```
現在の2D座標をUMAP初期値として使用することで、ズーム後も空間的位置関係が保持される。

### 2. Base64エンコード
```python
buff = io.BytesIO()
np.save(buff, array, allow_pickle=False)
b64 = base64.b64encode(buff.getvalue()).decode('utf-8')
```
バイナリ配列をJSON互換形式に変換。

### 3. エラーハンドリング
```python
if not HAS_GPU:
    return {"status": "error", "message": "..."}
```
GPUがない環境でも適切なエラーメッセージを返す。

### 4. 型安全性
```python
class ZoomRedrawRequest(BaseModel):
    point_ids: List[int]
```
Pydanticで自動検証。

---

## 動作確認

### 最小限のテスト
```bash
cd src/d3-app/src/backend

# API起動
uvicorn main_d3:app --port 8000

# 別ターミナルでテスト
curl -X POST http://localhost:8000/api/zoom/redraw \
  -H "Content-Type: application/json" \
  -d '{"point_ids": [0, 1, 2, 3, 4]}'
```

### 完全なテスト
```bash
python test_zoom_api.py
```

---

## ファイルパス参照

- 実装: `src/d3-app/src/backend/services/d3_data_manager.py`
- API: `src/d3-app/src/backend/main_d3.py`
- テスト: `src/d3-app/src/backend/test_zoom_api.py`
- ベクトル: `src/d3-app/data/vector.npy`
- 座標: `src/d3-app/data/projection.npy`
