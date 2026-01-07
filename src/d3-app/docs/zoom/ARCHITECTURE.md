# ズーム機能 - アーキテクチャ

## システム構成図

```
┌─────────────────────────────────────────────────────────────────────┐
│                         フロントエンド (React)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  DRVisualization.tsx                                                 │
│  └─ "ズーム" ボタン ─────┐                                            │
│                         │                                            │
│  Fetch.ts               │                                            │
│  └─ fetchZoomRedraw()   │                                            │
│     └─ Base64 デコード   │                                            │
└─────────────────────────┼───────────────────────────────────────────┘
                          │
                    HTTP POST
                          │
┌─────────────────────────▼───────────────────────────────────────────┐
│                    バックエンド (FastAPI)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  main_d3.py                                                          │
│  └─ POST /api/zoom/redraw endpoint                                  │
│     └─ ZoomRedrawRequest を受け取る                                  │
│     └─ data_manager.zoom_redraw() を呼び出す                         │
│                         │                                            │
│  d3_data_manager.py    │                                            │
│  └─ zoom_redraw()      │                                            │
│     ├─ vector.npy から高次元ベクトル読込 (N, D)                     │
│     ├─ projection.npy から現在の2D座標読込 (N, 2)                   │
│     ├─ GPU転送 (CuPy)                                               │
│     ├─ GPU UMAP実行                                                 │
│     │  └─ init = current_coords (メンタルマップ保持)                │
│     ├─ CPU転送                                                      │
│     └─ Base64 エンコード                                            │
│                         │                                            │
│  ZoomRedrawResponse で返却                                           │
│  └─ coordinates (Base64), shape, point_ids                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## データフロー

### 1. ユーザーが「ズーム」ボタンをクリック
```
getZoomTargetPoints(data) 
  → [point_id_0, point_id_1, ..., point_id_N]
```

### 2. API リクエスト送信
```json
{
  "point_ids": [0, 1, 2, ...],
  "dr_method": "umap",
  "n_neighbors": 15,
  "min_dist": 0.1,
  "n_epochs": 200
}
```

### 3. バックエンド処理
```
a) ファイル読込
   vector.npy[point_ids] → shape (N, D)
   projection.npy[point_ids] → shape (N, 2)

b) GPU 転送
   vectors → GPU: (N, D)
   init_coords → GPU: (N, 2)

c) UMAP 実行
   embedding = UMAP.fit_transform(
     vectors,
     init=init_coords  ← メンタルマップ保持
   ) → shape (N, 2)

d) Base64 エンコード
   embedding → numpy.save → base64.b64encode
```

### 4. レスポンス
```json
{
  "status": "success",
  "coordinates": "gAN9cQA...",
  "shape": [N, 2],
  "point_ids": [0, 1, 2, ...]
}
```

### 5. フロントエンド処理
```
a) Base64 デコード
   atob(coordinates) → Uint8Array → Float32Array

b) 座標更新
   setDRPoints() で新しい座標を適用

c) 再描画
   D3 visualization を更新
```

---

## メンタルマップ保持メカニズム

### 問題
ズーム時に座標を完全に新規計算すると、ユーザーの「mental map」が失われる

### 解決策
現在の2D座標を **UMAP の初期位置** として使用

```python
# UMAP にアルゴリズムの開始位置を指定
umap_model = UMAP(
    n_components=2,
    init=current_2d_coords,  # ← ここが重要
    n_epochs=200
)
```

### 効果
1. アルゴリズムが現在の座標から開始
2. 最適化中も近い点は近いままで進化
3. ユーザーが見た「配置」が大きく変わらない
4. 新しい選択ポイント間の距離がより正確になる

---

## Base64 エンコード/デコード

### なぜ Base64？
- バイナリ（numpy 配列）を JSON で送信可能にする
- 標準的で互換性がある
- 効率的（オーバーヘッド約 33%）

### 実装

**バックエンド (Python)**
```python
def _numpy_to_b64(array: np.ndarray) -> str:
    buff = io.BytesIO()
    np.save(buff, array, allow_pickle=False)
    return base64.b64encode(buff.getvalue()).decode('utf-8')

def _b64_to_numpy(data_b64: str) -> np.ndarray:
    decoded = base64.b64decode(data_b64)
    return np.load(io.BytesIO(decoded))
```

**フロントエンド (TypeScript)**
```typescript
function base64ToFloat32Array(b64: string): Float32Array {
  const binaryString = atob(b64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}
```

---

## GPU UMAP パラメータ

| パラメータ | デフォルト | 効果 |
|-----------|---------|------|
| `n_neighbors` | 15 | 小: 局所的 / 大: グローバル |
| `min_dist` | 0.1 | 小: ポイント密集 / 大: 分散 |
| `n_epochs` | 200 | 小: 高速 / 大: 高品質 |

### 推奨値

**少ないポイント（<100）**
```
n_neighbors: 20
n_epochs: 300
```

**多いポイント（>1000）**
```
n_neighbors: 10
n_epochs: 100
```

---

## ファイル構成

```
src/d3-app/
├── src/
│   ├── backend/
│   │   ├── services/
│   │   │   └── d3_data_manager.py       (zoom_redraw メソッド)
│   │   ├── main_d3.py                   (POST /api/zoom/redraw)
│   │   └── test_zoom_api.py             (テストスクリプト)
│   └── components/
│       ├── DRVisualization.tsx          (UI ボタン - 未実装)
│       └── Fetch.ts                     (API 関数 - 未実装)
├── data/
│   ├── vector.npy                       (高次元ベクトル)
│   └── projection.npy                   (現在の2D座標)
└── docs/
    └── zoom/
        ├── README.md
        ├── OVERVIEW.md
        ├── QUICK_START.md
        ├── ARCHITECTURE.md (このファイル)
        ├── FRONTEND_GUIDE.md
        ├── IMPLEMENTATION.md
        ├── CONFIGURATION.md
        └── TROUBLESHOOTING.md
```

---

## パフォーマンス分析

### 処理時間の内訳（1000ポイント）

| ステップ | 時間 |
|---------|------|
| ファイル読込 | 0.5s |
| GPU 転送 | 1s |
| UMAP 計算 | 20-30s |
| GPU → CPU | 1s |
| Base64 エンコード | 0.5s |
| **合計** | **23-33s** |

### メモリ使用

```
GPU メモリ = 4 × N_points × D_dims (bytes)
例: 1000 ポイント × 300 次元 = 1.2 GB
```

---

次のドキュメント: **FRONTEND_GUIDE.md** (実装手順)
