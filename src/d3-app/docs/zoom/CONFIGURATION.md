# ズーム機能 - 設定ガイド

## パラメータ調整

### UMAP パラメータの効果

#### n_neighbors
**用途**: ローカル vs グローバル構造のバランス

| 値 | 特性 | 用途 |
|----|------|------|
| 5-10 | 局所的（ノイズに敏感） | 小規模ズーム、密集クラスタ |
| 15 | バランス型（推奨） | 一般的な用途 |
| 20-30 | グローバル（広い視点） | 大規模ズーム、全体構造 |
| 50+ | 非常にグローバル | 超大規模データ |

**使い分け:**
```json
// 少数ポイント（<100）
{"n_neighbors": 20}

// 標準（100-1000）
{"n_neighbors": 15}

// 多数ポイント（>1000）
{"n_neighbors": 10}
```

#### min_dist
**用途**: ポイント間の最小距離制御

| 値 | 特性 | 用途 |
|----|------|------|
| 0.01-0.05 | 非常に密集 | クラスタ内の詳細確認 |
| 0.1 | 標準（推奨） | 一般的な用途 |
| 0.2-0.3 | 分散 | クラスタ間の分離強調 |
| 0.5+ | 非常に分散 | アウトライア確認 |

**使い分け:**
```json
// クラスタ内を詳しく見たい
{"min_dist": 0.05}

// バランス型
{"min_dist": 0.1}

// クラスタ間を強調
{"min_dist": 0.2}
```

#### n_epochs
**用途**: 最適化の深さ（実行時間と品質のバランス）

| 値 | 時間 | 品質 | 用途 |
|----|------|------|------|
| 50 | 2-5s | 低 | プレビュー、テスト |
| 100 | 5-10s | 中 | 中規模データ |
| 200 | 10-20s | 高 | 推奨（デフォルト） |
| 300+ | 20-40s | 非常に高 | 本番、小規模 |

**使い分け:**
```json
// 高速プレビュー
{"n_epochs": 100}

// バランス型（推奨）
{"n_epochs": 200}

// 高品質（時間に余裕がある場合）
{"n_epochs": 300}
```

---

## 推奨設定パターン

### パターン 1: 小規模ズーム（<100ポイント）
```json
{
  "dr_method": "umap",
  "n_neighbors": 20,
  "min_dist": 0.1,
  "n_epochs": 300
}
```
- 実行時間: 5-10s
- 品質: 最高
- 用途: 詳細確認、少数エリア

### パターン 2: 標準ズーム（100-500ポイント）
```json
{
  "dr_method": "umap",
  "n_neighbors": 15,
  "min_dist": 0.1,
  "n_epochs": 200
}
```
- 実行時間: 10-20s
- 品質: 高
- 用途: 一般的な用途（推奨）

### パターン 3: 大規模ズーム（500-1000ポイント）
```json
{
  "dr_method": "umap",
  "n_neighbors": 12,
  "min_dist": 0.1,
  "n_epochs": 150
}
```
- 実行時間: 15-30s
- 品質: 中〜高
- 用途: 大規模選択

### パターン 4: 超高速プレビュー
```json
{
  "dr_method": "umap",
  "n_neighbors": 10,
  "min_dist": 0.15,
  "n_epochs": 50
}
```
- 実行時間: 2-5s
- 品質: 低〜中
- 用途: 高速フィードバック

### パターン 5: 高品質確認
```json
{
  "dr_method": "umap",
  "n_neighbors": 25,
  "min_dist": 0.08,
  "n_epochs": 400
}
```
- 実行時間: 20-40s
- 品質: 最高
- 用途: 最終確認、プレゼン

---

## シナリオ別チューニング

### シナリオ 1: 密集クラスタの詳細確認

**目標**: クラスタ内のポイント配置を詳しく見る

```json
{
  "n_neighbors": 8,      // 局所的に
  "min_dist": 0.05,      // 密集に
  "n_epochs": 300        // 高品質
}
```

**理由**:
- n_neighbors を小さく → 局所構造重視
- min_dist を小さく → ポイント密集
- n_epochs 大 → 最適配置

### シナリオ 2: クラスタ間の分離確認

**目標**: 複数クラスタ間の距離関係を見る

```json
{
  "n_neighbors": 20,     // グローバルに
  "min_dist": 0.2,       // 分散に
  "n_epochs": 200
}
```

**理由**:
- n_neighbors 大 → グローバル構造重視
- min_dist 大 → クラスタ分離強調

### シナリオ 3: 大規模データの迅速探索

**目標**: 速度と品質のバランス

```json
{
  "n_neighbors": 10,     // 中程度
  "min_dist": 0.12,      // 中程度
  "n_epochs": 100        // 高速
}
```

**理由**:
- 全体的にバランス取った設定
- 実行時間を短縮

### シナリオ 4: アウトライア検出

**目標**: 外れ値とノイズを視覚化

```json
{
  "n_neighbors": 30,     // グローバル
  "min_dist": 0.25,      // かなり分散
  "n_epochs": 250
}
```

**理由**:
- n_neighbors 大 → グローバルノイズ検出
- min_dist 大 → アウトライア分離

---

## パフォーマンスチューニング

### 遅い場合の対応

```json
{
  // 現在の設定が遅い場合
  "n_epochs": 200,
  "n_neighbors": 15,
  
  // ↓ これを実施
  "n_epochs": 100,       // 50% 削減
  "n_neighbors": 10      // 33% 削減
  // 実行時間は 50-60% 短縮可能
}
```

### メモリ不足の場合

```json
{
  // メモリ不足時は
  "n_neighbors": 15,     // ← 最重要：これを 5-10 に削減
  "min_dist": 0.1,
  "n_epochs": 150        // これを削減しても効果は小さい
}
```

**理由**: n_neighbors が GPU メモリ使用に最も影響する

---

## API 呼び出し例

### JavaScript から API 呼び出し

```typescript
async function zoomWithCustomSettings(
  pointIds: number[],
  quality: 'fast' | 'balanced' | 'high'
) {
  const settings = {
    fast: {
      n_neighbors: 10,
      min_dist: 0.15,
      n_epochs: 50
    },
    balanced: {
      n_neighbors: 15,
      min_dist: 0.1,
      n_epochs: 200
    },
    high: {
      n_neighbors: 20,
      min_dist: 0.08,
      n_epochs: 300
    }
  };

  const response = await fetch('/api/zoom/redraw', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      point_ids: pointIds,
      dr_method: 'umap',
      ...settings[quality]
    })
  });

  return response.json();
}

// 使用例
await zoomWithCustomSettings([0, 1, 2, ...], 'balanced');
```

---

## 設定値の保存

### ユーザー設定を LocalStorage に保存

```typescript
// 設定を保存
const zoomSettings = {
  n_neighbors: 15,
  min_dist: 0.1,
  n_epochs: 200
};
localStorage.setItem('zoomSettings', JSON.stringify(zoomSettings));

// 設定を読込
const saved = JSON.parse(localStorage.getItem('zoomSettings'));
```

### UI コンポーネント例

```typescript
function ZoomSettings() {
  const [nNeighbors, setNNeighbors] = useState(15);
  const [minDist, setMinDist] = useState(0.1);
  const [nEpochs, setNEpochs] = useState(200);

  return (
    <div>
      <label>
        n_neighbors:
        <input type="range" min="5" max="50" value={nNeighbors} onChange={e => setNNeighbors(+e.target.value)} />
        <span>{nNeighbors}</span>
      </label>

      <label>
        min_dist:
        <input type="range" min="0.01" max="0.5" step="0.01" value={minDist} onChange={e => setMinDist(+e.target.value)} />
        <span>{minDist.toFixed(2)}</span>
      </label>

      <label>
        n_epochs:
        <input type="range" min="50" max="400" step="50" value={nEpochs} onChange={e => setNEpochs(+e.target.value)} />
        <span>{nEpochs}</span>
      </label>
    </div>
  );
}
```

---

## デフォルト設定

**推奨デフォルト**（すべてのユースケースに対応）:
```json
{
  "dr_method": "umap",
  "n_neighbors": 15,
  "min_dist": 0.1,
  "n_epochs": 200
}
```

---

次のドキュメント: **TROUBLESHOOTING.md** (問題解決)
