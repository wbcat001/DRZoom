# デンドログラムのソート機能 設計書 & 影響分析

## 現在のデンドログラムのデータ構造

### デンドログラム要素にバインドされているデータ

**Linkage Matrix** (`data.linkageMatrix`):
```typescript
interface LinkageMatrixEntry {
  child1: number;      // Sequential index of left child node
  child2: number;      // Sequential index of right child node
  distance: number;    // Merging distance (height in dendrogram)
  size: number;        // Size of merged cluster
  stability?: number;  // Stability metric
  strahler?: number;   // Strahler number
}
type LinkageMatrix = LinkageMatrixEntry[];
```

**デンドログラム座標** (linkage matrix から計算):
- `icoord`: 水平位置（葉のインデックス / x軸）
- `dcoord`: 垂直位置（距離 / 高さ / y軸）
- `leafOrder`: 葉の順序（どの元クラスタがどのx位置に現れるか）

**セグメント描画** (Dendrogram.tsx の 130-155 行):
```
各セグメント（線）はインデックス i を持つ:
  mergeIdx = floor(i / 3)
  linkageEntry = data.linkageMatrix[mergeIdx]
  
セグメントにバインドされた属性:
  - child1, child2 (clusterIdMap でルックアップ)
  - distance (色付け/スタイリング用)
  - size (スタイリング用)
  - stability, strahler (フィルタリング用)
```

---

## 読むべきコードの場所

### 1. **デンドログラムデータの計算場所**
- **ファイル**: `src/utils/dendrogramCoords.ts`
- **関数**: `computeDendrogramCoords(Z, nPoints)` (25行目)
- **処理内容**: Linkage matrix Z を受け取り、座標 (icoord, dcoord, leafOrder) を返す
- **重要点**: ツリーをトップダウンで走査し、x位置を順番に（左から右へ）割り当てる

### 2. **座標の描画場所**
- **ファイル**: `src/components/Dendrogram/Dendrogram.tsx`
- **useEffect**: メイン描画ロジック (77-400行目)
- **データバインディング** (130行目):
  ```tsx
  g.selectAll('.dendrogram-segment')
    .data(dendrogramData.segments, (d, i) => i)
  ```
- **ストロークの色付け** (147行目): mergeIdx を使って linkageEntry をルックアップ

### 3. **葉の位置マッピング場所**
- **ファイル**: `src/components/Dendrogram/Dendrogram.tsx`
- **関数**: `reverseClusterIdMap` (75-85行目)
- **用途**: 元のクラスタID → linkage matrix で使う連番インデックスへマッピング

---

## ソート戦略と必要なコード変更

### Option A: クラスタサイズでソート

**目標**: デンドログラムを並び替えて、大きいクラスタが左、小さいクラスタが右に来るようにする

**必要な変更**:

1. **`dendrogramCoords.ts` を修正 - ソートパラメータを追加**:
   ```typescript
   export function computeDendrogramCoords(
     Z: LinkageMatrix,
     nPoints: number,
     sortBy?: 'size' | 'distance' | 'stability' | 'none'  // 新規
   ): DendrogramCoordinates
   ```

2. **葉の割り当てロジックを更新**:
   - 現在: 葉に x=1, 2, 3, ... と順番に割り当て
   - 新規: 走査前に、葉をサイズで並び替え（降順）
   - その順序でx位置を割り当て

3. **Dendrogram.tsx 内** (62行目):
   ```tsx
   const coords = computeDendrogramCoords(
     data.linkageMatrix,
     data.linkageMatrix.length + 1,
     'size'  // 新規パラメータ
   );
   ```

4. **UIコントロールを追加** (オプション):
   ソートモードを切り替えるドロップダウン/ボタンを追加:
   ```tsx
   <select value={sortMode} onChange={(e) => setSortMode(e.target.value)}>
     <option value="none">デフォルト順序</option>
     <option value="size">クラスタサイズでソート</option>
     <option value="stability">安定度でソート</option>
   </select>
   ```

### Option B: Stability/Strahler でソート

Option A と同様だが、`data.linkageMatrix[leafIdx].stability` または `.strahler` で葉をソート

**場所**: `dendrogramCoords.ts` 35-50行目（葉の初期化）

### Option C: クラスタ間類似度でソート

**目標**: 類似したクラスタが隣接するようにデンドログラムを並び替える

#### データ形式の選択肢

**推奨: ハイブリッド形式（配列 + Map）**

バックエンドからフロントへの転送:
```typescript
{
  clusterSimilarities: [
    [cluster_id_1, cluster_id_2, distance],
    [cluster_id_1, cluster_id_3, distance],
    ...
  ]
}
```

フロント内部での変換 & 使用:
```typescript
interface ClusterSimilarityData {
  similarityMap: Map<string, number>;  // key: "id1-id2" (小さいIDを先に)
}

// 変換関数
function buildSimilarityMap(pairs: [number, number, number][]): Map<string, number> {
  const map = new Map<string, number>();
  pairs.forEach(([id1, id2, dist]) => {
    const key = id1 < id2 ? `${id1}-${id2}` : `${id2}-${id1}`;
    map.set(key, dist);
  });
  return map;
}

// 使用例
function getSimilarity(clusterId1: number, clusterId2: number): number {
  const key = clusterId1 < clusterId2 
    ? `${clusterId1}-${clusterId2}` 
    : `${clusterId2}-${clusterId1}`;
  return similarityMap.get(key) ?? Infinity; // 見つからない場合は最大距離
}
```

**この形式の利点:**
1. ✅ JSON で簡単に転送できる（配列はシリアライズしやすい）
2. ✅ スパースデータに対してメモリ効率が良い（必要なペアのみ保存）
3. ✅ TypeScript で扱いやすい（Map のルックアップは O(1)）
4. ✅ 対称性を自動処理（(a,b) と (b,a) を区別しない）
5. ✅ 欠損値の処理が簡単（getで undefined なら Infinity）

**他の選択肢との比較:**

| 形式 | メリット | デメリット | 推奨度 |
|------|---------|-----------|--------|
| **配列+Map** | バランス良好、実装簡単 | - | ⭐⭐⭐ |
| 完全行列 `number[][]` | アクセス単純 | メモリ大（N²）、JSON肥大化 | ⭐ |
| ネストMap `Map<number, Map<number, number>>` | TypeScript的 | JSON変換が面倒 | ⭐⭐ |
| 辞書 `{[key: string]: number}` | JSON互換 | キー生成が冗長 | ⭐⭐ |

#### ソートアルゴリズムの実装

デンドログラムの各マージノードで、左右の子をソート:

```typescript
function sortChildrenBySimilarity(
  leftClusterId: number,
  rightClusterId: number,
  referenceClusterId: number,  // 親や隣接ノードのID
  similarityMap: Map<string, number>
): [number, number] {
  const leftSim = getSimilarity(leftClusterId, referenceClusterId);
  const rightSim = getSimilarity(rightClusterId, referenceClusterId);
  
  // 類似度が高い（距離が小さい）方を referenceCluster に近い側に配置
  return leftSim < rightSim 
    ? [leftClusterId, rightClusterId]
    : [rightClusterId, leftClusterId];
}
```

**実装箇所:**
- `dendrogramCoords.ts` の `computeDendrogramCoords` 関数内
- ノードを走査する際に、子ノードの順序を類似度に基づいて決定

#### データサイズの考慮

**クラスタ数とデータ量の関係:**

| クラスタ数 | 全組み合わせ数 | 配列サイズ (JSON) | メモリ使用量 |
|-----------|--------------|-----------------|-------------|
| 100 | 4,950 | ~150 KB | 小 |
| 500 | 124,750 | ~3.7 MB | 中 |
| 1,000 | 499,500 | ~15 MB | 大 |
| 5,000 | 12,497,500 | ~375 MB | 非常に大 |

**対策:**
- 1,000 クラスタ以下: 全組み合わせを送っても問題なし
- それ以上: Top-K 類似（各クラスタにつき最も近い K 個のみ）に絞る
- または: クライアント側でオンデマンド計算（必要なペアのみAPIで取得）

---

## 影響分析：他に何が壊れるか？

### ✅ **低リスク（動作するはず）**:
- ✅ レンダリング（D3 はソート順に関係なく icoord/dcoord を使用）
- ✅ ホバーハイライト（mergeIdx → linkageEntry ルックアップは変わらず）
- ✅ ブラシ選択（ブラシ座標を引き続き使用）
- ✅ 色付けロジック（linkageEntry に基づく、位置に基づかない）

### ⚠️ **中リスク（要確認）**:
- ⚠️ **leafOrder マッピング**: DRVisualization が leafOrder を使って元のクラスタIDに戻す場合
  - **ファイル**: `src/components/DRVisualization/DRVisualization.tsx` 45-50行目
  - **リスク**: デンドログラムの葉にホバーした時に leafOrder インデックスを使う場合、ソートによって識別されるクラスタが変わる
  - **緩和策**: `computeDendrogramCoords` が `leafOrder` を正しく更新することをテスト

- ⚠️ **デンドログラムホバー → DR点選択**:
  - **ファイル**: `src/components/Dendrogram/Dendrogram.tsx` 270-300行目（葉へのホバー）
  - **ロジック**: マウス位置から葉インデックスを取得、`reverseClusterIdMap` でマッピング
  - **リスク**: 葉の位置が変わると、マッピングが正しくないといけない
  - **緩和策**: ソート後に `leafOrder` が最新であることを確認

### ❌ **高リスク（壊れる可能性が高い）**:
- ❌ **葉ノードのホバー検出** (Dendrogram.tsx 344行目):
  ```tsx
  const leafCount = data.linkageMatrix.length + 1;
  const leafIndex = Math.floor(xValue / (width / leafCount));
  ```
  - **問題点**: 葉が等間隔に並んでいると仮定している。ソートすると、実際の icoord 値に基づいて再計算する必要がある
  - **必要な修正**: 実際の葉の位置を保存し、逆引きルックアップを行う

#### なぜこのロジックがソートで壊れるのか？

**現在のロジックの仮定:**
```
葉ノード数が n 個の場合、デンドログラムの幅を n 等分して配置
葉 0: x = 0 ~ width/n
葉 1: x = width/n ~ 2*width/n
葉 2: x = 2*width/n ~ 3*width/n
...
```

マウスのx座標を `width / leafCount` で割れば、どの葉かが分かる。

**ソート後の問題:**

しかし、`computeDendrogramCoords` で葉をソートすると、**icoord の値自体が変わる**。例えば：
- ソート前: 葉 [0, 1, 2, 3] が icoord [5, 15, 25, 35] に配置
- ソート後: 葉 [2, 0, 3, 1] が icoord [5, 15, 25, 35] に配置（並び順が変わる）

この時、`leafIndex = floor(xValue / (width / 4))` という計算式は、**ソート前の順序（0,1,2,3）**を返してしまい、実際の葉の並び（2,0,3,1）と一致しない。

**具体例:**
```
ソート前: 葉の並び = [A, B, C, D]
マウスx = 20 (幅100の場合) → leafIndex = floor(20/(100/4)) = 0 → 葉A

ソート後: 葉の並び = [C, A, D, B] (サイズ順)
マウスx = 20 → leafIndex = 0 → でも実際にはAではなくC！
```バックエンドから)
        ↓
computeDendrogramCoords(Z, sortBy='size')
        ↓
coords.icoord (ソート後の葉のx位置)
coords.leafOrder (ソート後の葉インデックス)
        ↓
D3 レンダリング (セグメント、葉)
        ↓
葉ホバー時:
  - マウスのx位置を取得
  - leafOrder で一致する葉を見つける
  - clusterIdMap 経由で元のクラスタIDをルックアップ
  - DR ビューにホバー信号を発信
```

---

## テストチェックリスト

- [ ] ソート後にデンドログラムのセグメントが正しく描画される
- [ ] 葉ノードが正しい水平位置にある
- [ ] 葉にホバーすると DR ビューで正しいクラスタがハイライトされる
- [ ] 葉をクリックすると正しいクラスタが選択される
- [ ] ソートされたデンドログラムでブラシ選択が動作する
- [ ] 色付け/安定度フィルタリングが動作する
- [ ] mergeIdx の範囲外エラーがコンソールに出ない
- [ ] ソートモードを切り替えるとビューがスムーズに更新される

### 類似度ソート固有のテスト

- [ ] バックエンドから `clusterSimilarities` データが正しく返る
- [ ] `buildSimilarityMap` が配列を Map に正しく変換する
- [ ] `getSimilarity(a, b)` と `getSimilarity(b, a)` が同じ値を返す
- [ ] 類似度データがない組み合わせは Infinity を返す
- [ ] 類似度順にソートされた結果、似たクラスタが隣接している
- [ ] 類似度行列のサイズが大きくてもパフォーマンスが劣化しない

---

## データフロー概要

```
data.linkageMatrix (from backend)
        ↓
computeDendrogramCoords(Z, sortBy='size')
        ↓
coords.icoord (leaf x-positions after sorting)
coords.leafOrder (sorted leaf indices)
        ↓
D3 rendering (segments, leaves)
        ↓
On hover leaf:
  - Get mouse x-position
  - Find matching leaf in leafOrder
  - Lookup original cluster ID via clusterIdMap
  - Emit hover signal to DR view
```

---

## Testing Checklist

- [ ] Dendrogram segments render correctly after sorting
- [ ] Leaf nodes are at correct horizontal positions
- [ ] Hovering leaf highlights correct cluster in DR view
- [ ] Clicking leaf selects correct cluster
- [ ] Brush selection still works on sorted dendrogram
- [ ] Color/stability filtering still works
- [ ] No console errors about mergeIdx out of range
- [ ] Switching sort modes updates view smoothly
