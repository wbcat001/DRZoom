# アプリケーション内の選択状態構造

## 現在の選択状態の種類

アプリケーション内では複数の選択状態が管理されています。以下に全体像を示します。

### 1. **DR View（次元削減ビュー）での選択**

#### `selectedPointIds: Set<number>`
- **意味**: ユーザーが次元削減ビューでlasso/brush選択したポイントのID
- **更新元**: DRVisualization.tsxのlasso/brush選択終了時
- **更新アクション**: `SELECT_POINTS`
- **用途**: 
  - ポイント詳細表示
  - 選択ポイント群のクラスタ集計
  - 選択ポイントの強調表示

#### `selectedClusterIds: Set<number>`
- **意味**: ユーザーが直接選択したクラスタID（個別クリック時）
- **更新元**: DRVisualization.tsxやDendrogram.tsxでのクリック
- **更新アクション**: `SELECT_CLUSTERS`
- **用途**:
  - クラスタメタ情報表示
  - Dendrogram内の強調表示
  - 関連情報の表示

### 2. **自動派生選択（DR View）**

#### `drSelectedClusterIds: Set<number>`
- **意味**: selectedPointIdsから自動計算された「選択ポイントが属するクラスタ」
- **更新元**: useAppStore内のuseEffectで自動計算
- **計算ロジック**: 
  ```
  selectedPointIds → API呼び出し(/api/clusters_from_points) 
  → containment_ratioが閾値以上のクラスタ → drSelectedClusterIds
  ```
- **用途**:
  - DRビュー内でのクラスタ単位の強調表示
  - Dendrogramの連動強調表示
  - 次元削減時のズーム対象範囲特定

#### `nearbyClusterIds: Set<number>`
- **意味**: ユーザーがDRビューでポイントをホバーしたときの「近傍クラスタ」
- **更新元**: DRVisualization.tsxのマウスホバー時
- **API**: `/api/clusters/{clusterId}/nearby`
- **用途**:
  - ホバーポイントのクラスタとその近傍を赤線で強調
  - 一時的な視覚的フィードバック

### 3. **Dendrogram View での選択**

#### `dendrogramHoveredCluster: number | null`
- **意味**: Dendrogramのマージ操作（ノード）にホバーしたインデックス
- **更新元**: Dendrogram.tsxのマウスホバー
- **更新アクション**: `SET_DENDROGRAM_HOVERED`
- **用途**:
  - Dendrogramのセグメント強調
  - DRビューでの対応クラスタの強調
  - 一時的な視覚的フィードバック

### 4. **Heatmap View での選択**

#### `heatmapClickedClusters: Set<number>`
- **意味**: Heatmapでクリックされたクラスタのセット
- **更新元**: ClusterHeatmap.tsxでのセル/行クリック
- **更新アクション**: `SET_HEATMAP_CLICKED`
- **用途**:
  - Heatmap内での強調表示
  - 他ビューでの対応クラスタの強調
  - 比較分析の対象指定

### 5. **検索機能での選択**

#### `searchQuery: string`
- **意味**: ユーザーが入力した検索クエリ
- **更新元**: SearchBar.tsxでのテキスト入力
- **用途**: クエリ自体の保存

#### `searchResultPointIds: Set<number>`
- **意味**: 検索クエリにマッチしたポイントのIDセット
- **更新元**: SearchBar.tsxでの検索実行時（API呼び出し後）
- **更新アクション**: `SET_SEARCH_RESULTS`
- **用途**:
  - 検索結果のDRビュー内での強調表示
  - テキストアノテーション表示

### 6. **メタ情報**

#### `lastInteractionSource: 'dr' | 'dendrogram' | 'heatmap' | 'none'`
- **意味**: 最後に相互作用があったビューの種類
- **用途**: ビュー間同期時の優先度判定

---

## 状態遷移図

```
┌─────────────────────────────────────────────────────────┐
│                  DRVisualization View                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ① Lasso/Brush選択 → selectedPointIds                   │
│                      ↓                                   │
│                    自動派生                              │
│                      ↓                                   │
│                drSelectedClusterIds (API計算)            │
│                                                          │
│  ② ポイントホバー → API呼び出し                         │
│                      ↓                                   │
│                nearbyClusterIds                         │
│                                                          │
│  ③ ポイント/クラスタクリック → selectedClusterIds      │
│                                                          │
└─────────────────────────────────────────────────────────┘
           ↕ (相互参照・同期)
┌─────────────────────────────────────────────────────────┐
│                  Dendrogram View                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ① ノードホバー → dendrogramHoveredCluster              │
│       ↓                                                  │
│     DRビューで対応クラスタを強調                        │
│                                                          │
│  ② クラスタクリック → selectedClusterIds               │
│                                                          │
└─────────────────────────────────────────────────────────┘
           ↕ (相互参照・同期)
┌─────────────────────────────────────────────────────────┐
│                  ClusterHeatmap View                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ① セルクリック → heatmapClickedClusters               │
│       ↓                                                  │
│     DRビューで対応クラスタを強調                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 「ズーム時に使用する選択」の決定

**ズーム実行時の対象範囲決定フロー**:

```
ユーザーが「Zoom」ボタンをクリック
           ↓
┌─────────────────────────────────────────────────┐
│ 優先度順で対象ポイント群を決定:                 │
│ 1. selectedPointIds が存在？                     │
│    → YES: 選択ポイント群をズーム対象に           │
│ 2. selectedClusterIds が存在？                  │
│    → YES: クラスタ内の全ポイントをズーム対象に   │
│ 3. drSelectedClusterIds が存在？                │
│    → YES: クラスタ群内の全ポイントをズーム対象に │
│ 4. なし                                         │
│    → 警告: 何も選択されていません               │
└─────────────────────────────────────────────────┘
           ↓
ズーム対象ポイント群が確定
           ↓
初期座標を保持したまま次元削減を再実行
           ↓
DRビューを更新（ズームインした表示）
```

---

## 曖昧性の原因と改善提案

### 現在の曖昧性

1. **複数の選択状態が同時に存在する**
   - selectedPointIds（ポイント選択）
   - selectedClusterIds（クラスタ選択）
   - drSelectedClusterIds（自動派生）
   - 同時に複数が活性化する場合がある → 表示優先度が不明確

2. **自動派生選択と手動選択の混在**
   - drSelectedClustersはAPI自動計算
   - selectedClustersはユーザー手動選択
   - 同期がとれないと矛盾が生じる

3. **「近傍クラスタ」の位置づけが不明確**
   - 一時的な視覚フィードバックなのか
   - 永続的な選択なのか不明確

### 改善案

#### オプション1: 「ズーム対象」として明示的な状態を追加

```typescript
interface SelectionState {
  // ... 既存の選択状態 ...
  
  // ズーム対象として明示的に指定
  zoomTargetPoints: Set<number>;
  zoomTargetClusters: Set<number>;
  isZoomActive: boolean;
}
```

**メリット**: ズーム用の状態が明確になる

#### オプション2: 選択状態を「スコープ」で分類

```typescript
type SelectionScope = 'points' | 'clusters' | 'derived' | 'search' | 'temporary';

interface Selection {
  scope: SelectionScope;
  ids: Set<number>;
  source: 'dr' | 'dendrogram' | 'heatmap' | 'auto';
  isPersistent: boolean;  // 永続的な選択か一時的か
  priority: number;       // 表示優先度
}

interface SelectionState {
  activeSelections: Selection[];
  getPrimarySelection(): Selection | null;
}
```

**メリット**: 優先度と属性が明確になる

---

## 推奨: ズーム機能の実装方針

**ステップ1: 選択状態を整理**
- ズーム対象ポイントを確定するための「優先度」を定義
- 各状態の「用途」を明確化（表示用 vs ズーム用）

**ステップ2: 新しいアクション追加**
```typescript
| { type: 'SET_ZOOM_TARGET'; payload: { pointIds: number[], source: 'manual' | 'derived' } }
| { type: 'EXECUTE_ZOOM'; payload: { method: 'umap' | 'tsne' | 'pca' } }
```

**ステップ3: UI追加**
- 「Zoom In」ボタンを配置
- 対象ポイント数を表示: "123 points selected - Click to zoom"
- 処理中の進捗表示

