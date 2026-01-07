# フロントエンド実装ガイド

ズーム機能をフロントエンドで使えるようにするためのステップバイステップ実装ガイド

## 準備状況

### ✅ 既に実装済み
- State管理: `useAppStore.tsx`
  - `zoomTargetPoints`, `zoomTargetClusters`, `isZoomActive`
  - `setZoomTarget()`, `clearZoomTarget()`, `setZoomActive()`
  - `getZoomTargetPoints()` ヘルパー

### ⏳ 実装が必要
1. API クライアント関数
2. UI ボタン
3. 統合テスト

---

## ステップ 1: API クライアント関数を作成（15-20分）

**ファイル**: `src/components/Fetch.ts`

```typescript
import * as base64 from 'base64-js';

/**
 * ズーム機能用に選択ポイントの2D座標を再計算
 * @param pointIds - ズーム対象のポイントID配列
 * @returns Base64デコード済みの浮動小数点座標配列
 */
export async function fetchZoomRedraw(pointIds: number[]): Promise<Float32Array> {
  const response = await fetch('/api/zoom/redraw', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      point_ids: pointIds,
      dr_method: 'umap',
      n_neighbors: 15,
      min_dist: 0.1,
      n_epochs: 200
    })
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  const data = await response.json();
  
  if (data.status !== 'success') {
    throw new Error(data.message || 'Unknown error');
  }

  // Base64をデコード
  const binaryString = atob(data.coordinates);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  
  return new Float32Array(bytes.buffer);
}
```

---

## ステップ 2: UI ボタンを追加（10-15分）

**ファイル**: `src/components/DRVisualization.tsx`

```typescript
import { fetchZoomRedraw } from './Fetch';
import { useSelection } from '../app/useAppStore';

// コンポーネント内：

const { getZoomTargetPoints, setZoomActive } = useSelection();
const [isZoomLoading, setIsZoomLoading] = useState(false);

const handleZoomIn = async () => {
  const pointIds = getZoomTargetPoints(filteredData);
  
  if (pointIds.length === 0) {
    alert('ポイントを選択してください');
    return;
  }

  setIsZoomLoading(true);
  setZoomActive(true);

  try {
    const newCoords = await fetchZoomRedraw(pointIds);

    // DRポイントを新しい座標で更新
    setDRPoints(prevPoints => {
      const updated = [...prevPoints];
      for (let i = 0; i < pointIds.length; i++) {
        const pointId = pointIds[i];
        updated[pointId] = {
          ...updated[pointId],
          x: newCoords[i * 2],
          y: newCoords[i * 2 + 1]
        };
      }
      return updated;
    });

    // 再描画をトリガー
    if (svgRef.current) {
      updateVisualization();
    }
  } catch (error) {
    console.error('Zoom failed:', error);
    alert(`ズーム失敗: ${error}`);
  } finally {
    setIsZoomLoading(false);
    setZoomActive(false);
  }
};

// JSXのボタン部分：
<button
  onClick={handleZoomIn}
  disabled={isZoomLoading}
  style={{ marginLeft: '10px', padding: '8px 16px' }}
>
  {isZoomLoading ? (
    <>
      <span>🔄 ズーム中...</span>
    </>
  ) : (
    <>
      <span>🔍 ズーム ({getZoomTargetPoints(filteredData).length})</span>
    </>
  )}
</button>
```

---

## ステップ 3: 統合テスト（30分）

### テスト項目

1. **小規模ズーム（10-50ポイント）**
   - [ ] ボタンクリックで計算開始
   - [ ] ローディング表示される
   - [ ] 新しい座標が表示される
   - [ ] 処理完了

2. **中規模ズーム（100-500ポイント）**
   - [ ] 15-25秒程度で完成
   - [ ] メモリ使用量確認（1GB以下）
   - [ ] 座標が正しく更新される

3. **大規模ズーム（1000+ポイント）**
   - [ ] 25-40秒程度で完成
   - [ ] ユーザー体験は許容範囲か
   - [ ] エラーハンドリング動作確認

4. **エラーハンドリング**
   - [ ] ポイント選択なしで実行 → アラート表示
   - [ ] GPU 未対応環境 → エラーメッセージ
   - [ ] API 失敗時 → エラーハンドリング

---

## ステップ 4: パフォーマンス最適化（任意）

### 遅い場合
```typescript
// n_epochs を減らす
const newCoords = await fetchZoomRedraw(pointIds, {
  n_epochs: 100  // デフォルト 200 → 100
});
```

### メモリ不足の場合
```typescript
// 大規模選択時は n_neighbors を減らす
if (pointIds.length > 500) {
  n_neighbors = 10;  // デフォルト 15 → 10
}
```

---

## コード例：完全版

```typescript
// DRVisualization.tsx 内

import { useState, useRef } from 'react';
import { fetchZoomRedraw } from './Fetch';
import { useSelection } from '../app/useAppStore';

export function DRVisualization() {
  const svgRef = useRef(null);
  const [dRPoints, setDRPoints] = useState([]);
  const [isZoomLoading, setIsZoomLoading] = useState(false);
  
  const { getZoomTargetPoints, setZoomActive } = useSelection();

  const handleZoomIn = async () => {
    const pointIds = getZoomTargetPoints(dRPoints);
    
    if (pointIds.length === 0) {
      alert('ポイントを選択してください');
      return;
    }

    setIsZoomLoading(true);
    setZoomActive(true);

    try {
      console.log(`Zooming into ${pointIds.length} points...`);
      const newCoords = await fetchZoomRedraw(pointIds);

      setDRPoints(prevPoints => {
        const updated = [...prevPoints];
        for (let i = 0; i < pointIds.length; i++) {
          const pointId = pointIds[i];
          updated[pointId] = {
            ...updated[pointId],
            x: newCoords[i * 2],
            y: newCoords[i * 2 + 1]
          };
        }
        return updated;
      });

      alert('✅ ズーム完了！');
    } catch (error) {
      console.error('Zoom error:', error);
      alert(`❌ ズーム失敗: ${error}`);
    } finally {
      setIsZoomLoading(false);
      setZoomActive(false);
    }
  };

  return (
    <div>
      <button
        onClick={handleZoomIn}
        disabled={isZoomLoading}
        style={{
          padding: '8px 16px',
          backgroundColor: isZoomLoading ? '#ccc' : '#4CAF50',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: isZoomLoading ? 'wait' : 'pointer'
        }}
      >
        {isZoomLoading ? '🔄 ズーム中...' : `🔍 ズーム (${getZoomTargetPoints(dRPoints).length})`}
      </button>
      
      <svg ref={svgRef}>
        {/* D3 visualization */}
      </svg>
    </div>
  );
}
```

---

## トラブルシューティング

### 座標が更新されない
→ `setDRPoints()` の後に `updateVisualization()` を呼んでいるか確認

### Base64 デコードエラー
→ `atob()` は UTF-16 でのデコードが必要な場合があります
→ 代替: `base64-js` ライブラリを使用

### タイムアウト
```typescript
// フロントエンドでタイムアイト増やす
const controller = new AbortController();
const timeout = setTimeout(() => controller.abort(), 120000); // 120秒

fetch(..., { signal: controller.signal });
```

---

## 参考ドキュメント

- **ARCHITECTURE.md** - システムの仕組み
- **QUICK_START.md** - バックエンドテスト
- **TROUBLESHOOTING.md** - 問題解決
