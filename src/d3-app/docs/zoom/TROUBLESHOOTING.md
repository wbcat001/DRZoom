# ズーム機能 - トラブルシューティング

## よくある問題と解決策

### 問題 1: "GPU UMAP not available"

**症状**
```json
{
  "status": "error",
  "message": "GPU UMAP not available. Install cupy and cuml."
}
```

**原因**
- CuPy または cuML がインストールされていない
- NVIDIA ドライバが古い
- CUDA バージョンが合っていない

**解決策**

```bash
# 1. 既存のインストールをクリア
pip uninstall cupy cuml -y

# 2. CUDA バージョン確認
nvidia-smi  # CUDA バージョンを確認

# 3. 新規インストール（CUDA 11.2 の場合）
conda create -n drzoom -c rapids -c conda-forge \
  cuml cupy cudatoolkit=11.2 python=3.9

# 4. アクティベート
conda activate drzoom

# 5. GPU テスト
python -c "import cupy; import cuml; print('✓ GPU Ready')"
```

---

### 問題 2: "Vector file not found at..."

**症状**
```json
{
  "status": "error",
  "message": "Vector file not found at .../vector.npy"
}
```

**原因**
- `vector.npy` ファイルが存在しない
- ファイルパスが間違っている
- データディレクトリが見つからない

**解決策**

```bash
# 1. ファイルの存在確認
ls -la src/d3-app/data/vector.npy

# 2. ファイルのサイズ確認（100+MB あるはず）
ls -lh src/d3-app/data/vector.npy

# 3. NumPy で内容確認
python -c "
import numpy as np
data = np.load('src/d3-app/data/vector.npy')
print(f'Shape: {data.shape}')  # (N, D) で表示されるはず
print(f'Dtype: {data.dtype}')
"

# 4. ファイルが破損している場合は再生成
# （元のデータから再処理が必要）
```

---

### 問題 3: "Point IDs out of range"

**症状**
```json
{
  "status": "error",
  "message": "Point IDs out of range [0, 99999]"
}
```

**原因**
- ポイントID が無効な範囲
- データ数より大きいID を指定

**解決策**

```bash
# 1. データのサイズ確認
python -c "
import numpy as np
data = np.load('src/d3-app/data/projection.npy')
print(f'Total points: {data.shape[0]}')
print(f'Valid range: [0, {data.shape[0]-1}]')
"

# 2. リクエストのポイントID を確認
# point_ids: [0, 1, 2, ..., N-1]
# ただし N は総ポイント数
```

---

### 問題 4: "API connection refused"

**症状**
```
Error: connect ECONNREFUSED 127.0.0.1:8000
```

**原因**
- バックエンドが起動していない
- ポート 8000 が別のプロセスで使用中

**解決策**

```bash
# 1. バックエンドの起動確認
ps aux | grep uvicorn

# 2. バックエンド起動
cd src/d3-app/src/backend
uvicorn main_d3:app --host 0.0.0.0 --port 8000

# 3. ポート確認
lsof -i :8000  # ポート 8000 を使用しているプロセス表示

# 4. 別のポートで起動
uvicorn main_d3:app --port 8001
```

---

### 問題 5: GPU メモリ不足

**症状**
```
CUDA out of memory
```

**原因**
- 選択ポイント数が多すぎる
- 他の GPU プロセスが実行中

**解決策**

```bash
# 1. GPU メモリ状態確認
nvidia-smi

# 2. メモリ計算
# メモリ = 4 × ポイント数 × 次元数 (bytes)
# 例: 1000 × 300 × 4 = 1.2 GB

# 3. ポイント数を減らしてテスト
# ズーム対象を 100-500 ポイントに限定

# 4. n_neighbors を減らす
# デフォルト: 15 → 10 に変更

# 5. GPU メモリをクリア
# → 不要な GPU プロセスを終了
killall python
```

---

### 問題 6: 座標の更新が反映されない（フロントエンド）

**症状**
- ズーム API レスポンス成功但し表示変更なし
- コンソールに「coordinates not updated」

**原因**
- `setDRPoints()` 後に再描画がトリガーされていない
- Base64 デコードが失敗
- React の状態更新が完了していない

**解決策**

```typescript
// 1. Base64 デコード確認
const binaryString = atob(coordinates);
console.log(`Decoded length: ${binaryString.length}`);

// 2. Float32Array 変換確認
const bytes = new Uint8Array(binaryString.length);
const array = new Float32Array(bytes.buffer);
console.log(`Decoded array shape: (${array.length / 2}, 2)`);

// 3. 再描画をトリガー
setDRPoints(prev => [...prev]);  // 強制更新

// 4. D3 再描画
if (svgRef.current) {
  d3.select(svgRef.current).selectAll('circle')
    .data(dRPoints)
    .attr('cx', d => xScale(d.x))
    .attr('cy', d => yScale(d.y));
}
```

---

### 問題 7: タイムアウト

**症状**
```
Error: Request timeout after 30 seconds
```

**原因**
- ポイント数が多く処理に時間がかかる
- ネットワーク遅延
- GPU が遅い

**解決策**

```bash
# バックエンド側：
# 1. n_epochs を減らす
# デフォルト: 200 → 100

# 2. フロントエンド側：タイムアウト増加
// 120 秒に設定
const controller = new AbortController();
const timeout = setTimeout(() => controller.abort(), 120000);

fetch('/api/zoom/redraw', {
  signal: controller.signal,
  ...
});
```

---

## パフォーマンス診断

### 処理時間が遅い場合

```bash
# 1. GPU 負荷確認
nvidia-smi --loop=1

# 2. 別の GPU プロセス確認
ps aux | grep python

# 3. VRAM メモリ状態確認
nvidia-smi | grep python

# 4. CPU 側のボトルネック確認
time python -c "
import numpy as np
data = np.load('vector.npy')  # ファイル読込時間測定
"
```

### メモリ問題がある場合

```bash
# 1. メモリプロファイリング
memory_profiler を使用してメモリ使用量測定

# 2. より小さいポイント数でテスト
# 10, 50, 100 ポイントで段階的にテスト

# 3. パラメータを調整
{
  "n_neighbors": 10,  # デフォルト 15 から削減
  "n_epochs": 100,    # デフォルト 200 から削減
  "min_dist": 0.1
}
```

---

## ログ確認

### バックエンド側ログ

```bash
# コンソール出力をファイルに記録
uvicorn main_d3:app > backend.log 2>&1 &

# ログ確認
tail -f backend.log

# エラー検索
grep -i error backend.log
grep -i warning backend.log
```

### フロントエンド側ログ

```typescript
// コンソールで詳細ログを有効化
console.log('Zoom request point IDs:', pointIds);
console.log('API response:', data);
console.log('Decoded coordinates:', newCoords);
```

---

## デバッグコマンド

### GPU 状態確認
```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
```

### CuML バージョン確認
```bash
python -c "import cuml; print(cuml.__version__)"
```

### NumPy 配列検証
```bash
python -c "
import numpy as np
import base64
import io

# Base64 ラウンドトリップテスト
arr = np.random.randn(100, 2).astype(np.float32)
buff = io.BytesIO()
np.save(buff, arr, allow_pickle=False)
b64 = base64.b64encode(buff.getvalue()).decode('utf-8')

# デコード
decoded = base64.b64decode(b64)
restored = np.load(io.BytesIO(decoded))

print(f'Original: {arr.shape}, Restored: {restored.shape}')
print(f'Match: {np.allclose(arr, restored)}')
"
```

---

## よくある質問

**Q: なぜ毎回計算し直す必要があるのか？**  
A: ズーム対象のポイント間により適切な2D配置を見つけるため。選択ポイントだけでUMAPを実行することで、より良い距離解像度が得られます。

**Q: メンタルマップはどのくらい保持される？**  
A: 大体 70-90%。完全には保持されませんが、空間的な相対位置は大きく変わりません。

**Q: GPU がない場合は？**  
A: エラーが返されます。CPU フォールバックを追加することは可能です（別途実装必要）。

**Q: 複数回ズーム（nested zoom）は可能？**  
A: はい。再度新しい座標からズームできます。

---

次のドキュメント: **CONFIGURATION.md** (パラメータ調整)
