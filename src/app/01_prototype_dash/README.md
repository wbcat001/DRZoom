# DimRedVis-Agent UI Prototype

次元削減可視化システムのUIプロトタイプ実装です。

## 概要

このアプリケーションは、UI/UX設計検証のための静的表示Dashアプリケーションです。
バックエンドの計算ロジックは含まれておらず、ダミーデータを使用してUIコンポーネントの動作を確認できます。

## 実装仕様

### レイアウト（12カラムグリッド）

- **A: Control Panel (2列)** - 左端、設定入力エリア
- **B: DR View Area (4列)** - 中央左、メイン可視化エリア  
- **C: Dendrogram Area (4列)** - 中央右、クラスタ構造表示エリア
- **D: Detail & Info Panel (2列)** - 右端、詳細情報/ログエリア

### 主要コンポーネント

#### A: Control Panel
- `dataset-selector`: データセット選択ドロップダウン
- `dr-method-selector`: 次元削減手法選択ラジオボタン
- パラメータ設定エリア（手法に応じて動的変更）
- `execute-button`: 分析実行ボタン

#### B: DR View Area  
- `dr-visualization-plot`: 次元削減結果散布図
- `dr-interaction-mode-toggle`: インタラクションモード切替

#### C: Dendrogram Area
- `dendrogram-plot`: 階層クラスタリング樹形図
- `dendro-interaction-mode-toggle`: インタラクションモード切替
- `dendro-width-option-toggle`: 枝幅オプション切替

#### D: Detail & Info Panel
- `detail-info-tabs`: タブコンポーネント
  - Point Details: クリックしたポイントの詳細
  - Selection Stats: 選択範囲の統計情報
  - System Log: システムログ

## 使用方法

```bash
cd /Users/owner/work/DRZoom/src/app/01_prototype_dash
python app.py
```

ブラウザで http://localhost:8050 にアクセスしてUIを確認できます。

## ダミーデータ

- DR View: 100点のランダムな2次元座標データ（ラベル A, B, C）
- Dendrogram: 10点の簡単な階層構造データ

## 実装上の注意

- 再現性確保のため `random.seed(42)` を使用
- 実際の計算ロジックは含まれていません
- `execute-button` クリック時はコンソールにログ出力のみ
- すべてのUI仕様（`specification/ui_spec.md`）に準拠

## 今後の拡張

このプロトタイプをベースに、実際のバックエンド計算ロジックを統合することで、
完全な次元削減可視化システムを構築できます。
