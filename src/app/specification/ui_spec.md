# 6. ユーザーインターフェース（UI）実装仕様

このドキュメントは、フロントエンド開発（DashまたはD3.js）におけるUIコンポーネントの配置、ID、ウィジェットタイプを定義する。

## 1. レイアウト仕様（12カラムグリッド）

画面は横方向に12カラムで分割され、以下のエリアに固定配置される。

| エリア名 | 役割 | グリッド Col | 配置 |
| :--- | :--- | :---: | :--- |
| A: Control Panel | 全体の設定入力 | 2 | 左端。画面高さ全体を使用。 |
| B: DR View Area | メイン可視化 | 4 | 中央左。Aの右隣。 |
| C: Dendrogram Area | クラスタ構造表示 | 4 | 中央右。Bの右隣。 |
| D: Detail & Info Panel | 詳細情報/ログ | 2 | 右端。画面高さ全体を使用。 |

---

## 2. A: Control Panel (Col: 2) のコンポーネント

（省略されていた部分を実装フェーズで必要な要素として仮定義します）

### 2.1 データセット選択

* コンポーネントID: `dataset-selector`
* ウィジェット型: ドロップダウンリスト
* 初期値: 'iris'
* オプション: 'iris', 'digits', 'wine'

### 2.2 次元削減手法選択

* コンポーネントID: `dr-method-selector`
* ウィジェット型: ラジオボタン（縦積み）
* 初期値: 'UMAP'
* オプション: 'UMAP', 'TSNE', 'PCA'

### 2.3 パラメータ設定エリア

手法ごとに表示が切り替わるタブコンポーネントを使用する。

| 手法 | パラメータ | コンポーネントID | ウィジェット型 | 範囲/初期値 |
| :--- | :--- | :--- | :--- | :--- |
| UMAP | n_neighbors | `umap-n-neighbors` | スライダー | 5 ～ 50 (初期値: 15) |
| UMAP | min_dist | `umap-min-dist` | 数値入力 | 0.0 ～ 0.99 (初期値: 0.1) |
| TSNE | perplexity | `tsne-perplexity` | スライダー | 5 ～ 50 (初期値: 30) |
| PCA | n_components | `pca-n-components` | ドロップダウン | 2, 3 (初期値: 2) |

### 2.4 実行トリガー

* コンポーネントID: `execute-button`
* ウィジェット型: ボタン
* ラベル: 'Run Analysis'
* 動作: パラメータ変更後、このボタンが押された時のみ、バックエンドに計算を要求する。

---

## 3. B: DR View Area (Col: 4) のコンポーネント

次元削減結果を表示するメインの可視化コンポーネントの仕様。

* コンポーネントID: `dr-visualization-plot`
* 可視化型: 散布図（インタラクティブプロット）
* データソース: `apply_dimension_reduction` ツール関数からの戻り値。

### 3.1. エンコーディング

* Position: 必須（削減後の x, y 座標）
* Color: 必須（クラスラベルにマッピング）
* Size: 固定 2px。

### 3.2. インタラクションモード切替

* ID: `dr-interaction-mode-toggle`
* ウィジェット型: ラジオボタンまたはトグルスイッチ
* オプション: 'Brush Selection'（範囲選択モード）, 'Zoom/Pan'（ズーム/移動モード）
* 初期値: 'Zoom/Pan'

### 3.3. インタラクションの技術的トリガー

* 範囲選択: `dr-interaction-mode-toggle` が 'Brush Selection' のときのみ有効。選択領域の変更時、イベントID: `dr-selection-updated` をトリガーする。
* 単一点クリック: イベントID: `dr-point-clicked` をトリガーする。

---

## 4. C: Dendrogram Area (Col: 4) のコンポーネント

階層的クラスタリング結果を表示するビューの仕様。

* コンポーネントID: `dendrogram-plot`
* 可視化型: 樹形図
* データソース: バックエンドで生成された Linkage Matrix。

### 4.1. インタラクションモード切替

* ID: `dendro-interaction-mode-toggle`
* ウィジェット型: ラジオボタンまたはトグルスイッチ
* オプション: 'Node Selection'（ノード選択モード）, 'Zoom/Pan'（ズーム/移動モード）
* 初期値: 'Node Selection'

### 4.2. 表示オプション切替（トグル）

* ID: `dendro-width-option-toggle`
* ウィジェット型: チェックボックスまたはトグルスイッチ
* 機能: デンドログラムの枝の幅を、その枝に含まれるクラスタのサイズ（データポイントの数）に比例させるかどうかを切り替える。
* オプション: true（幅を反映）, false（固定幅）
* 初期値: false（固定幅）

---

## 5. D: Detail & Info Panel (Col: 2) のコンポーネント

情報表示用のパネル。タブ形式で複数の情報を切り替える。

* コンポーネントID: `detail-info-tabs`
* ウィジェット型: タブコンポーネント

### 5.1 タブ 1: ポイント詳細

* タブ ID: `tab-point-details`
* 表示内容:
    * DR Viewでクリックされたデータポイントの元の全特徴量のリスト。
    * そのデータポイントの所属するクラスタID。
    * 近傍点（k=5をデフォルトとする）のリスト。

### 5.2 タブ 2: 選択範囲の統計

* タブ ID: `tab-selection-stats`
* 表示内容:
    * DR Viewで範囲選択されたデータポイントの元の特徴量ごとの平均値と標準偏差。

### 5.3 タブ 3: システムログ

* タブ ID: `tab-system-log`
* 表示内容:
    * 計算開始/完了時刻、使用したパラメータ、実行時間などのシステムメッセージ。

---

## 6. ビュー間の連携（Linking & Brushing）

| トリガー（発生元） | イベント ID | アクション（影響先） | 具体的な変化（Emphasis/Highlight） |
| :--- | :--- | :--- | :--- |
| DR View で範囲選択 | `dr-selection-updated` | Dendrogram View | 選択されたデータポイントを含む最小限の共通クラスタノードをハイライトする。 |
| DR View で範囲選択 | `dr-selection-updated` | Detail View | 選択範囲内のデータポイントの元の特徴