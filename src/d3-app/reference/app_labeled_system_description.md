# HDBSCAN階層クラスタリング可視化システム - app_labeled.py

## システム概要

`app_labeled.py`は、HDBSCANによる階層クラスタリング結果を多次元で可視化・分析するためのインタラクティブなDashアプリケーションです。大規模データセット（数万〜数十万点）のクラスター構造を効率的に探索できるよう設計されています。

## 主要機能

### 1. 多視点可視化システム
- **2D散布図**: UMAP埋め込み空間での点分布
- **階層デンドログラム**: クラスター合併構造の可視化
- **類似度ヒートマップ**: クラスター間関係の行列表示
- **詳細情報パネル**: 選択されたクラスターの統計情報

### 2. インタラクティブ連動機能
- 各ビュー間でのリアルタイム選択連動
- 色分けによる視覚的ハイライト
- マルチモーダルなデータ探索

### 3. 階層フィルタリング機能
- **Strahlter数**: 階層レベルによる構造フィルタリング
- **Stability値**: クラスター品質による表示制御
- 動的な表示更新とパフォーマンス最適化

## データ構造と前処理

### データ読み込み処理

#### 1. 埋め込みデータの読み込み
```python
# UMAP埋め込み結果とポイントラベル
data_file_path = "src/experiments/18_rapids/result/20251203_053328/data.npz"
data = np.load(data_file_path)
embedding = data['embedding']  # 2D座標
labels = data['words']         # 各ポイントのラベル
```

#### 2. HDBSCANクラスタリング結果の読み込み
```python
# HDBSCAN凝縮木構造
hdbscan_condensed_tree_file_path = "src/experiments/18_rapids/result/20251203_053328/condensed_tree_object.pkl"
with open(hdbscan_condensed_tree_file_path, 'rb') as f:
    hdbscan_condensed_tree = pickle.load(f)
```

#### 3. クラスター類似度行列の読み込み
```python
# クラスター間類似度（複数の距離指標）
similarity_file_path = "src/experiments/19_tree/processed_data/cluster_similarities.pkl"
with open(similarity_file_path, 'rb') as f:
    similarity = pickle.load(f)
    similarity_kl = similarity["kl_divergence"]           # KL距離
    similarity_bc = similarity["bhattacharyya_coefficient"] # バタチャリヤ係数
    similarity_m = similarity["mahalanobis_distance"]      # マハラノビス距離
```

#### 4. クラスター代表ラベルの読み込み
```python
# クラスターの代表単語とキーワードリスト
cluster_label_file_path = "src/experiments/19_tree/processed_data/cluster_to_label.csv"
cluster_representative_labels = {}  # cluster_id -> representative_label
cluster_word_lists = {}            # cluster_id -> [word1, word2, ...]
```

### データ変換処理

#### HDBSCAN凝縮木からLinkage Matrix変換
```python
def get_linkage_matrix_from_hdbscan(condensed_tree):
    """
    HDBSCANの凝縮木をSciPy互換のLinkage Matrixに変換
    
    処理内容:
    1. 凝縮木をlambda値とparent IDでソート
    2. 隣接する2行をペアとして結合操作を再構成
    3. scipy.cluster.hierarchy形式のZ行列を生成
    4. ノードID再マッピング（0からN-1の連続ID）
    
    Returns:
        linkage_matrix: (N-1, 5) array [child1, child2, parent, distance, size]
        old_new_id_map: 元ID -> 新ID のマッピング辞書
    """
```

**主要処理ステップ:**
1. **ペア抽出**: condensed treeから同一lambda値の隣接行を結合ペアとして認識
2. **前提チェック**: lambda値とparent IDの一致性検証
3. **距離変換**: lambda値を距離に変換（max_lambda - lambda）
4. **ID再マッピング**: 葉ノード（0~N-1）と内部ノードの連続ID割り当て

#### Strahlter数計算
```python
def calculate_strahler(Z_matrix, n_leaves):
    """
    各結合ノードのStrahler数（階層複雑度指標）を計算
    
    アルゴリズム:
    - 葉ノード: Strahler数 = 1
    - 内部ノード: 
      - 子ノードのStrahler数が等しい場合: max + 1
      - 子ノードのStrahler数が異なる場合: max
      
    用途: 階層レベルによるデンドログラムフィルタリング
    """
```

#### Stability計算
```python
def compute_stability(raw_tree):
    """
    HDBSCANクラスターのStability（安定性）スコアを計算
    
    計算式: Stability(C) = Σ (lambda_death - lambda_birth) * size
    
    処理手順:
    1. 各クラスターのlambda_birth（誕生時刻）を特定
    2. condensed treeから各分離イベントのlambda_death取得
    3. 生存期間 × サイズの累積でStabilityスコア算出
    
    用途: 低品質クラスターの除外フィルタリング
    """
```

## デンドログラム座標計算システム

### 基本座標計算
```python
def compute_dendrogram_coords(Z, n_points):
    """
    標準的なデンドログラム座標計算
    
    処理フロー:
    1. ノード情報構造体の初期化
    2. 葉ノード順序の決定（サイズベース）
    3. X座標の再帰的計算（子ノード平均）
    4. U字型線分座標の生成
    
    Returns:
        icoord: X座標リスト（4点/セグメント）
        dcoord: Y座標リスト（4点/セグメント）
        leaf_order: 葉ノード表示順序
    """
```

### サイズ考慮版座標計算
```python
def compute_dendrogram_coords_with_size(Z, n_points):
    """
    クラスターサイズを反映したデンドログラム座標計算
    
    拡張機能:
    - 葉ノードサイズをraw_treeから取得
    - point_cluster_mapでクラスター内ポイント数をカウント
    - サイズに応じた視覚的重み付け
    
    用途: クラスターサイズの視覚的強調表示
    """
```

## カラーハイライトシステム

### 色設定定義
```python
HIGHLIGHT_COLORS = {
    'default': '#4A90E2',          # デフォルト状態（明るい青）
    'default_dimmed': '#B8D4F0',   # 背景状態（薄い青）
    'dr_selection': '#FFA500',     # DR選択時（オレンジ）
    'heatmap_click': '#FF0000',    # ヒートマップクリック時（赤）
    'heatmap_to_dr': '#FF1493',   # ヒートマップ→DR連動（ディープピンク）
    'dendrogram_to_dr': '#32CD32'  # デンドログラム→DR連動（ライムグリーン）
}
```

### ハイライト処理システム
**目的**: 複数ビュー間での選択状態を色で区別し、データ探索の文脈を視覚的に提供

**適用場面:**
1. **DR散布図選択**: ラッソ選択されたポイント群をオレンジで強調
2. **ヒートマップクリック**: クリックされたセルの関連クラスターを赤で表示
3. **デンドログラムホバー**: ホバーされたノードの関連クラスターを緑で表示
4. **背景dimming**: 非選択要素を薄色化して選択要素を相対的に強調

## パフォーマンス最適化機能

### 表示制限設定
```python
# パフォーマンス設定
ENABLE_HEATMAP_CLUSTER_LIMIT = True  # ヒートマップクラスタ数制限
MAX_HEATMAP_CLUSTERS = 200           # ヒートマップ最大表示クラスタ数
DR_SELECTION_CLUSTER_RATIO_THRESHOLD = 0.1  # DR選択時の含有率閾値
MAX_CLUSTER_WORDS_DISPLAY = 10       # 詳細パネル最大単語数
MAX_CLUSTER_WORDS_HOVER = 5          # ホバー最大単語数
```

### 動的フィルタリング戦略
1. **大規模ヒートマップ対策**: 上位200クラスターのみ表示してレンダリング時間を短縮
2. **DR選択フィルタ**: 含有率10%未満のクラスターを除外してノイズを削減
3. **テキスト表示制限**: 単語リストを制限して表示領域を最適化

## Dashアプリケーション構造

### レイアウト設計
```python
# Bootstrap Grid System: 12列レイアウト
app.layout = dbc.Container([
    dbc.Row([
        # Column 1: Controls (width=2)
        dbc.Col([...], width=2),
        
        # Column 2: DR Visualization (width=4)  
        dbc.Col([...], width=4),
        
        # Column 3: Dendrogram (width=4)
        dbc.Col([...], width=4),
        
        # Column 4: Details (width=2)
        dbc.Col([...], width=2)
    ]),
    
    dbc.Row([
        # Column 5-12: Heatmap (width=12)
        dbc.Col([...], width=12)
    ])
])
```

### データストア設計
```python
# 選択状態管理
dcc.Store(id='selected-cluster-store', data={'cluster_ids': []}),
dcc.Store(id='dr-selection-store', data={'selected_points': []}),
dcc.Store(id='heatmap-click-store', data={'clicked_clusters': []}),
```

## コールバック関数システム

### 1. パラメータ更新コールバック
```python
@app.callback(
    [Output('dendrogram-plot', 'figure'),
     Output('dendrogram-update-status', 'children')],
    [Input('strahler-slider', 'value'),
     Input('stability-slider', 'value'),
     Input('selected-cluster-store', 'data')]
)
def update_dendrogram_with_highlighting(strahler_value, stability_value, cluster_data):
    """
    デンドログラム表示の動的更新
    
    処理内容:
    1. Strahlter値とStability値によるフィルタリング
    2. フィルタ済みLinkage Matrixの生成
    3. デンドログラム座標の再計算
    4. 選択クラスターのハイライト適用
    5. Plotly図の更新
    
    連動要素:
    - スライダー値変更 → デンドログラム再描画
    - クラスター選択 → ハイライト更新
    """
```

### 2. 散布図更新コールバック  
```python
@app.callback(
    Output('dr-plot', 'figure'),
    [Input('selected-cluster-store', 'data'),
     Input('dr-selection-store', 'data'),
     Input('heatmap-click-store', 'data')]
)
def update_dr_plot_with_highlighting(cluster_data, dr_data, heatmap_data):
    """
    DR散布図の色分け表示更新
    
    ハイライト優先順位:
    1. ヒートマップクリック → 赤色（最優先）
    2. DR選択 → オレンジ色
    3. デンドログラム選択 → 緑色
    4. その他 → デフォルト青色 or dimmed薄青色
    
    処理手順:
    1. 全ポイントをデフォルト色で初期化
    2. 各選択ソース毎に該当ポイントの色を更新
    3. customdataを使用してホバー情報を埋め込み
    4. マーカーサイズとOpacityの調整
    """
```

### 3. ヒートマップ更新コールバック
```python
@app.callback(
    Output('heatmap-plot', 'figure'),
    [Input('similarity-metric-dropdown', 'value'),
     Input('selected-cluster-store', 'data')]
)
def update_heatmap_with_selection(similarity_metric, cluster_data):
    """
    クラスター類似度ヒートマップの更新
    
    類似度指標:
    - KL Divergence: 確率分布間の相違度
    - Bhattacharyya Coefficient: 分布重複度
    - Mahalanobis Distance: 多変量統計距離
    
    選択表示機能:
    1. 選択クラスターの行/列を強調表示
    2. 非選択セルの透明度調整
    3. カラーバーとアノテーションの更新
    """
```

### 4. 選択連動コールバック
```python
@app.callback(
    Output('selected-cluster-store', 'data'),
    [Input('dr-plot', 'selectedData'),
     Input('dendrogram-plot', 'hoverData'),
     Input('heatmap-plot', 'clickData')]
)
def update_selected_clusters(dr_selected, dendro_hover, heatmap_click):
    """
    マルチビュー選択状態の統合管理
    
    選択検出ロジック:
    1. DR選択: selectedDataからポイントインデックス抽出
    2. ポイント→クラスターマッピング（point_cluster_map使用）
    3. 含有率閾値によるフィルタリング
    4. 選択状態ストアの更新
    
    デンドログラムホバー:
    - ホバーされたセグメントからクラスターID抽出
    - 階層関係を考慮した関連クラスター特定
    
    ヒートマップクリック:
    - クリックセルの行/列インデックスからクラスターID取得
    - 類似度に基づく関連クラスター抽出
    """
```

### 5. 詳細パネル更新コールバック
```python
@app.callback(
    Output('detail-panel-content', 'children'),
    [Input('detail-tabs', 'active_tab'),
     Input('selected-cluster-store', 'data'),
     Input('dr-selection-store', 'data')]
)
def update_detail_panel(active_tab, cluster_data, dr_data):
    """
    詳細情報パネルの動的コンテンツ生成
    
    タブ構成:
    1. Cluster Details: 選択クラスターの統計情報
    2. Selection Stats: DR選択の統計サマリ
    3. System Log: アプリケーション操作履歴
    
    表示情報:
    - クラスターサイズとポイント数
    - 代表単語とキーワードリスト
    - Stabilityスコアと品質指標
    - 類似クラスターランキング
    """
```

## データ処理アルゴリズム

### 1. クラスター含有率計算
```python
def get_clusters_from_dr_selection(selected_points, point_cluster_map, threshold=0.1):
    """
    DR選択ポイントからクラスターを特定
    
    アルゴリズム:
    1. 選択ポイント → クラスターマッピング
    2. 各クラスターの含有率計算
    3. 閾値以上のクラスターのみ抽出
    
    含有率 = 選択されたクラスター内ポイント数 / クラスター総ポイント数
    """
```

### 2. 階層関係トラバース
```python
def get_cluster_hierarchy_path(cluster_id, linkage_matrix):
    """
    指定クラスターから根までの階層パスを取得
    
    用途:
    - デンドログラムでの祖先ノード特定
    - 階層レベル判定
    - 関連クラスター抽出
    """
```

### 3. 類似度行列処理
```python
def get_similar_clusters(cluster_id, similarity_matrix, top_k=5):
    """
    指定クラスターと最も類似するクラスターを特定
    
    処理:
    1. 類似度行列から該当行を抽出
    2. 自己類似度を除外
    3. 上位K個のクラスターを返却
    """
```

## ユーザーインタラクション設計

### 1. DR散布図インタラクション
- **ラッソ選択**: 複数ポイントの自由領域選択
- **ボックス選択**: 矩形領域での効率的選択
- **ズーム・パン**: 詳細部分の拡大表示
- **ホバー**: ポイント詳細情報の即座表示

### 2. デンドログラムインタラクション  
- **セグメントホバー**: クラスター統計情報の表示
- **ズーム**: 階層構造の詳細表示
- **フィルタスライダー**: Strahlter/Stability値による動的絞り込み

### 3. ヒートマップインタラクション
- **セルクリック**: 特定クラスターペアの詳細分析
- **行列ハイライト**: 選択クラスターの全関係表示
- **類似度指標切替**: 異なる距離尺度での比較

## 技術スタック

### フロントエンド
- **Dash**: Webアプリケーションフレームワーク
- **Plotly**: インタラクティブ可視化ライブラリ
- **Dash Bootstrap Components**: レスポンシブUIコンポーネント

### バックエンド計算
- **NumPy**: 高速数値計算
- **SciPy**: 科学計算（階層クラスタリング）
- **Pandas**: データフレーム操作

### データ保存・読み込み
- **pickle**: Pythonオブジェクトのシリアライゼーション
- **CSV**: 構造化データの読み書き
- **NPZ**: NumPy配列の効率的保存

## システムの特徴

### 1. スケーラビリティ
- 数万点規模のデータセットに対応
- 階層フィルタリングによる計算量削減
- 表示要素数の動的制限

### 2. ユーザビリティ
- 直感的なマルチビュー連動
- 色による視覚的フィードバック
- レスポンシブなインタラクション

### 3. 拡張性  
- モジュラー設計による機能追加容易性
- 設定可能なパフォーマンスパラメータ
- プラグイン可能な類似度指標

### 4. 信頼性
- エラー処理とフォールバック機能
- データ整合性チェック
- デバッグ情報の充実

このシステムにより、複雑な階層クラスタリング結果を効率的に探索し、データの潜在的構造とクラスター関係を直感的に理解することが可能になります。
