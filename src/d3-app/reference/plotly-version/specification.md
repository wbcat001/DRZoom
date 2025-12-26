1. サーバー側の基本データ構造

| **データ名** | **サーバー側のデータソース** | **クライアントへの転送形式 (JSON)** | **備考** |
| --- | --- | --- | --- |
| **Points** | `embedding` + `labels` | Array of Objects: `[{x: 0.1, y: -0.2, c: 123, l: "word_A", i: 0}, ...]` | `i` (Index), `c` (Cluster ID), `l` (Label) を付与。約100K点。 |
| **Linkage Matrix** | `linkage_matrix` (Z) | Array of Arrays: `[[c1, c2, dist, count], ...]` | **デンドログラムの構造**。約10K行。**SciPy互換形式**で転送し、座標計算はクライアントへ委譲。 |
| **Cluster Meta** | `stability_dict` + `strahler_values` | Object (`Map`): `{ "cluster_id": {s: 0.5, h: 3, z: 1234}, ... }` | `s` (Stability), `h` (Strahler), `z` (Size/Count) など、フィルタリングに必要な**クラスタ単位のメタデータ**を統合。 |

| **データ名** | **サーバー側のデータソース** | **クライアントへのアクセスAPI** | **備考** |
| --- | --- | --- | --- |
| **Similarity Matrix** | `similarity_kl`, `similarity_bc`, etc. | `/api/heatmap?metric=kl&n=200` | **クライアントメモリ非保持**。リクエストに応じて**上位200クラスタ**にフィルタリングし、JSONで返却。 |
| **Cluster Details** | `cluster_word_lists` | `/api/cluster/{id}` | 選択された**単一クラスタの詳細情報**のみを返す。 |
| **Point-Cluster Map** | `point_cluster_map` (ポイントID $\to$ クラスタID) | サーバーメモリ保持 | DR選択後の**集計計算**（クラスタ含有率計算）のためにサーバー側で参照。 |

1. クライアント側の状態保持

| **ステート名** | **データ形式** | **更新トリガー** | **描画への影響** |
| --- | --- | --- | --- |
| **`selectedClusterIDs`** | `Set<Number>` | DR選択、デンドログラムクリック/ホバー、ヒートマップクリック | 全ビューの**ハイライトの色**を決定する最終的なソース。 |
| **`filterParams`** | `{stability: [min, max], strahler: [min, max]}` | スライダー操作 | デンドログラムの**表示構造**をフィルタリング。 |
| **`currentMetric`** | `String` (`'kl_divergence'`) | ドロップダウン操作 | ヒートマップのAPIリクエストパラメータ。 |
| **`dendrogramCoords`** | Array (線分データ) | `filterParams`の変更 | デンドログラムの**座標データ**。`Z_matrix`から動的に計算。 |

jsonのスキーマに

| **データ名** | **JSON Key** | **型** | **構造例** | **備考** |
| --- | --- | --- | --- | --- |
| **Points** | `points` | `Array<Object>` | `[{i: 0, x: 0.1, y: 0.2, c: 123, l: "word_A"}, ...]` | DRプロット用。`i`はインデックス、`c`はクラスタID。 |
| **Linkage** | `zMatrix` | `Array<Array>` | `[[c1, c2, dist, size], ...]` | デンドログラム構造。 |
| **Meta** | `clusterMeta` | `Object` (`Map`) | `{ "123": {s: 0.5, h: 3, z: 1234}, ...}` | Stability (`s`), Strahler (`h`), Size (`z`)。 |

| **API名** | **リクエスト形式** | **レスポンス形式** | **備考** |
| --- | --- | --- | --- |
| `POST /api/map_selection` | `{"pointIDs": [10, 20, 30, ...]}` | `{"clusterIDs": [123, 456], "stats": {...}}` | DR選択ポイントIDから**含有率フィルタリング済み**のクラスタIDリストを取得。 |
| `GET /api/heatmap` | `?metric=kl&n=200` | `{"matrix": [[0.1, 0.2, ...], ...], "clusterOrder": [123, 456, ...]}` | 上位Nクラスタの類似度行列を取得。`clusterOrder`で軸の順序を定義。 |
| `GET /api/cluster/{id}` | - | `{"id": 123, "label": "Topic A", "words": ["w1", "w2", ...]}` | 単一クラスタの詳細。 |

## 3. 処理の分担

| **処理内容** | **実行場所** | **備考** |
| --- | --- | --- |
| **デンドログラム座標計算** | **クライアント (d3.js)** | `Z`行列のフィルタリングと`compute_dendrogram_coords`のロジックをJSに移植し、高速な再描画を実現。 |
| **DR選択 $\to$ ClusterIDs** | **クライアント $\to$ サーバーAPI** | 1. クライアントが選択されたポイントIDリストを取得。 2. `/api/map_selection` に送信。 3. サーバーが`Point-Cluster Map`を用いて**含有率計算**と**フィルタリング**を実行。 4. サーバーがフィルタ済み`selectedClusterIDs`を返す。 |
| **ハイライト描画** | **クライアント (d3.js)** | `selectedClusterIDs`ストアを監視し、DR散布図とデンドログラムの**SVG要素のスタイル（色）を直接更新**。Plotlyオブジェクト全体を更新するオーバーヘッドを排除。 |
| **ヒートマップ描画** | **クライアント (d3.js)** | APIから取得した200x200の**類似度行列**データに基づき、d3.jsで高速なヒートマップ（SVGまたはCanvas）を描画。 |

## 4. レイアウトに関する情報

- plotly versionのレイアウト情報
    
    ```python
    # HDBSCAN階層クラスタリング可視化システム - レイアウト設計書
    
    ## レイアウト概要
    
    `app_labeled.py`のUIレイアウトは、Bootstrap Grid System（12列レイアウト）を基盤とした5エリア構成で設計されています。各エリアが連動して、HDBSCANクラスタリング結果の包括的な分析環境を提供します。
    
    ## レイアウト構造図
    
    ```
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           Main Container (fluid=True)                       │
    │ ┌───────────┬─────────────────────┬─────────────────────┬─────────────────┐ │
    │ │     A     │          B          │          C          │        D        │ │
    │ │  Control  │    DR Scatter       │    Dendrogram       │    Details      │ │
    │ │  Panel    │    Visualization    │     Hierarchy       │    & Info       │ │
    │ │ (Col: 2)  │     (Col: 4)        │     (Col: 4)        │   (Col: 2)      │ │
    │ │           │                     │                     │                 │ │
    │ └───────────┴─────────────────────┴─────────────────────┴─────────────────┘ │
    │ ┌─────────────────────────────────────┬─────────────────────────────────────┐ │
    │ │                E1                   │                E2                   │ │
    │ │       Cluster Heatmap               │       Cluster Details               │ │
    │ │         (Col: 6)                    │         (Col: 6)                    │ │
    │ │                                     │                                     │ │
    │ └─────────────────────────────────────┴─────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────────┘
    ```
    
    ## CSS高さ管理システム
    
    ### ビューポート全体使用の設定
    ```css
    html, body, #react-entry-point {
        height: 100%;
        margin: 0;
    }
    .dbc-container-fluid {
        height: 100%;
    }
    ```
    
    ### 高さ継承チェーン
    ```
    Viewport (100vh) 
        → Container (h-100) 
            → Row (h-100, g-0) 
                → Col (h-100, p-2) 
                    → Card (calc(100vh - 40px)) 
                        → CardBody (d-flex flex-column h-100)
    ```
    
    ## エリア別詳細設計
    
    ### A: Control Panel (width=2)
    
    **目的**: パラメータ制御とアプリケーション設定
    
    **構成要素:**
    ```python
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H4("Control Panel"),
                
                # データセット選択
                dcc.Dropdown(id='dataset-selector'),
                
                # DR手法選択  
                dbc.RadioItems(id='dr-method-selector'),
                
                # パラメータ設定（動的生成）
                html.Div(id='parameter-settings', className="flex-grow-1 overflow-auto"),
                
                # 実行ボタン
                dbc.Button(id='execute-button', className="mt-auto")
            ])
        ], className="h-100 d-flex flex-column")
    ], width=2, className="p-2 h-100")
    ```
    
    **レイアウト特徴:**
    - **縦方向Flexbox**: `d-flex flex-column`でコンテンツを縦配置
    - **自動拡張**: `flex-grow-1`でパラメータエリアが残りスペースを占有
    - **固定配置**: `mt-auto`で実行ボタンを下部に固定
    
    ### B: DR Visualization (width=4)
    
    **目的**: UMAP/t-SNE/PCA埋め込み空間の2D散布図表示
    
    **構成要素:**
    ```python
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H4("DR Visualization"),
                
                # インタラクション制御
                dbc.Row([
                    dbc.Col([
                        dbc.RadioItems(id='dr-interaction-mode-toggle', inline=True)
                    ], width=8),
                    dbc.Col([
                        dbc.Checklist(id='dr-label-annotation-toggle')
                    ], width=4)
                ]),
                
                # メインプロット
                dcc.Graph(id='dr-visualization-plot', className="flex-grow-1")
            ], className="d-flex flex-column p-3 h-100")
        ], style={'height': 'calc(100vh - 40px)'})
    ], width=4, className="p-2 h-100")
    ```
    
    **インタラクション機能:**
    - **Brush Selection**: ラッソ/ボックス選択によるマルチポイント選択
    - **Zoom/Pan**: 詳細領域の拡大・移動操作
    - **Label Annotation**: データポイントのラベル表示切り替え
    
    ### C: Dendrogram Hierarchy (width=4)
    
    **目的**: 階層クラスタ構造のデンドログラム可視化
    
    **構成要素:**
    ```python
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H4("Cluster Dendrogram"),
                
                # 3列制御パネル
                dbc.Row([
                    dbc.Col([
                        dbc.RadioItems(id='dendro-interaction-mode-toggle', inline=True)
                    ], width=4),
                    dbc.Col([
                        dbc.Checklist(id='dendro-width-option-toggle')
                    ], width=4),
                    dbc.Col([
                        dbc.Checklist(id='dendro-label-annotation-toggle')  
                    ], width=4)
                ]),
                
                # デンドログラムプロット
                dcc.Graph(id='dendrogram-plot', className="flex-grow-1")
            ], className="d-flex flex-column p-3 h-100")
        ], style={'height': 'calc(100vh - 40px)'})
    ], width=4, className="p-2 h-100")
    ```
    
    **表示オプション:**
    - **Node Selection**: ノードクリック・ホバーによる階層探索
    - **Proportional Width**: クラスターサイズに応じた線幅調整
    - **Label Annotation**: クラスター代表ラベル表示
    
    ### D: Details & Info Panel (width=2)
    
    **目的**: 選択要素の詳細情報とシステム状態表示
    
    **タブ構成:**
    ```python
    dbc.Tabs(
        id='detail-info-tabs',
        children=[
            dbc.Tab(label='Point Details', tab_id='tab-point-details'),
            dbc.Tab(label='Selection Stats', tab_id='tab-selection-stats'),
            dbc.Tab(label='Cluster Size Dist', tab_id='tab-cluster-size'),
            dbc.Tab(label='System Log', tab_id='tab-system-log')
        ]
    )
    ```
    
    **表示情報:**
    - **Point Details**: クリック点の詳細統計（ID、座標、ラベル、近傍情報）
    - **Selection Stats**: 選択領域の統計サマリ（点数、特徴量統計）
    - **Cluster Size Distribution**: クラスターサイズ分布の可視化
    - **System Log**: アプリケーション操作履歴
    
    ### E1: Cluster Heatmap (width=6) 
    
    **目的**: クラスター間類似度の行列表示
    
    **機能:**
    - **類似度指標**: KL距離、Bhattacharyya係数、Mahalanobis距離
    - **Colorscale制御**: 正規/逆転カラースケール切り替え
    - **Cluster Reorder**: 類似度に基づく階層的並び替え
    - **選択連動**: 他ビューの選択に応じたハイライト表示
    
    ### E2: Cluster Details (width=6)
    
    **目的**: 選択クラスターの詳細分析情報
    
    **表示内容:**
    - クラスターサイズと構成点数
    - 代表単語とキーワードリスト
    - Stabilityスコアと品質指標
    - 類似クラスターランキング
    
    ## データストア設計
    
    ### 選択状態管理
    ```python
    dcc.Store(id='selected-ids-store', data={
        'dr_selected_clusters': [],         # DRビューで選択されたクラスタ
        'dr_selected_points': [],           # DRビューで選択されたポイント
        'heatmap_clicked_clusters': [],     # ヒートマップでクリックされたクラスタ
        'heatmap_highlight_points': [],     # ヒートマップクリックによるハイライトポイント
        'dendrogram_clicked_clusters': [],  # デンドログラムでクリックされたクラスタ
        'dendrogram_highlight_points': [],  # デンドログラムクリックによるハイライトポイント
        'last_interaction_type': None       # インタラクション種別追跡
    })
    ```
    
    ### ズーム状態保持
    ```python
    dcc.Store(id='dr-zoom-store', data={
        'xaxis_range': None,
        'yaxis_range': None
    })
    ```
    
    ## 動的パラメータ設定システム
    
    ### UMAP パラメータUI
    ```python
    html.Div([
        html.Label("n_neighbors:"),
        dcc.Slider(
            id='umap-n-neighbors',
            min=5, max=50, step=1, value=15,
            marks={i: str(i) for i in range(5, 51, 10)},
            tooltip={'placement': 'bottom', 'always_visible': True}
        ),
        html.Label("min_dist:"),
        dcc.Slider(
            id='umap-min-dist', 
            min=0.0, max=0.99, step=0.01, value=0.1,
            marks={i/10: f"{i/10:.1f}" for i in range(0, 10, 2)},
            tooltip={'placement': 'bottom', 'always_visible': True}
        )
    ])
    ```
    
    ### t-SNE パラメータUI
    ```python
    html.Div([
        html.Label("perplexity:"),
        dcc.Slider(
            id='tsne-perplexity',
            min=5, max=50, step=1, value=30,
            marks={i: str(i) for i in range(5, 51, 10)},
            tooltip={'placement': 'bottom', 'always_visible': True}
        )
    ])
    ```
    
    ### PCA パラメータUI
    ```python
    html.Div([
        html.Label("n_components:"),
        dcc.Dropdown(
            id='pca-n-components',
            options=[
                {'label': '2', 'value': 2},
                {'label': '3', 'value': 3}
            ],
            value=2
        )
    ])
    ```
    
    ## レスポンシブデザイン戦略
    
    ### ブレークポイント対応
    - **Bootstrap Grid**: 自動的なカラム調整
    - **Fluid Container**: 画面幅に応じた伸縮
    - **Flexbox Layout**: コンテンツの動的配置
    
    ### 画面サイズ最適化
    ```python
    # 大画面用（デスクトップ）
    style={'height': 'calc(100vh - 40px)'}  # ビューポート最大活用
    
    # 中画面用（タブレット）  
    className="d-flex flex-column"  # 縦積みレイアウト
    
    # 小画面用（モバイル）
    className="overflow-auto"  # スクロール対応
    ```
    
    ## パフォーマンス考慮事項
    
    ### レンダリング最適化
    1. **グラフサイズ固定**: `calc(100vh - 40px)`で再描画最小化
    2. **Flexbox活用**: CSS Gridより軽量なレイアウト
    3. **オーバーフロー制御**: `overflow-auto`で部分スクロール
    
    ### メモリ効率化
    1. **Store分離**: 機能別データストアで更新範囲限定
    2. **条件付きレンダリング**: 表示状態に応じたコンポーネント生成
    3. **遅延読み込み**: 大型データの段階的ロード
    
    ## アクセシビリティ配慮
    
    ### キーボード操作
    - **Tab順序**: 論理的なフォーカス遷移
    - **ARIA属性**: スクリーンリーダー対応
    - **ショートカット**: 主要操作のキーボード割り当て
    
    ### 視覚的配慮
    - **高コントラスト**: 色覚異常への対応
    - **フォントサイズ**: 可読性確保
    - **色以外の区別**: 形状・パターンでの情報伝達
    
    ## 拡張性設計
    
    ### モジュラー構造
    ```python
    # エリア別コンポーネント関数化例
    def create_control_panel():
        return dbc.Col([...], width=2)
    
    def create_dr_visualization():  
        return dbc.Col([...], width=4)
    
    def create_dendrogram_area():
        return dbc.Col([...], width=4)
    
    def create_details_panel():
        return dbc.Col([...], width=2)
    ```
    
    ### 設定可能パラメータ
    - **カラム幅**: Grid System比率調整
    - **高さ設定**: ビューポート使用率変更
    - **レイアウトモード**: 縦積み/横並び切り替え
    
    ## ブラウザ互換性
    
    ### 対応ブラウザ
    - Chrome 80+（推奨）
    - Firefox 75+
    - Safari 13+  
    - Edge 80+
    
    ### 非対応時の代替表示
    ```python
    else:
        app.layout = html.Div([
            html.H3("Dash Bootstrap Components not available. Please install: pip install dash-bootstrap-components")
        ])
    ```
    
    このレイアウト設計により、複雑なHDBSCANクラスタリング結果を直感的に探索できる統合環境が実現されています。各エリアの独立性と連動性のバランスを取ることで、効率的なデータ分析ワークフローを支援します。
    
    ```
    

→ D3.js versionに

## 5. D3.jsの使いそうな機能

data bind

- **`selection.data()`**: データ配列をSVG要素やHTML要素に結合（バインド）し、データ駆動で要素を作成・更新・削除する基盤を確立します。
- **`selection.enter()`**: 新しいデータポイントに対応するプレースホルダー要素群を取得し、それらの要素を実際にDOMに追加する際に使用します。
- **`selection.exit()`**: データから削除された要素群を取得し、DOMからスムーズに削除（アニメーション含む）する際に使用します。
- **`selection.attr()` / `selection.style()`**: SVG要素の属性（例: `cx`, `cy`, `d`）やCSSスタイル（例: `fill`, `opacity`）をデータに基づいて設定・更新します。特に**ハイライトの色変更**を高速に行う際に使用します。

scale

- **`d3.scaleLinear()`**: データ空間（ドメイン）の数値を画面ピクセル空間（レンジ）に線形変換します。DR散布図やデンドログラムのX軸・Y軸の変換に使用します。ポイントID (`d.i`)
- **`d3.scaleOrdinal()`**: 離散的な値（クラスタIDなど）を、色やシンボルといった離散的な視覚的要素にマッピングします。クラスタごとの**一貫した色分け**に使用します。
- **`d3.scaleSequential()`**: 連続的な数値（類似度、Stabilityスコア）を、一連のグラデーションカラー（例: `d3.interpolateViridis`）にマッピングします。ヒートマップの色の濃淡や Stability に基づく表現に使用します。
- **`d3.axisTop()` / `d3.axisBottom()`**: スケール設定に基づき、目盛り線、数値ラベル、軸線を含む完全なSVGの軸を自動的に描画します。

draw

- **`d3.line()`**: 座標の配列（`[{x: 10, y: 20}, ...]`）を受け取り、デンドログラムの線分やグラフの線に必要なSVGパスデータ文字列 (`<path d="...">`) を生成します。
- **`d3.symbol()`**: 散布図マーカー（円、四角、カスタムシェイプなど）のパスデータを生成します。
- **`d3.path()`**: d3のAPIを使用して、複数の線や曲線を手動で結合し、複雑なSVGパスを効率的に構築します。デンドログラムのU字型描画ロジック内で役立ちます。

interaction

- **`selection.on(event, handler)`**: SVG要素に対してクリック、マウスオーバー、ドラッグなどのイベントハンドラを設定します。ヒートマップの**セルクリック**やデンドログラムの**ノードホバー**を処理します。
- **`d3.zoom()`**: ズームとパン（移動）の動作をコンテナ要素にバインドし、大規模データの探索を可能にします。
- **`d3.drag()`**: カスタムのドラッグ処理を実装するためのイベントハンドラを提供します。DR散布図での**ラッソ選択**機能の実装基盤として使用できます。
- **`d3.brush()`**: 矩形領域による効率的な選択（ボックス選択）のための標準的なインタフェースを提供します。DR散布図での代替選択手法として利用できます。
- `d3.transition()`: 色、座標などのアニメーション変化

other

- **`d3.json()` / `d3.fetch()`**: サーバーAPIエンドポイント（例: `/api/v1/initial_data`）からJSONデータを取得します。
- **`d3.max()` / `d3.min()`**: 配列内の最大値・最小値を迅速に計算し、スケールのドメイン設定やデータ統計処理に使用します。

## 6. 作るエンドポイント一覧

機能: 

- 初期状態のscatter, dendrogramの描画
- クリック時の情報取得 → 類似点、所属クラスタ、ラベル
- パラメータ選択(data, dr-method)
- dr上での点群選択→ dendorogmra(highlight), heatmap?, クラスタ情報
- dendrogram上での選択 → highlight
- heatmap上での選択 → highlight
- drのズーム:

エンドポイント

- /initial_data : option → point, zmatrix, cluster meta
- /point_to_cluster: 選択したポイントからクラスタidを取得
- /heatmap: クラスタid集合から類似度行列を取得？
- /cluster: クラスタの詳細情報取得
- /point: ポイントの詳細情報取得(近傍)
- /about_data: データセットに関する情報を取得？

- /recalculate_scatter: points → drの座標
- /recalculate_dendrogram: param → zmatrix
- /similarity_color: simialrityをHSVに射影して色を決定
- /search: キーワードで特定の点を検索