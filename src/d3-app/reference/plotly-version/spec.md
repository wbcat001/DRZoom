Python版アプリケーション（Dash + Plotly）の機能一覧
📊 可視化・描画機能
DR散布図（UMAP埋め込み空間）
✅ 2D散布図表示（Plotlyグラフ）
✅ クラスタ別の色分け
✅ ラッソ選択（brush/lasso）とズーム対応
✅ インタラクションモード切替（brush vs zoom）
✅ ポイントホバーでカスタムデータ（ID・ラベル・クラスタ）表示
✅ クラスターラベルアノテーション追加（show_labels時）
✅ ズーム状態の保存・復元機能
デンドログラム可視化
✅ リンク行列からセグメント座標を計算
✅ HDBSCAN凝縮木をLinkage Matrix形式に変換
✅ 標準版: 均等なX座標配置
✅ サイズ重み付け版: クラスタサイズに応じた視覚的重み付け（対数スケール）
✅ 線幅でクラスターサイズを表現
✅ ホバーで詳細情報（Strahler値・Stability・代表単語5個）表示
✅ クリック対応（別途処理）
クラスター類似度ヒートマップ
✅ 複数距離指標対応
KL発散
Bhattacharyya係数
Mahalanobis距離
✅ 対話的な色スケール反転（Reverse Colorscale）
✅ クラスター自動再順序付け（Cluster Reorder）
✅ パフォーマンス制限（MAX_HEATMAP_CLUSTERS=200）
✅ 単一クラスター選択時の特別表示
✅ セル値表示（セルサイズに応じて）
🎨 ハイライト・色分けシステム
色設定管理
✅ デフォルト色（明るい青）
✅ 背景dimmed色（薄い青）
✅ DR選択色（オレンジ）
✅ ヒートマップクリック色（赤）
✅ ヒートマップ→DR連動色（ディープピンク）
✅ デンドログラム→DR連動色（ライムグリーン）
ハイライト優先度システム
✅ 複数選択状態の優先度制御（ヒートマップ→DR選択→その他）
✅ ポイント毎のハイライト判定
✅ Opacity動的調整（選択/非選択）
📈 データ処理・変換機能
HDBSCAN凝縮木処理
✅ HDBSCANの凝縮木をSciPy形式に変換（get_linkage_matrix_from_hdbscan）
✅ ノードID再マッピング（葉0～N-1、内部ノード連続ID）
✅ Lambda値→距離値の変換
✅ ペアビリティ検証（lambda値・parent ID一致確認）
メタデータ計算
✅ Strahler数計算: 階層複雑度指標（DFS・再帰版）
✅ Stability計算: クラスター品質スコア（lambda×size）
✅ クラスターサイズ計測: point_cluster_mapでポイント数カウント
デンドログラム座標計算
✅ 葉ノード順序決定（サイズベースソート）
✅ X座標計算（子の平均）
✅ U字型線分生成（icoord/dcoord）
✅ サイズ考慮版X座標配置（log正規化、幅可変）
🔗 インタラクション・選択管理
複数ビュー間の選択連動
✅ DR散布図のラッソ/ズーム選択 → クラスタIDリスト取得
✅ ヒートマップセルクリック → クラスタID取得
✅ デンドログラムクリック → クラスタID取得
✅ グローバルストア（selected-ids-store）で状態統一管理
選択フィルタリング
✅ 含有率閾値フィルタ: DR選択時にクラスタ含有率10%以下は除外（DR_SELECTION_CLUSTER_RATIO_THRESHOLD）
✅ ノイズポイント除外（cluster_id=-1）
✅ 複数選択状態の保持・更新
📋 詳細情報パネル
タブシステム（4タブ）
Point Details

✅ 選択ポイント情報（ID・ラベル）
✅ 座標表示
Selection Stats

✅ 選択ポイント数
✅ 選択クラスタ数
✅ クラスタ内ポイント総数
✅ カバレッジ率（%）
Cluster Size Distribution

✅ クラスターサイズ分布（散布図版）
✅ サイズランキング表示
System Log

✅ ログメッセージ出力
✅ パフォーマンス情報
クラスター詳細表示
✅ クラスターID・名称表示
✅ 代表単語リスト（最大10個）
✅ Stability値・Strahler数・サイズ表示
⚙️ パフォーマンス最適化
表示制限設定
✅ ENABLE_HEATMAP_CLUSTER_LIMIT: ヒートマップ表示クラスタ数制限有効化フラグ
✅ MAX_HEATMAP_CLUSTERS: 上限設定（デフォルト200）
✅ MAX_CLUSTER_WORDS_DISPLAY: 詳細パネル単語表示上限（10個）
✅ MAX_CLUSTER_WORDS_HOVER: ホバー単語表示上限（5個）
動的フィルタリング戦略
✅ 大規模ヒートマップの自動制限＆警告表示
✅ DR選択時のノイズ除外
✅ テキスト表示の最適化（省略表示）
🎛️ パラメータ設定UI
DR手法選択
✅ UMAP：n_neighbors, min_dist スライダー
✅ TSNE：perplexity スライダー
✅ PCA：n_components スライダー
フィルタコントロール
✅ Strahler値レンジスライダー
✅ Stability値レンジスライダー
✅ リアルタイム更新
オプションチェックボックス
✅ Reverse Colorscale（ヒートマップ）
✅ Cluster Reorder（ヒートマップ）
✅ Show Labels（デンドログラム・DR図）
✅ Size Weight（デンドログラム）
✅ Ignore Noise（DR選択フィルタ）
📊 データ読み込み機能
対応ファイル形式
✅ npz: UMAP埋め込み + ポイントラベル
✅ pkl: HDBSCAN凝縮木・クラスター類似度行列
✅ csv: クラスター代表ラベル・単語リスト
データキャッシング
✅ メモリ効率化（大規模データセット対応）
🔍 可視化ユーティリティ
座標計算関数群
✅ compute_dendrogram_coords(): 基本版座標計算
✅ compute_dendrogram_coords_with_size(): サイズ重み付け版
✅ get_dendrogram_segments(): セグメント生成
✅ plot_dendrogram_plotly(): Plotly描画
✅ plot_dendrogram_plotly_with_size(): サイズ対応版描画
✅ add_dendrogram_label_annotations(): ラベルアノテーション追加
✅ add_dr_cluster_label_annotations(): DR図へのラベル追加
ヘルパー関数
✅ get_clusters_from_points(): ポイント→クラスタID変換
✅ _get_leaves(): リーフノード抽出（DFS）
✅ _recurse_leaf_dfs(): 再帰DFS実装