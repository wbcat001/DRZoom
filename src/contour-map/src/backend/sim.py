import numpy as np
import pandas as pd
import hdbscan
import umap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from typing import List, Tuple
# mnist
from sklearn.datasets import fetch_openml
# --- 1. データ生成とHDBSCANの実行 ---

# 階層的な構造を持つダミーデータを生成
X, y = make_blobs(n_samples=2000, centers=8, cluster_std=1.0, random_state=42)
# さらにクラスタを近づけて複雑な階層構造を作る
X[:500, :] += 5 
X[500:1000, :] -= 5

# HDBSCANの実行
hdb = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5, prediction_data=True).fit(X)

# UMAPによる初期の2次元埋め込み (全データ対象)
mapper = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit(X)
initial_embedding = mapper.embedding_

# HDBSCANのデンドログラムからシミュレーション用のλ値を定義
# λが小さいほど粗いクラスタリング (Overview)
condensed_tree = hdb.condensed_tree_.to_pandas()
min_lambda = condensed_tree['lambda_val'].min()
max_lambda = condensed_tree['lambda_val'].max()
print(f"Lambda range: {min_lambda} to {max_lambda}")

# 3つのズームレベルに対応するλ値を設定 (Overview, Mid, Details)
lambda_levels = {
    "Level 1: Overview (粗い)": min_lambda + (max_lambda - min_lambda) * 0.1,
    "Level 2: Mid-Zoom (中間)": min_lambda + (max_lambda - min_lambda) * 0.4,
    "Level 3: Details (詳細)": min_lambda + (max_lambda - min_lambda) * 0.8
}

df_full = pd.DataFrame(initial_embedding, columns=['x', 'y'])
df_full['index'] = df_full.index


# --- 2. 分析ロジックとプロット関数の定義 ---

def get_snapshot_data(X: np.ndarray, current_lambda: float, level_name: str) -> pd.DataFrame:
    """
    指定されたlambda値でクラスタリングを行い、代表点を抽出し、UMAPを再計算する (シミュレーション)。
    
    Args:
        X: 高次元データ
        current_lambda: HDBSCANのカットレベル (lambda値)
        level_name: プロット用のレベル名
        
    Returns:
        プロット用のDataFrame
    """
    
    # 1. 指定lambda値でのクラスタリングを取得
    # hdbscan.label.hdbscan_tree_to_labelsは内部関数であり、ここではシミュレーションとして
    # 簡略化されたラベル抽出ロジックを使用
    labels, _ = hdbscan.label.hdbscan_tree_to_labels(
        hdb.condensed_tree_, X.shape[0], lambda_val=current_lambda
    )
    
    # 2. 代表点（コア点）の抽出 (シミュレーション)
    representative_indices = []
    unique_clusters = np.unique(labels[labels != -1])
    
    for cid in unique_clusters:
        member_indices = np.where(labels == cid)[0]
        if len(member_indices) > 0:
            # 実際の代表点ロジック: Core Distance最小の点を抽出
            core_distances = hdb.core_distances_[member_indices]
            core_point_index_in_member = np.argmin(core_distances)
            representative_indices.append(member_indices[core_point_index_in_member])

    if not representative_indices:
        # 代表点がない場合、全点を使用 (フォールバック)
        representative_indices = np.arange(X.shape[0])
    
    # 3. 代表点のみに限定したUMAPの再計算 (または以前の結果からの初期配置)
    # ここでは、**代表点のみ**を使ってUMAPを再計算する**「意味的ズーム」**をシミュレーション
    X_rep = X[representative_indices]
    
    # UMAPの初期配置を前の埋め込み結果から継承する処理は複雑なため省略し、ここでは新規計算
    mapper_rep = umap.UMAP(n_neighbors=5, min_dist=0.1, random_state=42).fit(X_rep)
    embedding_rep = mapper_rep.embedding_
    
    # 4. DataFrameの作成
    df_rep = pd.DataFrame(embedding_rep, columns=['x', 'y'])
    df_rep['cluster'] = labels[representative_indices]
    df_rep['is_rep'] = True
    df_rep['level'] = level_name
    
    return df_rep

def plot_snapshots(df_list: List[pd.DataFrame]):
    """複数のスナップショットをPlotlyで描画"""
    fig = go.Figure()
    
    # 1. 全てのデータを一つのDataFrameに結合
    df_combined = pd.concat(df_list)
    
    # 2. Plotlyのサブプロット設定（1行3列）
    for i, (level, df) in enumerate(zip(lambda_levels.keys(), df_list)):
        row = 1
        col = i + 1

        # 3. 散布図の追加
        fig.add_trace(
            go.Scatter(
                x=df['x'],
                y=df['y'],
                mode='markers',
                name=level,
                marker=dict(
                    size=10, 
                    # クラスタIDで色付け
                    color=df['cluster'],
                    colorscale='Turbo', 
                    showscale=False,
                    line=dict(width=1, color='DarkSlateGrey') 
                ),
                hovertemplate=f"Cluster: %{{customdata[0]}}<br>Index: %{{customdata[1]}}<extra>{level}</extra>",
                customdata=np.stack((df['cluster'], df['index'] if 'index' in df.columns else np.arange(len(df))), axis=-1),
            ),
            row=row,
            col=col,
            # タイトルは後のレイアウトで設定
        )

        # 4. レイアウトの調整
        fig.update_xaxes(title_text="UMAP Dimension 1", row=row, col=col, showgrid=False, zeroline=False)
        fig.update_yaxes(title_text="UMAP Dimension 2", row=row, col=col, showgrid=False, zeroline=False)

    # 5. グローバルレイアウト
    fig.update_layout(
        title_text="階層的ドリルダウン分析のスナップショット",
        height=500,
        showlegend=False,
        # サブプロットのレイアウトを定義 (1行3列)
        grid={'rows': 1, 'columns': 3, 'pattern': 'independent'}, 
    )
    
    # 各プロットにタイトルを適用
    annotations = []
    for i, level in enumerate(lambda_levels.keys()):
        annotations.append(dict(
            xref='x domain', yref='y domain',
            x=0.5 + i, y=1.05,
            text=f"**{level}** ({len(df_list[i])} 代表点)",
            showarrow=False,
            font=dict(size=14),
            xanchor='center',
            yanchor='bottom'
        ))
    fig.update_layout(annotations=annotations)
    
    return fig


from plotly.subplots import make_subplots

# --- 3. スナップショットの生成 ---
snapshot_data_list = []
for level_name, lambda_val in lambda_levels.items():
    df_snapshot = get_snapshot_data(X, lambda_val, level_name)
    df_snapshot['index'] = df_snapshot.index # インデックスを一旦保持
    snapshot_data_list.append(df_snapshot)

# --- 4. 結果の描画 ---
# Plotlyのサブプロット機能を使って3つの図を並列表示

# サブプロットキャンバスの準備
fig = make_subplots(rows=1, cols=3, 
                    subplot_titles=list(lambda_levels.keys()),
                    horizontal_spacing=0.05)

# 各レベルのデータをプロットに追加
for i, df in enumerate(snapshot_data_list):
    level_name = list(lambda_levels.keys())[i]
    
    # Scatter Trace の設定
    trace = go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(
            size=10, 
            # クラスタIDで色付け
            color=df['cluster'],
            colorscale='Turbo',
            colorbar=dict(title='Cluster ID', titleside='right') if i == 2 else None, # 最後のプロットのみカラーバー表示
            showscale=False,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        name=level_name,
        hovertemplate=f"Cluster: %{{customdata[0]}}<br>Point: %{{customdata[1]}}<extra>{level_name}</extra>",
        customdata=np.stack((df['cluster'], df['index']), axis=-1)
    )
    
    fig.add_trace(trace, row=1, col=i + 1)
    
    # サブプロットタイトルを更新 (Plotly Subplotsの都合上、後で更新)
    fig.layout.annotations[i].update(text=f"**{level_name}** ({len(df)} 代表点)")

# グローバルレイアウトの調整
fig.update_layout(
    title_text="HDBSCAN階層を利用した意味的ズーム分析スナップショット",
    height=500,
    width=1200,
    showlegend=False,
)

fig.update_xaxes(showgrid=False, zeroline=False)
fig.update_yaxes(showgrid=False, zeroline=False)

fig.show()