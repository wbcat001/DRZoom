import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
import dash_bootstrap_components as dbc
import os
import pickle

# クラスタリング用のimport（オプション）
try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available, cluster reordering will be disabled")
    SCIPY_AVAILABLE = False

np.set_printoptions(suppress=True, precision=4)

###########################################################################
# Color Configuration for Highlighting
HIGHLIGHT_COLORS = {
    'default': '#4A90E2',          # 明るい青 - デフォルト状態
    'default_dimmed': '#B8D4F0',   # 薄い青 - heatmap選択時の背景
    'dr_selection': '#FFA500',     # orange - DR選択時
    'heatmap_click': '#FF0000',    # red - heatmapクリック時
    'heatmap_to_dr': '#FF1493',   # ディープピンク - heatmapからDRへのハイライト
    'dendrogram_to_dr': '#32CD32'  # ライムグリーン - デンドログラムからDRへのハイライト
}

###########################################################################
# Performance Configuration
ENABLE_HEATMAP_CLUSTER_LIMIT = True  # Heatmapクラスタ数制限の有効/無効
MAX_HEATMAP_CLUSTERS = 200           # Heatmapで表示可能な最大クラスタ数

# DR Selection Filtering Configuration
DR_SELECTION_CLUSTER_RATIO_THRESHOLD = 0.1  # DRラッソ選択時のクラスタ含有率閾値（10%未満は除外）

# Cluster Words Display Configuration
MAX_CLUSTER_WORDS_DISPLAY = 10  # Detail panelで表示する最大単語数
MAX_CLUSTER_WORDS_HOVER = 5     # デンドログラムホバーで表示する最大単語数

###########################################################################
# helpers
def _get_leaves(condensed_tree):
    cluster_tree = condensed_tree[condensed_tree['child_size'] > 1]
    print(len(cluster_tree))
    if cluster_tree.shape[0] == 0:
        # Return the only cluster, the root
        return [condensed_tree['parent'].min()]

    root = cluster_tree['parent'].min()
    return _recurse_leaf_dfs(cluster_tree, root)
  
def _recurse_leaf_dfs(cluster_tree, current_node):
  children = cluster_tree[cluster_tree['parent'] == current_node]['child']
  if len(children) == 0:
      return [current_node,]
  else:
      return sum([_recurse_leaf_dfs(cluster_tree, child) for child in children], [])
  
def get_leaves(cluster_tree):
    """
    cluster_tree: (u, v, lambda_val, child_size, parent)
    """
    root = cluster_tree[:, 2].max()
    print(f"root: {root}")
    return recurse_leaf_dfs(cluster_tree, root)
    

def recurse_leaf_dfs(cluster_tree, current_node):
    # print(f"Visiting Node: {current_node}")
    child1 = cluster_tree[cluster_tree[:,2] == current_node][:,0]
    child2 = cluster_tree[cluster_tree[:,2] == current_node][:,1]
    # print(f"Children of Node {current_node}: Child1 {child1}, Child2 {child2}")

    if len(child1) == 0 and len(child2) == 0:
        
        return [current_node,]
    else:
        return sum([recurse_leaf_dfs(cluster_tree, child) for child in np.concatenate((child1, child2))], [])


def get_linkage_matrix_from_hdbscan(condensed_tree):
    """
    (child1, child2, parent, lambda_val, count)
    """
    print("Generating linkage matrix from HDBSCAN condensed tree...")
    linkage_matrix = []
    raw_tree = condensed_tree._raw_tree
    condensed_tree = condensed_tree.to_pandas()
    cluster_tree = condensed_tree[condensed_tree['child_size'] > 1]
    sorted_condensed_tree = cluster_tree.sort_values(by=['lambda_val','parent'], ascending=True)
    print(f"len of sorted condensed tree: {len(sorted_condensed_tree)}")

    for i in range(0, len(sorted_condensed_tree), 2):
    
        # 偶数行（i）と次の奇数行（i+1）をペアとして取得
        if i + 1 < len(sorted_condensed_tree):
            
            row_a = sorted_condensed_tree.iloc[i]
            row_b = sorted_condensed_tree.iloc[i+1]
            
            # **前提チェック**: lambda_valが同じであることを確認
            if row_a['lambda_val'] != row_b['lambda_val']:
                # lambda_valが異なる場合は、次の処理に進む（結合の前提が崩れる）
                raise ValueError(f"Lambda value mismatch at rows {i} and {i+1}: {row_a['lambda_val']} vs {row_b['lambda_val']}")
                
            # Parent IDが同じであることを確認 (同じ結合の結果である可能性が高い)
            if row_a['parent'] != row_b['parent']:
                # Parent IDが異なる場合は、このペアは単一の結合ではない可能性が高い
                raise ValueError(f"Parent ID mismatch at rows {i} and {i+1}: {row_a['parent']} vs {row_b['parent']}")
            
            child_a = row_a['child']
            child_b = row_b['child']
            lam = row_a['lambda_val']
            
            # count (サイズ) は、結合された2つの子ノードのサイズ合計を使うのが論理的だが、
            # HDBSCANは親ノードのサイズをリストで持っているため、ここではそのサイズを使用
            # より正確には、このParent IDを持つ全子ノードのサイズの合計を使うべきだが、
            # 2行の child_size の合計で暫定的に対応
            # total_size = row_a['child_size'] + row_b['child_size']


            total_size = raw_tree[raw_tree['child'] == row_a['parent']]['child_size']
            if len(total_size) == 0:
                total_size = row_a['child_size'] + row_b['child_size']
            else:
                total_size = total_size[0]
            # print(total_size)
            parent_id = row_a['parent']

            linkage_matrix.append([
                int(child_a), 
                int(child_b), 
                int(parent_id),
                lam, 
                total_size,
        ])   
    print(f"len of linkage matrix: {len(linkage_matrix)}")


    # 葉ノードに0-N-1のIDを振る
    node_id_map = {}
    current_id = 0
    leaves = _get_leaves(raw_tree)
    print(f"Number of leaves: {len(leaves)}")

    for leaf in leaves:
        node_id_map[int(leaf)] = current_id
        current_id += 1

    print(f"Leaf ID Map Size: {len(node_id_map)}")
    print(f"current id: {current_id}")

    # 結合ノードにIDを振る(linkage matrixのparent)
    for row in linkage_matrix.__reversed__():
        parent_id = row[2]
        if parent_id not in node_id_map:
            node_id_map[parent_id] = current_id
            current_id += 1

        else:
            print(f"Duplicate Parent ID found: {parent_id}")
            raise ValueError(f"Node ID {parent_id} already assigned!")
    print(f"Total Node ID Map Size: {len(node_id_map)}")
    print(f"current_id: {current_id}")

    # linkage matrixを書き換え
    max_lambda = max(row[3] for row in linkage_matrix)
    print(f"Max Lambda: {max_lambda}")
    linkage_matrix_mapped = [ 
        [node_id_map[row[0]], node_id_map[row[1]], node_id_map[row[2]], max_lambda - row[3], row[4]] 
        for row in linkage_matrix.__reversed__()
    ]

    return np.array(linkage_matrix_mapped), node_id_map # linkage matrix, parentid -> newid





def compute_dendrogram_coords(Z, n_points):
    """
    Linkage Matrixからデンドログラム描画用の座標を計算
    Z: (n_merges x 4) array like [c1, c2, dist, count]
    n_points: 葉の数

    Returns: icoord, dcoord, leaf_order
    """
    # --- 1. ノード情報の準備 ---
    n_nodes = 2 * n_points - 1
    nodes = [{'x': None, 'y': 0.0, 'size': 1, 'left': None, 'right': None} for _ in range(n_points)]

    # Z の各行は c1, c2, dist, count
    for i in range(n_points - 1):
        c1, c2, dist, count = Z[i]
        nodes.append({
            'x': None,
            'y': float(dist),
            'size': int(count),
            'left': int(c1),
            'right': int(c2)
        })

    def get_leaf_order_sorted(node_idx):
        node = nodes[node_idx]
        if node_idx < n_points:
            return [node_idx]
        C1_idx, C2_idx = node['left'], node['right']
        size_C1, size_C2 = nodes[C1_idx]['size'], nodes[C2_idx]['size']
        if size_C1 < size_C2:
            C1_idx, C2_idx = C2_idx, C1_idx
        order_left = get_leaf_order_sorted(C1_idx)
        order_right = get_leaf_order_sorted(C2_idx)
        return order_left + order_right

    def calculate_x_coord(node_idx, leaf_to_x):
        node = nodes[node_idx]
        if node_idx < n_points:
            x_coord = leaf_to_x[node_idx]
            node['x'] = x_coord
            return x_coord
        x_left = calculate_x_coord(node['left'], leaf_to_x)
        x_right = calculate_x_coord(node['right'], leaf_to_x)
        x_coord = (x_left + x_right) / 2.0
        node['x'] = x_coord
        return x_coord

    root_node_idx = n_points - 1 + (n_points - 1)
    leaf_order = get_leaf_order_sorted(root_node_idx)
    leaf_to_x = {leaf_idx: 2 * i + 1 for i, leaf_idx in enumerate(leaf_order)}
    calculate_x_coord(root_node_idx, leaf_to_x)

    icoord = []
    dcoord = []
    for i in range(n_points - 1):
        P = n_points + i
        C1 = nodes[P]['left']
        C2 = nodes[P]['right']
        y_P = nodes[P]['y']
        y_C1 = nodes[C1]['y']
        y_C2 = nodes[C2]['y']
        x_P = nodes[P]['x']
        x_C1 = nodes[C1]['x']
        x_C2 = nodes[C2]['x']
        icoord.append([x_C1, x_C1, x_C2, x_C2])
        dcoord.append([y_C1, y_P, y_P, y_C2])

    return icoord, dcoord, leaf_order


def compute_dendrogram_coords_with_size(Z, n_points):
    """
    サイズ考慮版：Linkage Matrixからデンドログラム描画用の座標を計算
    Z: (n_merges x 4) array like [c1, c2, dist, count]
    n_points: 葉の数
    クラスタサイズにX座標を反映

    Returns: icoord, dcoord, leaf_order
    """
    print(f"check Z head sample:\n{Z[:5]}")
    # --- 1. ノード情報の準備 ---
    n_nodes = 2 * n_points - 1
    
    # 各葉ノードを初期化（サイズ情報は後で設定）
    nodes = [{'x': None, 'y': 0.0, 'size': 1, 'left': None, 'right': None} for _ in range(n_points)]
    
    # 葉ノードのサイズをraw_treeから取得して設定
    global raw_tree  # raw_treeがグローバルに定義されているためアクセス
    leaf_entries = raw_tree[raw_tree['child_size'] == 1]  # 葉ノード（サイズ1のエントリ）
    
    # new_id -> old_id マッピングを使って葉ノードサイズを設定
    for i in range(n_points):
        if i in new_old_id_map:
            old_id = new_old_id_map[i]
            # このold_idに対応するraw_treeエントリを探す
            matching_entries = raw_tree[raw_tree['child'] == old_id]
            if len(matching_entries) > 0:
                # 実際には葉ノードは通常サイズ1だが、そのクラスタが表現するポイント数を使用
                # point_cluster_mapでこのクラスタに属するポイント数をカウント
                cluster_id = old_id
                point_count = sum(1 for pid, cid in point_cluster_map.items() if cid == cluster_id)
                nodes[i]['size'] = max(1, point_count)  # 最低でも1
            else:
                nodes[i]['size'] = 1

    # Z の各行は c1, c2, dist, count
    for i in range(n_points - 1):
        c1, c2, dist, count = Z[i]
        nodes.append({
            'x': None,
            'y': float(dist),
            'size': int(count),
            'left': int(c1),
            'right': int(c2)
        })

    def get_leaf_order_sorted(node_idx):
        node = nodes[node_idx]
        if node_idx < n_points:
            return [node_idx]
        C1_idx, C2_idx = node['left'], node['right']
        size_C1, size_C2 = nodes[C1_idx]['size'], nodes[C2_idx]['size']
        if size_C1 < size_C2:
            C1_idx, C2_idx = C2_idx, C1_idx
        order_left = get_leaf_order_sorted(C1_idx)
        order_right = get_leaf_order_sorted(C2_idx)
        return order_left + order_right

    def calculate_x_coord_with_size(node_idx):
        """サイズを考慮したX座標計算（葉ノードにサイズ幅、内部ノードは平均）"""
        node = nodes[node_idx]
        if node_idx < n_points:
            # 葉ノードの場合：既に設定済みのX座標を返す
            return node['x']
        
        # 内部ノードの場合：左右の子の平均
        left_idx = node['left']
        right_idx = node['right']
        
        x_left = calculate_x_coord_with_size(left_idx)
        x_right = calculate_x_coord_with_size(right_idx)
        
        # シンプルに左右の平均
        x_coord = (x_left + x_right) / 2.0
        node['x'] = x_coord
        return x_coord

    # 1. 葉の順序を取得
    root_node_idx = n_points - 1 + (n_points - 1)
    leaf_order = get_leaf_order_sorted(root_node_idx)
    
    # 2. 葉ノードにサイズに応じた幅を割り当て（logスケール）
    import math
    
    # 幅の設定
    min_width = 0.5   # 最小幅
    max_width = 8.0   # 最大幅
    
    # 各葉ノードのサイズを取得
    leaf_sizes = [nodes[leaf_idx]['size'] for leaf_idx in leaf_order]
    min_size = min(leaf_sizes)
    max_size = max(leaf_sizes)
    print(f"Leaf sizes: min={min_size}, max={max_size}")
    
    current_x = 0.0
    
    for leaf_idx in leaf_order:
        leaf_size = nodes[leaf_idx]['size']
        
        if max_size > min_size:
            # logスケールで正規化 (log(size) - log(min)) / (log(max) - log(min))
            log_normalized = (math.log(leaf_size) - math.log(min_size)) / (math.log(max_size) - math.log(min_size))
            # 最小幅から最大幅の範囲にマッピング
            width = min_width + (max_width - min_width) * log_normalized
        else:
            # 全て同じサイズの場合は平均幅を使用
            width = (min_width + max_width) / 2.0
        
        # 葉ノードの中心位置を設定
        nodes[leaf_idx]['x'] = current_x + width / 2.0
        current_x += width
    
    # 3. 上位ノードのX座標を計算（左右の子の平均）
    calculate_x_coord_with_size(root_node_idx)

    icoord = []
    dcoord = []
    for i in range(n_points - 1):
        P = n_points + i
        C1 = nodes[P]['left']
        C2 = nodes[P]['right']
        y_P = nodes[P]['y']
        y_C1 = nodes[C1]['y']
        y_C2 = nodes[C2]['y']
        x_P = nodes[P]['x']
        x_C1 = nodes[C1]['x']
        x_C2 = nodes[C2]['x']
        icoord.append([x_C1, x_C1, x_C2, x_C2])
        dcoord.append([y_C1, y_P, y_P, y_C2])

    return icoord, dcoord, leaf_order


def get_dendrogram_segments_with_size(Z: np.ndarray):
    """サイズ考慮版：Linkage Matrixから描画用セグメントを生成。"""
    n_points = Z.shape[0] + 1
    print(f"n_points (size-aware): {n_points}")
    icoord, dcoord, leaf_order = compute_dendrogram_coords_with_size(Z, n_points)
    segments = []
    for icoords, dcoords in zip(icoord, dcoord):
        x1, x2, x3, x4 = icoords
        y1, y2, y3, y4 = dcoords
        segments.append([(x1, y1), (x2, y2)])
        segments.append([(x2, y2), (x3, y3)])
        segments.append([(x4, y4), (x3, y3)])
    return segments


def get_dendrogram_segments(Z: np.ndarray):
    """Linkage Matrixから描画用セグメントを生成。簡易実装。"""
    n_points = Z.shape[0] + 1
    print("n_points:", n_points)
    icoord, dcoord, leaf_order = compute_dendrogram_coords(Z, n_points)
    segments = []
    for icoords, dcoords in zip(icoord, dcoord):
        x1, x2, x3, x4 = icoords
        y1, y2, y3, y4 = dcoords
        segments.append([(x1, y1), (x2, y2)])
        segments.append([(x2, y2), (x3, y3)])
        segments.append([(x4, y4), (x3, y3)])
    return segments





def plot_dendrogram_plotly(segments, 
                           colors=None, 
                           scores=None, 
                           is_selecteds=None,
                           is_heatmap_clicked=None,
                           **kwargs):
    """Plotlyを使用したデンドログラムの描画（segmentsは線分タプルのリスト）"""

    # kwargsに含まれる情報が有効かチェック
    additional_data = []
    
    # kwargs内のリストも展開して保存
    for key, value_list in kwargs.items():
        
        # ✅ 有効なリストは3倍に展開してadditional_dataに追加
    
        additional_data.append((key, value_list))
        print(f"length of expanded {key}: {len(value_list)}")
    
  
    fig = go.Figure()
    for i, seg in enumerate(segments):
        index = i // 3  # 3つのセグメントごとに1つのクラスタに対応
       
        x_coords = [seg[0][0], seg[1][0]]
        y_coords = [seg[0][1], seg[1][1]]
        # 色設定の優先順位: heatmapクリック > DR選択 > デフォルト
        opacity = 1.0
        
        # heatmapクリックが最優先
        if is_heatmap_clicked is not None and is_heatmap_clicked[index]:
            color = HIGHLIGHT_COLORS['heatmap_click']
        elif is_selecteds is not None and is_selecteds[index]:
            color = HIGHLIGHT_COLORS['dr_selection']
        else:
            color = HIGHLIGHT_COLORS['default']

        hover_lines = []
        for key, value_list in additional_data:
            value = value_list[index] if index < len(value_list) else "N/A"
            if isinstance(value, float):
                hover_lines.append(f"{key}: {value:.4f}")
            else:
                hover_lines.append(f"{key}: {value}")
        
        # クラスタ内の単語をホバーに追加
        if index < len(linkage_matrix):
            cluster_id = new_old_id_map.get(int(linkage_matrix[index][2]), None)
            if cluster_id and cluster_id in cluster_word_lists:
                words = cluster_word_lists[cluster_id][:MAX_CLUSTER_WORDS_HOVER]
                words_text = ', '.join(words)
                if len(cluster_word_lists[cluster_id]) > MAX_CLUSTER_WORDS_HOVER:
                    words_text += f" (+{len(cluster_word_lists[cluster_id]) - MAX_CLUSTER_WORDS_HOVER} more)"
                hover_lines.append(f"Words: {words_text}")
        
        full_hover_text = '<br>'.join(hover_lines)
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color=color, width=1),
            showlegend=False,
            hoverinfo='text' if (is_selecteds is None or is_selecteds[index]) else 'skip',
            text=[full_hover_text] * len(x_coords),
            opacity=opacity
        ))

    return fig


def plot_dendrogram_plotly_with_size(segments, 
                                    colors=None, 
                                    scores=None, 
                                    is_selecteds=None,
                                    is_heatmap_clicked=None,
                                    cluster_sizes=None,
                                    **extra_info_kwargs):
    """サイズ考慮版：Plotlyを使用したデンドログラムの描画（線幅でクラスタサイズを表現）"""
    fig = go.Figure()
    
    # クラスタサイズから線幅を計算
    line_widths = [2] * len(segments)  # デフォルト幅
    if cluster_sizes is not None and len(cluster_sizes) > 0:
        # 対数スケールで正規化（1-8pxの範囲）
        min_size = min(cluster_sizes)
        max_size = max(cluster_sizes)
        
        if max_size > min_size:
            # 各セグメントに対応する線幅を計算（3つのセグメントごとに1つのクラスタ）
            for i in range(len(segments)):
                cluster_index = i // 3  # 3つのセグメントごとに1つのクラスタに対応
                if cluster_index < len(cluster_sizes):
                    size = cluster_sizes[cluster_index]
                    # 対数正規化で線幅を計算
                    norm_factor = np.log(size/min_size + 1) / np.log(max_size/min_size + 1)
                    width = 1 + norm_factor * 7  # 1-8pxの範囲
                    line_widths[i] = width
    
    for i, seg in enumerate(segments):
        index = i // 3  # 3つのセグメントごとに1つのクラスタに対応
        
        x_coords = [seg[0][0], seg[1][0]]
        y_coords = [seg[0][1], seg[1][1]]
        
        # 基本色とサイズ情報
        color = "blue"
        line_width = line_widths[i] if i < len(line_widths) else 2
        opacity = 1.0
        
        # 色設定の優先順位: heatmapクリック > DR選択 > デフォルト
        if is_heatmap_clicked is not None and is_heatmap_clicked[index]:
            color = HIGHLIGHT_COLORS['heatmap_click']
        elif is_selecteds is not None and is_selecteds[index]:
            color = HIGHLIGHT_COLORS['dr_selection']
        else:
            color = HIGHLIGHT_COLORS['default']

        hover_lines = []
        # サイズ情報をホバーテキストに追加
        if cluster_sizes is not None and index < len(cluster_sizes):
            hover_lines.append(f"Size: {cluster_sizes[index]}")

        # その他の追加情報をホバーテキストに追加（plot_dendrogram_plotlyと同じロジック）
        for key, value_list in extra_info_kwargs.items():
            value = value_list[index] if index < len(value_list) else "N/A"
            if isinstance(value, float):
                hover_lines.append(f"{key}: {value:.4f}")
            else:
                hover_lines.append(f"{key}: {value}")
        
        # クラスタ内の単語をホバーに追加
        if index < len(linkage_matrix):
            cluster_id = new_old_id_map.get(int(linkage_matrix[index][2]), None)
            if cluster_id and cluster_id in cluster_word_lists:
                words = cluster_word_lists[cluster_id][:MAX_CLUSTER_WORDS_HOVER]
                words_text = ', '.join(words)
                if len(cluster_word_lists[cluster_id]) > MAX_CLUSTER_WORDS_HOVER:
                    words_text += f" (+{len(cluster_word_lists[cluster_id]) - MAX_CLUSTER_WORDS_HOVER} more)"
                hover_lines.append(f"Words: {words_text}")
        
        full_hover_text = '<br>'.join(hover_lines)
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color=color, width=line_width),
            showlegend=False,
            hoverinfo='text' if (is_selecteds is None or is_selecteds[index]) else 'skip',
            text=[full_hover_text] * len(x_coords),
            opacity=opacity
        ))
    
    return fig


def get_clusters_from_points(point_ids, point_id_map):
    """選択されたポイントIDから対応するクラスタIDリストを返す（重複除去）"""
    cluster_ids = set()
    for pid in point_ids:
        if pid in point_id_map:
            cluster_ids.add(point_id_map[pid])
    return list(cluster_ids)

def add_dendrogram_label_annotations(fig, linkage_matrix, segments):
    """デンドログラムにクラスタラベルのアノテーションを追加する"""
    
    # 各結合ノード（クラスタ）の座標を計算
    # segmentsから結合点の座標を抽出
    for i, row in enumerate(linkage_matrix):
        if i * 3 + 1 < len(segments):  # 3つのセグメントごとに1つのクラスタ
            # 結合点（親ノード）の座標を取得
            # segments[i*3+1]が水平線分で、その中点が結合点
            horizontal_segment = segments[i * 3 + 1]
            x_coord = (horizontal_segment[0][0] + horizontal_segment[1][0]) / 2
            y_coord = horizontal_segment[0][1]  # 水平線のY座標
            
            # クラスタIDを取得
            cluster_id = new_old_id_map.get(int(row[2]), None)
            
            if cluster_id and cluster_id in cluster_representative_labels:
                representative_label = cluster_representative_labels[cluster_id]
                
                # "Cluster_"で始まるデフォルトラベルは除外
                if not representative_label.startswith('Cluster_'):
                    # アノテーションを追加
                    fig.add_annotation(
                        x=x_coord,
                        y=y_coord,
                        text=representative_label,
                        showarrow=False,
                        font=dict(size=8, color='darkblue'),
                        bgcolor='rgba(255,255,255,0.7)',
                        bordercolor='lightgray',
                        borderwidth=1,
                        xshift=0,
                        yshift=10  # テキストを結合点の少し上に表示
                    )

def add_dr_cluster_label_annotations(fig, df, point_cluster_map, cluster_representative_labels, cluster_word_lists):
    """DRプロットにクラスタラベルのアノテーションを追加する（簡素版）"""
    
    try:
        # 各クラスタの中心座標を計算
        cluster_centers = {}
        
        # 各クラスタに属するポイントの座標を集計
        for point_id, cluster_id in point_cluster_map.items():
            if point_id < len(df):  # 有効なポイントIDのみ
                if cluster_id not in cluster_centers:
                    cluster_centers[cluster_id] = {'x_sum': 0, 'y_sum': 0, 'count': 0}
                
                cluster_centers[cluster_id]['x_sum'] += df.iloc[point_id]['x']
                cluster_centers[cluster_id]['y_sum'] += df.iloc[point_id]['y']
                cluster_centers[cluster_id]['count'] += 1
        
        # サイズでソートし、上位のクラスタのみ表示（パフォーマンス向上）
        sorted_clusters = sorted(cluster_centers.items(), key=lambda x: x[1]['count'], reverse=True)
        max_annotations = min(1000, len(sorted_clusters))  # 最大50個まで
        
        annotation_count = 0
        for cluster_id, center_data in sorted_clusters:
            if annotation_count >= max_annotations:
                break
                
            if center_data['count'] >= 10:  # 最小10ポイント以上のクラスタのみ表示
                center_x = center_data['x_sum'] / center_data['count']
                center_y = center_data['y_sum'] / center_data['count']
                
                # 表示するラベルを決定（ラベルがない場合は表示しない）
                display_label = None
                
                # 1. 代表ラベルがある場合はそれを使用（"Cluster_"で始まるものは除外）
                if cluster_id in cluster_representative_labels:
                    representative_label = cluster_representative_labels[cluster_id]
                    if not representative_label.startswith('Cluster_'):
                        display_label = representative_label
                
                # 2. 代表ラベルがない場合はクラスタの先頭単語を使用
                if display_label is None and cluster_id in cluster_word_lists:
                    cluster_words = cluster_word_lists[cluster_id]
                    if cluster_words and len(cluster_words) > 0:
                        display_label = cluster_words[0]  # 先頭単語を使用
                
                # ラベルがある場合のみアノテーションを追加
                if display_label is not None:
                    fig.add_annotation(
                        x=center_x,
                        y=center_y,
                        text=display_label,
                        showarrow=False,
                        font=dict(size=15, color='darkred'),
                        bgcolor='rgba(255,255,255,0.7)',
                        bordercolor='darkred',
                        borderwidth=1,
                        xshift=0,
                        yshift=0
                    )
                    annotation_count += 1
    
    except Exception as e:
        print(f"Error in DR annotation: {e}")
        # エラーが発生してもfigを返すことができるよう、何もしない

def compute_stability(condensed_tree):

    # 1. 最小クラスタとクラスタ数を定義 (Cythonと同じロジック)
    smallest_cluster = condensed_tree['parent'].min()
    num_clusters = condensed_tree['parent'].max() - smallest_cluster + 1
    
    largest_child = max(condensed_tree['child'].max(), smallest_cluster)

    # 2. lambda_birth の計算 (クラスタの誕生時の最小 lambda)
    # condensed_tree を 'child' でソート
    sorted_child_data = np.sort(condensed_tree[['child', 'lambda_val']], axis=0)
    
    # births_arr は、child ID に対応する lambda_birth を保持する
    births_arr = np.nan * np.ones(largest_child + 1, dtype=np.double)
    
    current_child = -1
    min_lambda = 0

    # NumPyの structured array を Pythonループで処理 (Cythonの loopを模倣)
    for row in range(sorted_child_data.shape[0]):
        child = sorted_child_data[row]['child']
        lambda_ = sorted_child_data[row]['lambda_val']

        if child == current_child:
            min_lambda = min(min_lambda, lambda_)
        elif current_child != -1:
            births_arr[current_child] = min_lambda
            current_child = child
            min_lambda = lambda_
        else:
            # Initialize
            current_child = child
            min_lambda = lambda_

    if current_child != -1:
        births_arr[current_child] = min_lambda
        
    births_arr[smallest_cluster] = 0.0 # ルートクラスタの lambda_birth は 0
    
    # 3. Stability スコアの計算
    
    # NumPyのベクトル演算で高速化可能だが、Cythonを模倣しループで計算
    result_arr = np.zeros(num_clusters, dtype=np.double)
    
    parents = condensed_tree['parent']
    sizes = condensed_tree['child_size']
    lambdas = condensed_tree['lambda_val']

    for i in range(condensed_tree.shape[0]):
        parent = parents[i]
        lambda_ = lambdas[i]
        child_size = sizes[i]
        result_index = parent - smallest_cluster
        
        # Stability(C) = Σ (lambda_death - lambda_birth) * size
        # condensed_treeの各行は、parent が child_size のクラスタを 'lambda_' で吸収/結合するステップを示す
        # この lambda_ は、HDBSCANロジックでは lambda_death と見なされる
        
        lambda_birth = births_arr[parent]
        
        # NOTE: HDBSCANのStability定義は複雑なため、ここはHDBSCANの内部ロジックを正確に模倣する必要があります。
        # オリジナルのCythonコードを再現:
        result_arr[result_index] += (lambda_ - lambda_birth) * child_size
        
    # 4. ID とスコアを辞書に変換
    node_ids = np.arange(smallest_cluster, condensed_tree['parent'].max() + 1)
    result_pre_dict = np.vstack((node_ids, result_arr)).T

    # フィルタリングされていないノードを含むため、dictに変換してIDとスコアを対応させる
    # [ID, Score] のペアの配列を辞書に変換
    return dict(zip(result_pre_dict[:, 0].astype(int), result_pre_dict[:, 1]))



#########################################################################
# データ準備
_base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"_base_dir: {_base_dir}")
data_file_path = "src/experiments/18_rapids/result/20251203_053328/embedding.npz"# os.path.join(_base_dir, 'data', 'dr_dummy_data.csv')
word_file_path = "src/experiments/18_rapids/result/20251203_053328/data.npz"# os.path.join(_base_dir, 'data', 'dr_dummy_data.csv')


label_file_path = "src/experiments/22_word_label/processed_data/w2v_words_100000.txt"# os.path.join(_base_dir, 'data', 'dr_dummy_data.csv')
hdbscan_condensed_tree_file_path = "src/experiments/18_rapids/result/20251203_053328/condensed_tree_object.pkl" # os.path.join(_base_dir, 'data', 'hdbscan_condensed_tree.csv')

# クラスタラベルファイルパス
cluster_label_file_path = "src/experiments/19_tree/processed_data/cluster_to_label.csv"  # クラスタID -> 代表単語ラベルのマッピング

# with open(label_file_path, 'r', encoding='utf-8') as f:
#     true_labels = [line.strip() for line in f.readlines()]
# print(f"Loaded true labels from {label_file_path}, length: {len(true_labels)}, sample: {true_labels[:5]}")

true_labels = np.load(word_file_path)['words'].tolist()

# クラスタ代表単語ラベルの読み込み
cluster_representative_labels = {}  # cluster_id -> representative_label
cluster_word_lists = {}  # cluster_id -> [word1, word2, ...]
try:
    import csv
    if os.path.exists(cluster_label_file_path):
        with open(cluster_label_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # ヘッダー行をスキップ
            for row in reader:
                if len(row) >= 2:
                    cluster_id = int(row[0])
                    representative_label = row[1]
                    cluster_representative_labels[cluster_id] = representative_label
                    
                    # クラスタ内の単語リスト（3列目以降があれば）
                    if len(row) > 2:
                        words = [word.strip() for word in row[2:] if word.strip()]
                        cluster_word_lists[cluster_id] = words
        print(f"Loaded cluster labels from {cluster_label_file_path}: {len(cluster_representative_labels)} clusters")
    else:
        print(f"Cluster label file not found: {cluster_label_file_path}")
        # デモ用のサンプルラベルを作成
        for cluster_id in range(100):  # 最初の100クラスタにサンプルラベル
            cluster_representative_labels[cluster_id] = f"Cluster_{cluster_id}"
            cluster_word_lists[cluster_id] = [f"word{cluster_id}_{i}" for i in range(5)]
except Exception as e:
    print(f"Error loading cluster labels: {e}")
    cluster_representative_labels = {}
    cluster_word_lists = {}

similarity_file_path = "src/experiments/19_tree/processed_data/cluster_similarities.pkl"
with open(similarity_file_path, 'rb') as f:
    similarity = pickle.load(f)
similarity_kl = similarity["kl_divergence"]
similarity_bc = similarity["bhattacharyya_coefficient"]
similarity_m = similarity["mahalanobis_distance"]

# HDBSCAN condensed tree
with open(hdbscan_condensed_tree_file_path, 'rb') as f:
    hdbscan_condensed_tree = pickle.load(f)
print(f"Loaded HDBSCAN condensed tree data from {hdbscan_condensed_tree_file_path}")

# Embedding(UMAP)
data = np.load(data_file_path)
embedding = data['embedding']
labels = data['words']
print(f"Loaded embedding data from {data_file_path}, shape: {data['embedding'].shape}")
print(f'words sample: {labels[:5]}  ')
# linkage matrix
linkage_matrix, old_new_id_map = get_linkage_matrix_from_hdbscan(hdbscan_condensed_tree)
print(f"prepared linkage matrix, shape: {linkage_matrix.shape}")
new_old_id_map = {v: k for k, v in old_new_id_map.items()}
print(f"linkage matrix: {linkage_matrix[:5]}")

raw_tree = hdbscan_condensed_tree._raw_tree
leaf_rows = raw_tree[raw_tree['child_size'] == 1]
point_cluster_map = {int(row['child']): int(row['parent']) for row in leaf_rows}
print(f"prepared point to cluster map, size: {len(point_cluster_map)}")

stability_dict = compute_stability(raw_tree)
print(f"prepared stability dict, size: {len(stability_dict)}")




extra_info_kwargs = {
    "cluster parent" : [new_old_id_map[int(row[2])] for row in linkage_matrix],
    "cluster child 1": [new_old_id_map[int(row[0])] for row in linkage_matrix],
    "cluster child 2": [new_old_id_map[int(row[1])] for row in linkage_matrix],
    "Cluster Size": [row[4] for row in linkage_matrix],
    "Stability": [stability_dict[new_old_id_map[int(row[2])]] if new_old_id_map[int(row[2])] in stability_dict else 0.0 for row in linkage_matrix],
    "Representative Label": [cluster_representative_labels.get(new_old_id_map[int(row[2])], "No Label") for row in linkage_matrix]
}
print(f'extra_info head sample: {extra_info_kwargs["Cluster Size"][:5]}')

df = pd.DataFrame({
    "x": embedding[:,0],
    "y": embedding[:,1],
    "label": true_labels,
    "cluster_label": labels,
    "cluster_id": [point_cluster_map[pid] for pid in point_cluster_map.keys()]
})

# # 修正版：リーフノードのみのクラスタサイズ分布
# def get_leaf_cluster_sizes():
#     """リーフノード（最終クラスタ）のサイズのみを取得"""
#     # point_cluster_mapから各クラスタに属するポイント数をカウント
#     leaf_cluster_sizes = {}
#     for point_id, cluster_id in point_cluster_map.items():
#         if cluster_id in leaf_cluster_sizes:
#             leaf_cluster_sizes[cluster_id] += 1
#         else:
#             leaf_cluster_sizes[cluster_id] = 1
    
#     return list(leaf_cluster_sizes.values())

# # リーフクラスタサイズの分布を取得
# leaf_sizes = get_leaf_cluster_sizes()
# print(f"Leaf cluster count: {len(leaf_sizes)}")
# print(f"Leaf cluster sizes sample: {sorted(leaf_sizes, reverse=True)[:10]}")

# # ヒストグラム作成
# fig_cluster_hist = px.histogram(
#     [size for size in leaf_sizes if size <= 1000], 
#     nbins=min(50, len(set(leaf_sizes))), 
#     title="Leaf Cluster Size Distribution",
#     labels={"value": "Cluster Size", "count": "Number of Leaf Clusters"}
# )
# データ準備の部分で一度だけ実行
leaf_cluster_sizes = {}
for point_id, cluster_id in point_cluster_map.items():
    leaf_cluster_sizes[cluster_id] = leaf_cluster_sizes.get(cluster_id, 0) + 1

# サイズでソート
sorted_sizes = sorted(leaf_cluster_sizes.values(), reverse=True)
ranks = list(range(1, len(sorted_sizes) + 1))

# 散布図
fig_cluster_hist = px.scatter(
    x=ranks,
    y=sorted_sizes,
    title="Cluster Rank vs Size"
)


# DBCが利用可能かチェック（Canvas環境では通常利用可能）
DBC_AVAILABLE = True 

# Dashアプリケーション初期化
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP] 
)

# グラフ無限拡大を防ぐためのHTML/Bodyの高さ設定
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <style>
            /* ビューポートの高さを確実に使用するように設定 */
            html, body, #react-entry-point {
                height: 100%;
                margin: 0;
            }
            .dbc-container-fluid {
                height: 100%;
            }
        </style>
    </body>
</html>
'''

# レイアウト定義（DBC版 - グラフ無限拡大対策済み）
if DBC_AVAILABLE:
    app.layout = dbc.Container([
        # データストア（選択されたクラスタID情報を保持）
        dcc.Store(id='selected-ids-store', data={
            'dr_selected_clusters': [],         # DRビューで選択されたクラスタ
            'dr_selected_points': [],           # DRビューで選択されたポイント
            'heatmap_clicked_clusters': [],     # heatmapビューでクリックされたクラスタ
            'heatmap_highlight_points': [],     # heatmapクリックによるハイライトポイント
            'dendrogram_clicked_clusters': [],  # デンドログラムでクリックされたクラスタ
            'dendrogram_highlight_points': [],  # デンドログラムクリックによるハイライトポイント
            'last_interaction_type': None       # 'dr_selection' or 'heatmap_click' or 'dendrogram_click'
        }),
        
        # DRプロットのズーム状態保存用ストア
        dcc.Store(id='dr-zoom-store', data={
            'xaxis_range': None,
            'yaxis_range': None
        }),
        # Rowに h-100 を適用し、親コンテナ (dbc.Container) の高さを継承
        dbc.Row([
            # A: Control Panel (Col: 2)
            dbc.Col([
                # h-100 と d-flex flex-column で高さ管理
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Control Panel", className="text-center mb-3"),
                        
                        # データセット選択
                        dbc.Row([
                            dbc.Label("Dataset:", html_for="dataset-selector", className="fw-bold"),
                            dcc.Dropdown(
                                id='dataset-selector',
                                options=[
                                    {'label': 'Iris', 'value': 'iris'},
                                    {'label': 'Digits', 'value': 'digits'},
                                    {'label': 'Wine', 'value': 'wine'}
                                ],
                                value='iris'
                            )
                        ], className="mb-3"),
                        
                        # 次元削減手法選択
                        dbc.Row([
                            dbc.Label("DR Method:", className="fw-bold"),
                            dbc.RadioItems(
                                id='dr-method-selector',
                                options=[
                                    {'label': 'UMAP', 'value': 'UMAP'},
                                    {'label': 'TSNE', 'value': 'TSNE'},
                                    {'label': 'PCA', 'value': 'PCA'}
                                ],
                                value='UMAP'
                            )
                        ], className="mb-3"),
                        
                        # パラメータ設定
                        dbc.Row([
                            dbc.Label("Parameters:", className="fw-bold"),
                            # flex-grow-1: 残りのスペースを占有
                            html.Div(id='parameter-settings', className="flex-grow-1 overflow-auto") 
                        ], className="mb-3 d-flex flex-column"), 
                        
                        # 実行ボタン
                        dbc.Button(
                            'Run Analysis',
                            id='execute-button',
                            n_clicks=0,
                            color="primary",
                            size="lg",
                            className="w-100 mt-auto" # mt-autoで実行ボタンを下に寄せる
                        )
                    ])
                ], className="h-100 d-flex flex-column"), # h-100でRowの高さを継承
            ], width=2, className="p-2 h-100"), # Colにも h-100 が必要
            
            # B: DR View Area (Col: 4)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("DR Visualization", className="text-center mb-1"),
                        
                        # インタラクションモード切替
                        dbc.Row([
                            dbc.Col([
                                dbc.RadioItems(
                                    id='dr-interaction-mode-toggle',
                                    options=[
                                        {'label': 'Brush Selection', 'value': 'brush'},
                                        {'label': 'Zoom/Pan', 'value': 'zoom'}
                                    ],
                                    value='zoom',
                                    inline=True
                                )
                            ], width=8),
                            dbc.Col([
                                dbc.Checklist(
                                    id='dr-label-annotation-toggle',
                                    options=[{'label': 'Show Labels', 'value': 'show_labels'}],
                                    value=[],
                                )
                            ], width=4)
                        ], className="mb-2"),
                        
                        # DR可視化プロット
                        # h-100とflex-grow-1で、親CardBody内の残りのスペースを占有
                        dcc.Graph(id='dr-visualization-plot', className="flex-grow-1") 
                    ], className="d-flex flex-column p-3 h-100")
                # calc(100vh - 40px) は、ビューポート全体からRowの上下のパディングを引いたおおよその高さ
                ], style={'height': 'calc(100vh - 40px)'}, className="h-100"), 
            ], width=4, className="p-2 h-100"),
            
            # C: Dendrogram Area (Col: 4)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Cluster Dendrogram", className="text-center mb-1"),
                        
                        # インタラクションモード切替とオプション
                        dbc.Row([
                            dbc.Col([
                                dbc.RadioItems(
                                    id='dendro-interaction-mode-toggle',
                                    options=[
                                        {'label': 'Node Selection', 'value': 'node'},
                                        {'label': 'Zoom/Pan', 'value': 'zoom'}
                                    ],
                                    value='node',
                                    inline=True
                                )
                            ], width=4),
                            dbc.Col([
                                dbc.Checklist(
                                    id='dendro-width-option-toggle',
                                    options=[{'label': 'Proportional Width', 'value': 'prop_width'}],
                                    value=[],
                                )
                            ], width=4),
                            dbc.Col([
                                dbc.Checklist(
                                    id='dendro-label-annotation-toggle',
                                    options=[{'label': 'Show Labels', 'value': 'show_labels'}],
                                    value=[],
                                )
                            ], width=4)
                        ], className="mb-2"),
                        
                        # デンドログラムプロット
                        dcc.Graph(id='dendrogram-plot', className="flex-grow-1")
                    ], className="d-flex flex-column p-3 h-100")
                ], style={'height': 'calc(100vh - 40px)'}, className="h-100"),
            ], width=4, className="p-2 h-100"),
            
            # D: Detail & Info Panel (Col: 2)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Detail & Info", className="text-center mb-3"),
                        
                        # タブコンポーネント
                        dbc.Tabs(
                            id='detail-info-tabs',
                            active_tab='tab-point-details',
                            children=[
                                dbc.Tab(label='Point Details', tab_id='tab-point-details'),
                                dbc.Tab(label='Selection Stats', tab_id='tab-selection-stats'),
                                dbc.Tab(label='Cluster Size Dist', tab_id='tab-cluster-size'),
                                dbc.Tab(label='System Log', tab_id='tab-system-log')
                            ]
                        ),
                        
                        # タブコンテンツ
                        html.Div(id='detail-panel-content', className="mt-3 flex-grow-1 overflow-auto")
                    ], className="d-flex flex-column p-3 h-100")
                ], style={'height': 'calc(100vh - 40px)'}, className="h-100"),
            ], width=2, className="p-2 h-100")
        ], className="g-0 h-100"), # Rowのパディング(p-2)が相殺されるように、Rowの外側のマージンをg-0でなくす
        
        # E: Cluster Info Area
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Cluster Information", className="text-center mb-1"),
                        
                        # ヒートマップオプション
                        dbc.Row([
                            dbc.Col([
                                dbc.Checklist(
                                    id='heatmap-reverse-colorscale',
                                    options=[{'label': 'Reverse Colorscale', 'value': 'reverse'}],
                                    value=[],
                                    inline=True
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Checklist(
                                    id='heatmap-cluster-reorder',
                                    options=[{'label': 'Cluster Reorder', 'value': 'reorder'}],
                                    value=[],
                                    inline=True
                                )
                            ], width=6)
                        ], className="mb-2"),
                        
                        # クラスタ情報グラフ
                        dcc.Graph(id='cluster-info', className="flex-grow-1")
                    ], className="d-flex flex-column p-3 h-100")
                ], style={'height': '400px'}, className="h-100"),
            ], width=6, className="p-2"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Cluster Details", className="text-center mb-1"),
                        
                        # クラスタ詳細情報
                        html.Div(id='cluster-details-content', className="flex-grow-1 overflow-auto")
                    ], className="d-flex flex-column p-3 h-100")
                ], style={'height': '400px'}, className="h-100"),
            ], width=6, className="p-2")
        ], className="g-0")
    ], fluid=True, className="h-100")
else:
    # DBC が利用できない場合の従来のレイアウト
    app.layout = html.Div([
        html.H3("Dash Bootstrap Components not available. Please install: pip install dash-bootstrap-components")
    ])

# コールバック関数定義（以前のロジックを維持）

@app.callback(
    Output('parameter-settings', 'children'),
    Input('dr-method-selector', 'value')
)
def update_parameter_settings(method):
    """選択された次元削減手法に応じてパラメータ設定UIを更新"""
    if method == 'UMAP':
        return html.Div([
            html.Label("n_neighbors:"),
            dcc.Slider(
                id='umap-n-neighbors',
                min=5, max=50, step=1, value=15,
                marks={i: str(i) for i in range(5, 51, 10)},
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
            html.Br(),
            html.Label("min_dist:"),
            dcc.Slider(
                id='umap-min-dist',
                min=0.0, max=0.99, step=0.01, value=0.1,
                marks={i/10: f"{i/10:.1f}" for i in range(0, 10, 2)},
                tooltip={'placement': 'bottom', 'always_visible': True}
            )
        ])
    elif method == 'TSNE':
        return html.Div([
            html.Label("perplexity:"),
            dcc.Slider(
                id='tsne-perplexity',
                min=5, max=50, step=1, value=30,
                marks={i: str(i) for i in range(5, 51, 10)},
                tooltip={'placement': 'bottom', 'always_visible': True}
            )
        ])
    elif method == 'PCA':
        return html.Div([
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
    return html.Div()

# グラフ描画（パフォーマンス最適化版）
# DRプロットのズーム状態を保存するコールバック
@app.callback(
    Output('dr-zoom-store', 'data'),
    [Input('dr-visualization-plot', 'relayoutData')],
    [State('dr-zoom-store', 'data')]
)
def save_dr_zoom_state(relayoutData, current_zoom_state):
    """DRプロットのズーム・パン状態を保存"""
    if relayoutData is None:
        return current_zoom_state or {'xaxis_range': None, 'yaxis_range': None}
    
    # 現在のズーム状態を取得（Noneチェック）
    if current_zoom_state is None:
        current_zoom_state = {'xaxis_range': None, 'yaxis_range': None}
    
    # ズーム・パン操作を検出してレンジを保存
    updated_zoom_state = current_zoom_state.copy()
    
    # X軸のレンジ更新
    if 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
        updated_zoom_state['xaxis_range'] = [relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
    elif 'xaxis.range' in relayoutData:
        updated_zoom_state['xaxis_range'] = relayoutData['xaxis.range']
    
    # Y軸のレンジ更新
    if 'yaxis.range[0]' in relayoutData and 'yaxis.range[1]' in relayoutData:
        updated_zoom_state['yaxis_range'] = [relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']]
    elif 'yaxis.range' in relayoutData:
        updated_zoom_state['yaxis_range'] = relayoutData['yaxis.range']
    
    # オートスケールの場合はレンジをクリア
    if relayoutData.get('xaxis.autorange') or relayoutData.get('yaxis.autorange'):
        if relayoutData.get('xaxis.autorange'):
            updated_zoom_state['xaxis_range'] = None
        if relayoutData.get('yaxis.autorange'):
            updated_zoom_state['yaxis_range'] = None
    
    print(f"DR zoom state updated: x_range={updated_zoom_state['xaxis_range']}, y_range={updated_zoom_state['yaxis_range']}")
    
    return updated_zoom_state

@app.callback(
    Output('dr-visualization-plot', 'figure'),
    [Input('execute-button', 'n_clicks'),
     Input('dr-method-selector', 'value'),
     Input('dr-interaction-mode-toggle', 'value'),
     Input('dr-label-annotation-toggle', 'value'),
     Input('selected-ids-store', 'data')],
    [State('dr-zoom-store', 'data')],
    prevent_initial_call=False
)
def update_dr_plot(n_clicks, method, interaction_mode, label_options, stored_data, zoom_state):
    """DR可視化プロットを更新（軽量版・ズーム状態保持）"""
    # ストアから選択情報を取得
    if stored_data:
        dr_selected_points = stored_data.get('dr_selected_points', [])
        heatmap_highlight_points = stored_data.get('heatmap_highlight_points', [])
        dendrogram_highlight_points = stored_data.get('dendrogram_highlight_points', [])
    else:
        dr_selected_points = []
        heatmap_highlight_points = []
        dendrogram_highlight_points = []
    
    has_dr_selection = len(dr_selected_points) > 0
    has_heatmap_selection = len(heatmap_highlight_points) > 0
    has_dendrogram_selection = len(dendrogram_highlight_points) > 0
    
    print(f"DR plot update: {len(dr_selected_points)} DR selected, {len(heatmap_highlight_points)} heatmap highlight, {len(dendrogram_highlight_points)} dendrogram highlight points")
    
    # 4種類の状態に応じてマーカーの色、サイズ、opacityを設定（優先度順）
    colors = []
    sizes = []
    opacities = []
    
    for i in range(len(df)):
        if i in heatmap_highlight_points:
            # heatmap選択ハイライト: 最優先、大きく明るく表示
            colors.append(HIGHLIGHT_COLORS['heatmap_to_dr'])
            sizes.append(6)  # 最大サイズ
            opacities.append(0.95)  # 最高透明度
        elif i in dendrogram_highlight_points:
            # デンドログラム選択ハイライト: 2番目の優先度
            colors.append(HIGHLIGHT_COLORS['dendrogram_to_dr'])
            sizes.append(5)  # 大サイズ
            opacities.append(0.9)  # 高透明度
        elif i in dr_selected_points:
            # DRラッソ選択: 3番目の優先度
            colors.append(HIGHLIGHT_COLORS['dr_selection'])
            sizes.append(4)  # 中サイズ
            opacities.append(0.8)  # 中程度の透明度
        elif has_heatmap_selection or has_dendrogram_selection or has_dr_selection:
            # 何らかの選択がある時の背景ポイント: 薄く表示
            colors.append(HIGHLIGHT_COLORS['default_dimmed'])
            sizes.append(1.5)  # 小サイズ
            opacities.append(0.15)  # 低透明度
        else:
            # 通常状態: 標準的な表示
            colors.append(HIGHLIGHT_COLORS['default'])
            sizes.append(2.5)  # 標準サイズ
            opacities.append(0.7)  # 標準透明度
    
    # px.scatterでベースを作成（pxのスタイルを維持）
    fig = px.scatter(
        df,
        x='x',
        y='y',
        hover_data=['label'],
        custom_data=['cluster_label', 'label']  # true_labelとcluster_labelの両方を含める
    )
    
    # マーカーの色、サイズ、opacityを詳細設定
    fig.update_traces(
        marker=dict(
            size=sizes,
            color=colors,
            opacity=opacities,
            line=dict(
                width=0.3,  # 細い枠線
                color='white'  # 白い枠線でクリアな表示
            )
        )
    )
    
    # cluster_label=-1（ノイズポイント）のホバーを無効化するためのhoverinfoリストを作成
    hoverinfo_list = []
    for i in range(len(df)):
        cluster_label = df.iloc[i]['cluster_label']
        if cluster_label == -1:
            hoverinfo_list.append('skip')  # ノイズポイントはホバーをスキップ
        else:
            hoverinfo_list.append('all')   # 通常のポイントは全情報を表示
    
    # ホバーテンプレートをカスタマイズしてラベル情報と選択状態を表示
    hover_template = (
        '<b>True Label: %{customdata[1]}</b><br>'
        'Cluster Label: %{customdata[0]}<br>'
        'X: %{x:.4f}<br>'
        'Y: %{y:.4f}<br>'
    )
    
    # 選択状態がある場合は選択情報も追加
    if has_dr_selection or has_heatmap_selection or has_dendrogram_selection:
        hover_template += '<b>Selection Status: %{text}</b><br>'
        hover_text = []
        for i in range(len(df)):
            if i in heatmap_highlight_points:
                hover_text.append('Heatmap Highlighted')
            elif i in dendrogram_highlight_points:
                hover_text.append('Dendrogram Highlighted')
            elif i in dr_selected_points:
                hover_text.append('DR Selected')
            else:
                hover_text.append('Not Selected')
        hover_template += '<extra></extra>'
        fig.update_traces(hovertemplate=hover_template, text=hover_text, hoverinfo=hoverinfo_list)
    else:
        hover_template += '<extra></extra>'
        fig.update_traces(hovertemplate=hover_template, hoverinfo=hoverinfo_list)
    
    # レイアウト設定
    title_parts = []
    if has_dr_selection:
        title_parts.append(f"DR: {len(dr_selected_points)}")
    if has_heatmap_selection:
        title_parts.append(f"Heatmap: {len(heatmap_highlight_points)}")
    if has_dendrogram_selection:
        title_parts.append(f"Dendrogram: {len(dendrogram_highlight_points)}")
    
    title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""
    
    # ズーム状態を適用
    layout_updates = {
        'showlegend': False,  # 凡例無効化
        'xaxis_title': 'Dimension 1',
        'yaxis_title': 'Dimension 2',
        'title': f'DR Visualization{title_suffix}',
        'margin': dict(l=10, r=10, t=35, b=10),  # タイトル用に上マージンを少し増やし
        'plot_bgcolor': '#f8f9fa',  # コンテナの色に合わせて背景色統一
        'paper_bgcolor': '#f8f9fa'  # 外枠背景色統一
    }
    
    # 保存されたズーム状態があれば適用
    if zoom_state:
        if zoom_state.get('xaxis_range'):
            layout_updates['xaxis'] = dict(range=zoom_state['xaxis_range'], title='Dimension 1')
        if zoom_state.get('yaxis_range'):
            layout_updates['yaxis'] = dict(range=zoom_state['yaxis_range'], title='Dimension 2')
        print(f"Applying saved zoom state: x_range={zoom_state.get('xaxis_range')}, y_range={zoom_state.get('yaxis_range')}")
    
    fig.update_layout(**layout_updates)
    
    # インタラクションモードに応じた設定
    if interaction_mode == 'brush':
        drag_mode = 'lasso' # "brush"
    else:
        drag_mode = 'zoom'
    
    fig.update_layout(dragmode=drag_mode)
    
    # クラスタラベルアノテーションを追加（show_labelsがTrueの場合）
    show_labels = 'show_labels' in label_options
    if show_labels:
        add_dr_cluster_label_annotations(fig, df, point_cluster_map, cluster_representative_labels, cluster_word_lists)
        print("DR cluster labels annotated")
    else:
        print("DR cluster labels not annotated")
    
    return fig



# 選択状態を更新するコールバック（インタラクション -> ストア）
@app.callback(
    Output('selected-ids-store', 'data'),
    [
        Input("dr-visualization-plot", 'selectedData'),
        Input('cluster-info', 'clickData'),
        Input('dendrogram-plot', 'clickData')
    ],
    State('selected-ids-store', 'data')
)
def update_selected_clusters(selectedData, heatmapClickData, dendrogramClickData, current_state):
    """DR選択、heatmapクリック、デンドログラムクリックによる選択状態をストアに保存"""
    print("update_selected_clusters called")
    
    # 現在の状態を取得（Noneチェック）
    if current_state is None:
        current_state = {
            'dr_selected_clusters': [], 
            'dr_selected_points': [],
            'heatmap_clicked_clusters': [], 
            'heatmap_highlight_points': [],
            'dendrogram_clicked_clusters': [],
            'dendrogram_highlight_points': [],
            'last_interaction_type': None
        }
    
    # 既存の値を保持
    dr_selected_clusters = current_state.get('dr_selected_clusters', [])
    dr_selected_points = current_state.get('dr_selected_points', [])
    heatmap_clicked_cluster_ids = current_state.get('heatmap_clicked_clusters', [])
    heatmap_highlight_points = current_state.get('heatmap_highlight_points', [])
    dendrogram_clicked_clusters = current_state.get('dendrogram_clicked_clusters', [])
    dendrogram_highlight_points = current_state.get('dendrogram_highlight_points', [])
    last_interaction = current_state.get('last_interaction_type', None)
    
    # どの入力が変化したかを判定（シンプル版）
    dr_changed = selectedData is not None and len(selectedData.get('points', [])) > 0
    heatmap_changed = heatmapClickData is not None
    dendrogram_changed = dendrogramClickData is not None
    
    # DR選択による選択クラスタIDとポイントID取得（既存のheatmap状態を保持）
    if dr_changed and selectedData.get('points'):
        print(f"DR selection detected: {len(selectedData['points'])} points")
        try:
            # 選択されたポイントIDを取得
            selected_point_indices = [int(point['pointIndex']) for point in selectedData['points'] if point.get("customdata") and point["customdata"][0] != -1]
            
            # クラスタIDを取得
            new_selected_clusters = get_clusters_from_points(selected_point_indices, point_cluster_map)
            
            # DR選択が変わった場合のみ更新
            if new_selected_clusters != dr_selected_clusters or selected_point_indices != dr_selected_points:
                dr_selected_clusters = new_selected_clusters
                dr_selected_points = selected_point_indices
                last_interaction = 'dr_selection'
                print(f"Extracted cluster IDs from DR selection: {dr_selected_clusters[:10]}...")
                print(f"Selected point indices: {len(dr_selected_points)} points")
                print(f"Preserving existing heatmap state: {len(heatmap_clicked_cluster_ids)} clicked clusters")
            else:
                print("DR selection unchanged, skipping update")
        except Exception as e:
            print(f"Error processing DR selection: {e}")
    
    # heatmapクリックによる選択クラスタID取得（既存のDR状態を保持）
    if heatmap_changed:
        print(f"Heatmap click detected")
        try:
            clicked_x = heatmapClickData['points'][0]['x']
            clicked_y = heatmapClickData['points'][0]['y']
            if isinstance(clicked_x, (int, str)) and isinstance(clicked_y, (int, str)):
                new_heatmap_clusters = [int(clicked_x), int(clicked_y)]
                # heatmapクリックが変わった場合のみ更新
                if new_heatmap_clusters != heatmap_clicked_cluster_ids:
                    heatmap_clicked_cluster_ids = new_heatmap_clusters
                    last_interaction = 'heatmap_click'
                    print(f"Heatmap clicked cluster IDs: {heatmap_clicked_cluster_ids}")
                    print(f"Preserving existing DR selection: {len(dr_selected_clusters)} selected clusters, {len(dr_selected_points)} points")
                else:
                    print("Heatmap click unchanged, skipping update")
        except Exception as e:
            print(f"Error processing heatmap click: {e}")
    
    # デンドログラムクリックによる選択クラスタID取得（既存のDR・heatmap状態を保持）
    if dendrogram_changed:
        print(f"Dendrogram click detected")
        try:
            # クリックデータの詳細情報をデバッグ出力
            click_point = dendrogramClickData['points'][0]
            print(f"Full dendrogram click data: {click_point}")
            
            clicked_x = click_point['x']
            clicked_y = click_point['y']
            
            # curveNumber（トレース番号）から線分インデックスを取得
            curve_number = click_point.get('curveNumber', None)
            point_number = click_point.get('pointNumber', None)
            
            print(f"Dendrogram clicked - x={clicked_x}, y={clicked_y}, curve={curve_number}, point={point_number}")
            
            # curveNumberがセグメント（線分）のインデックスに対応
            if curve_number is not None:
                segment_index = curve_number
                cluster_index = segment_index // 3  # 3つのセグメントごとに1つのクラスタ
                
                print(f"Segment index: {segment_index}, Cluster index: {cluster_index}")
                
                # linkage_matrixから対応するクラスタ情報を取得
                if cluster_index < len(linkage_matrix):
                    linkage_row = linkage_matrix[cluster_index]
                    parent_cluster_id = int(linkage_row[2])  # parent cluster ID
                    
                    # new_old_id_mapで元のクラスタIDに変換
                    if parent_cluster_id in new_old_id_map:
                        original_cluster_id = new_old_id_map[parent_cluster_id]
                        new_dendrogram_clusters = [original_cluster_id]
                        
                        print(f"Linkage row {cluster_index}: {linkage_row}")
                        print(f"Parent cluster {parent_cluster_id} -> Original cluster {original_cluster_id}")
                        
                        if new_dendrogram_clusters != dendrogram_clicked_clusters:
                            dendrogram_clicked_clusters = new_dendrogram_clusters
                            last_interaction = 'dendrogram_click'
                            print(f"Dendrogram clicked cluster IDs: {dendrogram_clicked_clusters}")
                            print(f"Preserving existing DR and Heatmap selections")
                        else:
                            print("Dendrogram click unchanged, skipping update")
                    else:
                        print(f"Parent cluster {parent_cluster_id} not found in new_old_id_map")
                else:
                    print(f"Cluster index {cluster_index} out of range (max: {len(linkage_matrix)-1})")
            else:
                print("No curveNumber found in click data")
                
        except Exception as e:
            print(f"Error processing dendrogram click: {e}")
            import traceback
            traceback.print_exc()
    
    # heatmapクリック時のハイライト対象ポイントIDを計算（最適化版）
    if heatmap_clicked_cluster_ids:
        # パフォーマンス最適化: setで高速検索
        cluster_set = set(heatmap_clicked_cluster_ids)
        heatmap_highlight_points = [pid for pid, cid in point_cluster_map.items() if cid in cluster_set]
        print(f"Heatmap highlight points: {len(heatmap_highlight_points)} points from clusters {heatmap_clicked_cluster_ids}")
    else:
        heatmap_highlight_points = []
    
    # デンドログラムクリック時のハイライト対象ポイントIDを計算
    if dendrogram_clicked_clusters:
        cluster_set = set(dendrogram_clicked_clusters)
        dendrogram_highlight_points = [pid for pid, cid in point_cluster_map.items() if cid in cluster_set]
        print(f"Dendrogram highlight points: {len(dendrogram_highlight_points)} points from clusters {dendrogram_clicked_clusters}")
    else:
        dendrogram_highlight_points = []
    
    # 結果を構築
    result = {
        'dr_selected_clusters': dr_selected_clusters,
        'dr_selected_points': dr_selected_points,
        'heatmap_clicked_clusters': heatmap_clicked_cluster_ids,
        'heatmap_highlight_points': heatmap_highlight_points,
        'dendrogram_clicked_clusters': dendrogram_clicked_clusters,
        'dendrogram_highlight_points': dendrogram_highlight_points,
        'last_interaction_type': last_interaction
    }
    
    # 変更がない場合は現在の状態を返す（不要な更新を防止）
    if (current_state and 
        current_state.get('dr_selected_clusters') == dr_selected_clusters and
        current_state.get('dr_selected_points') == dr_selected_points and
        current_state.get('heatmap_clicked_clusters') == heatmap_clicked_cluster_ids and
        current_state.get('dendrogram_clicked_clusters') == dendrogram_clicked_clusters and
        current_state.get('last_interaction_type') == last_interaction):
        print("No changes detected, returning current state")
        return current_state
    
    print(f"Store updated: DR({len(dr_selected_clusters)} clusters, {len(dr_selected_points)} points), Heatmap({len(heatmap_clicked_cluster_ids)} clusters, {len(heatmap_highlight_points)} points), Dendrogram({len(dendrogram_clicked_clusters)} clusters, {len(dendrogram_highlight_points)} points), last_interaction: {last_interaction}")
    
    return result

# デンドログラム更新コールバック（ストア -> グラフ）
@app.callback(
    Output('dendrogram-plot', 'figure'),
    [
        Input('dendro-width-option-toggle', 'value'),
        Input('dendro-label-annotation-toggle', 'value'),
        Input('selected-ids-store', 'data')
    ]
)
def update_dendrogram_plot(width_options, label_options, stored_data):
    """デンドログラムプロットを更新"""
    prop_width = 'prop_width' in width_options
    print(f"Updating dendrogram plot with prop_width={prop_width}")
    
    # 簡単なデンドログラム風の可視化
    fig = go.Figure()
    
    # デンドログラム線を描画
    if prop_width:
        # サイズ考慮版を使用
        segments = get_dendrogram_segments_with_size(linkage_matrix[:, [0, 1, 3, 4]])
    else:
        # 通常版を使用
        segments = get_dendrogram_segments(linkage_matrix[:, [0, 1, 3, 4]])
    
    # ストアからデータを取得
    dr_selected_cluster_ids = stored_data.get('dr_selected_clusters', [])
    heatmap_clicked_cluster_ids = stored_data.get('heatmap_clicked_clusters', [])
    
    print(f"From store - DR selected clusters: {dr_selected_cluster_ids[:10] if dr_selected_cluster_ids else []}")
    print(f"From store - heatmap clicked clusters: {heatmap_clicked_cluster_ids}")
    
    # 新IDにマッピング
    selected_new_cluster_ids = [old_new_id_map[k] for k in dr_selected_cluster_ids if k in old_new_id_map]
    heatmap_clicked_new_cluster_ids = [old_new_id_map[k] for k in heatmap_clicked_cluster_ids if k in old_new_id_map]

    # デンドログラム上のクラスタIDの選択状態を計算
    # DR選択による選択状態
    is_selected = [True if (row[0] in selected_new_cluster_ids) or (row[1] in selected_new_cluster_ids) else False for row in linkage_matrix]
    
    # heatmapクリックによるハイライト状態（原色で強調）
    is_heatmap_clicked = [True if (row[0] in heatmap_clicked_new_cluster_ids) or (row[1] in heatmap_clicked_new_cluster_ids) else False for row in linkage_matrix]
    
    # テキスト生成
    if dr_selected_cluster_ids:
        text = f"選択されたデータポイントのクラスターid: {', '.join(map(str, selected_new_cluster_ids)) + '.'.join(map(str, dr_selected_cluster_ids))}"
    else:
        text = "選択されたデータポイントはありません。"
    
    print(f"len segment, is_selected: {len(segments)}, {len(is_selected)}")

    # サイズ表示オプションに基づいて描画関数を選択
    if prop_width:
        # サイズ情報を取得（linkage matrixの4列目がクラスタサイズ）
        cluster_sizes = [row[4] for row in linkage_matrix] if len(linkage_matrix) > 0 else []
        print(f"Using size-aware plotting with {len(cluster_sizes)} cluster sizes")
        fig = plot_dendrogram_plotly_with_size(
            segments, 
            is_selecteds=is_selected,
            is_heatmap_clicked=is_heatmap_clicked,
            cluster_sizes=cluster_sizes,
            **extra_info_kwargs
        )
    else:
        print("Using standard plotting")
        fig = plot_dendrogram_plotly(segments, is_selecteds=is_selected, is_heatmap_clicked=is_heatmap_clicked, **extra_info_kwargs)
    
    # ラベルアノテーションを追加（show_labelsがTrueの場合）
    show_labels = 'show_labels' in label_options
    if show_labels:
        add_dendrogram_label_annotations(fig, linkage_matrix, segments)
        print("Dendrogram labels added")
    
    # figureの基づくコンパクト化設定
    fig.update_layout(
        xaxis_title='Data Points',
        yaxis_title='Distance',
        title='Cluster Dendrogram' + (' (Size-Aware)' if prop_width else '') + (' + Labels' if show_labels else ''),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),  # マージン最小化
        plot_bgcolor='#f8f9fa',  # コンテナの色に合わせて背景色統一
        paper_bgcolor='#f8f9fa'  # 外枠背景色統一
    )

    return fig

# クラスタ情報更新コールバック（ストア -> クラスタ情報グラフ）
@app.callback(
    Output('cluster-info', 'figure'),
    [
        Input('selected-ids-store', 'data'),
        Input('heatmap-reverse-colorscale', 'value'),
        Input('heatmap-cluster-reorder', 'value')
    ]
)
def update_cluster_info(stored_data, reverse_colorscale_options, cluster_reorder_options):
    """クラスタ情報（類似度ヒートマップ）を更新"""
    selected_cluster_ids = stored_data.get('dr_selected_clusters', [])
    reverse_colorscale = 'reverse' in reverse_colorscale_options
    cluster_reorder = 'reorder' in cluster_reorder_options
    
    print(f"update_cluster_info called with {len(selected_cluster_ids)} selected clusters")
    print(f"Options - Reverse colorscale: {reverse_colorscale}, Cluster reorder: {cluster_reorder}")
    
    # パフォーマンス制限: 大量クラスタ選択時の処理スキップ（設定で制御可能）
    if ENABLE_HEATMAP_CLUSTER_LIMIT and len(selected_cluster_ids) > MAX_HEATMAP_CLUSTERS:
        print(f"Too many clusters selected ({len(selected_cluster_ids)}), skipping heatmap generation for performance (limit: {MAX_HEATMAP_CLUSTERS})")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Too many clusters selected ({len(selected_cluster_ids)})<br>Select fewer clusters (<= {MAX_HEATMAP_CLUSTERS}) for heatmap<br>Or disable ENABLE_HEATMAP_CLUSTER_LIMIT in code",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color="orange")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa'
        )
        return fig
    
    # 選択されたクラスタがない場合、空のプロットまたはメッセージを表示
    if len(selected_cluster_ids) == 0:
        print("No clusters selected, showing empty plot")
        fig = go.Figure()
        fig.add_annotation(
            text="Select clusters in DR view to see similarity heatmap",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa',
            margin=dict(l=10, r=10, t=30, b=10)
        )
        return fig
    
    # クラスタが1つだけの場合も特別処理
    if len(selected_cluster_ids) == 1:
        print("Single cluster selected, showing single cell heatmap")
        colorscale = 'Viridis_r' if reverse_colorscale else 'Viridis'
        title_suffix = ""
        if reverse_colorscale:
            title_suffix += " [Reversed Scale]"
        
        fig = px.imshow([[1.0]], 
                       x=[selected_cluster_ids[0]], 
                       y=[selected_cluster_ids[0]],
                       labels=dict(x="Cluster ID", y="Cluster ID", color="Mahalanobis Distance"),
                       color_continuous_scale=colorscale,
                       title=f"Cluster {selected_cluster_ids[0]} (Single Selection)" + title_suffix)
        fig.update_xaxes(type='category')
        fig.update_yaxes(type='category')
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa'
        )
        return fig
    
    # クラスタ(selected cluster ids)間の類似度を表示(similality_matrix_bc などを使用)
    n_clusters = len(selected_cluster_ids)
    selected_cluster_similarity_matrix1 = np.zeros((n_clusters, n_clusters))
    
    try:
        for i, cid1 in enumerate(selected_cluster_ids):
            for j, cid2 in enumerate(selected_cluster_ids):
                if i == j:
                    selected_cluster_similarity_matrix1[i, j] = 0.0
                elif i < j:
                    key = (cid1, cid2) if (cid1, cid2) in similarity_m else (cid2, cid1)
                    similarity_value = similarity_m.get(key, 0)
                    selected_cluster_similarity_matrix1[i, j] = similarity_value
                    selected_cluster_similarity_matrix1[j, i] = similarity_value
        
        print(f"Selected cluster similarity matrix shape: {selected_cluster_similarity_matrix1.shape}")
        print(f"Matrix values range: {selected_cluster_similarity_matrix1.min():.4f} to {selected_cluster_similarity_matrix1.max():.4f}")

        # クラスタリング並べ替え処理
        display_matrix = selected_cluster_similarity_matrix1.copy()
        display_cluster_ids = selected_cluster_ids.copy()
        
        if cluster_reorder and len(selected_cluster_ids) > 2 and SCIPY_AVAILABLE:
            try:
                
                # 類似度を距離に変換（大きい類似度 = 小さい距離）
                # 対角線を除いて距離行列を作成
                max_sim = np.max(selected_cluster_similarity_matrix1)
                distance_matrix = max_sim - selected_cluster_similarity_matrix1
                np.fill_diagonal(distance_matrix, 0)
                
                # 上三角行列を1次元配列に変換（scipy.cluster.hierarchy用）
                condensed_distances = squareform(distance_matrix)
                
                # 階層クラスタリング実行
                linkage_result = linkage(condensed_distances, method='average')
                dendro_result = dendrogram(linkage_result, no_plot=True)
                
                # デンドログラムの順序を取得
                reorder_indices = dendro_result['leaves']
                
                # 行列と軸ラベルを並べ替え
                display_matrix = selected_cluster_similarity_matrix1[np.ix_(reorder_indices, reorder_indices)]
                display_cluster_ids = [selected_cluster_ids[i] for i in reorder_indices]
                
                print(f"Reordered clusters: {display_cluster_ids}")
                
            except Exception as e:
                print(f"Error in cluster reordering: {e}")
        elif cluster_reorder and not SCIPY_AVAILABLE:
            print("Cluster reordering requested but scipy not available")
        
        # カラースケール設定
        colorscale = 'Viridis_r' if reverse_colorscale else 'Viridis'
        
        fig_similarity1 = px.imshow(display_matrix,
                        x=display_cluster_ids,
                        y=display_cluster_ids,
                        labels=dict(x="Cluster ID", y="Cluster ID", color="Mahalanobis Distance"),
                        color_continuous_scale=colorscale,
                        title=f"Cluster Similarity ({len(selected_cluster_ids)} clusters)" + 
                               (" [Reordered]" if cluster_reorder else "") +
                               (" [Reversed Scale]" if reverse_colorscale else ""))
        
        fig_similarity1.update_xaxes(type='category')
        fig_similarity1.update_yaxes(type='category')
        fig_similarity1.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa'
        )
        
        print(f"Successfully updated cluster info heatmap for {len(selected_cluster_ids)} clusters")
        
        return fig_similarity1
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        # エラー時はエラーメッセージを表示するプロットを作成
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating heatmap: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa'
        )
        return fig

# cluster-details-content
@app.callback(
    Output('cluster-details-content', 'children'),
    [Input('selected-ids-store', 'data')]
)
def update_cluster_details_content(stored_data):
    """クラスタ詳細コンテンツを更新"""
    
    # 選択されたクラスタ情報を取得
    dr_selected_clusters = stored_data.get('dr_selected_clusters', [])
    dr_selected_points = stored_data.get('dr_selected_points', [])
    heatmap_clicked_clusters = stored_data.get('heatmap_clicked_clusters', [])
    dendrogram_clicked_clusters = stored_data.get('dendrogram_clicked_clusters', [])
    last_interaction = stored_data.get('last_interaction_type', None)
    
    selected_cluster_info = []
    
    # DR選択がある場合のクラスタ分析
    if dr_selected_clusters and dr_selected_points:
        print(f"Analyzing DR selection: {len(dr_selected_clusters)} clusters, {len(dr_selected_points)} points")
        
        # 各クラスタの含有率を計算
        for cluster_id in dr_selected_clusters:
            cluster_points = [pid for pid, cid in point_cluster_map.items() if cid == cluster_id]
            total_cluster_points = len(cluster_points)
            selected_points_in_cluster = len([pid for pid in dr_selected_points if pid in cluster_points])
            selection_ratio = selected_points_in_cluster / total_cluster_points if total_cluster_points > 0 else 0
            
            # 閾値フィルタリング
            if selection_ratio >= DR_SELECTION_CLUSTER_RATIO_THRESHOLD:
                stability_score = stability_dict.get(cluster_id, 0.0)
                
                selected_cluster_info.append({
                    'cluster_id': cluster_id,
                    'point_count': total_cluster_points,
                    'selected_points': selected_points_in_cluster,
                    'selection_ratio': selection_ratio,
                    'stability': stability_score,
                    'sample_points': cluster_points[:5],
                    'representative_label': cluster_representative_labels.get(cluster_id, f"Cluster_{cluster_id}"),
                    'cluster_words': cluster_word_lists.get(cluster_id, [])
                })
            else:
                print(f"Cluster {cluster_id} filtered out: ratio {selection_ratio:.3f} < threshold {DR_SELECTION_CLUSTER_RATIO_THRESHOLD}")
        
        # 選択率とStabilityでソート
        selected_cluster_info.sort(key=lambda x: (x['selection_ratio'], x['stability']), reverse=True)
    
    # その他の選択タイプ（Heatmap、Dendrogram）の処理
    elif heatmap_clicked_clusters:
        for cluster_id in heatmap_clicked_clusters:
            cluster_points = [pid for pid, cid in point_cluster_map.items() if cid == cluster_id]
            stability_score = stability_dict.get(cluster_id, 0.0)
            
            selected_cluster_info.append({
                'cluster_id': cluster_id,
                'point_count': len(cluster_points),
                'selected_points': len(cluster_points),
                'selection_ratio': 1.0,
                'stability': stability_score,
                'sample_points': cluster_points[:5],
                'representative_label': cluster_representative_labels.get(cluster_id, f"Cluster_{cluster_id}"),
                'cluster_words': cluster_word_lists.get(cluster_id, [])
            })
    
    elif dendrogram_clicked_clusters:
        for cluster_id in dendrogram_clicked_clusters:
            cluster_points = [pid for pid, cid in point_cluster_map.items() if cid == cluster_id]
            stability_score = stability_dict.get(cluster_id, 0.0)
            
            selected_cluster_info.append({
                'cluster_id': cluster_id,
                'point_count': len(cluster_points),
                'selected_points': len(cluster_points),
                'selection_ratio': 1.0,
                'stability': stability_score,
                'sample_points': cluster_points[:5],
                'representative_label': cluster_representative_labels.get(cluster_id, f"Cluster_{cluster_id}"),
                'cluster_words': cluster_word_lists.get(cluster_id, [])
            })
        
        # Stabilityでソート
        selected_cluster_info.sort(key=lambda x: x['stability'], reverse=True)
    
    # クラスタ詳細を表示
    if selected_cluster_info:
        # 選択されたクラスタの総ポイント数を計算
        n_selected_points = sum(info['point_count'] for info in selected_cluster_info)
        n_selected_clusters = len(selected_cluster_info)
        
        # クラスタ一覧テーブル
        cluster_table_data = []
        for info in selected_cluster_info:
            # クラスタ内の単語を最初のn個まで表示
            cluster_words = info.get('cluster_words', [])
            words_display = ', '.join(cluster_words[:MAX_CLUSTER_WORDS_DISPLAY])
            if len(cluster_words) > MAX_CLUSTER_WORDS_DISPLAY:
                words_display += f" (+{len(cluster_words) - MAX_CLUSTER_WORDS_DISPLAY} more)"
            
            cluster_table_data.append({
                'Cluster ID': info['cluster_id'],
                'Label': info.get('representative_label', f"Cluster_{info['cluster_id']}"),
                'Selected/Total': f"{info['selected_points']}/{info['point_count']}",
                'Ratio': f"{info['selection_ratio']:.2%}",
                'Stability': f"{info['stability']:.4f}",
                'Words': words_display if words_display else 'No words available'
            })
        
        return html.Div([
            # 統計サマリー
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{n_selected_points}", className="text-primary mb-0"),
                            html.P("Selected Points", className="text-muted small mb-0")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{n_selected_clusters}", className="text-success mb-0"),
                            html.P("Unique Clusters", className="text-muted small mb-0")
                        ])
                    ])
                ], width=6)
            ], className="mb-3"),
            
            # クラスタ一覧テーブル
            html.H6("Cluster Information", className="fw-bold mb-2"),
            dash_table.DataTable(
                data=cluster_table_data,
                columns=[
                    {'name': 'Cluster ID', 'id': 'Cluster ID'},
                    {'name': 'Label', 'id': 'Label'},
                    {'name': 'Selected/Total', 'id': 'Selected/Total'},
                    {'name': 'Ratio', 'id': 'Ratio'},
                    {'name': 'Stability', 'id': 'Stability'},
                    {'name': 'Words (n words)', 'id': 'Words'}
                ],
                style_table={'height': '250px', 'overflowY': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'fontSize': '11px',
                    'padding': '5px'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
        ])
    else:
        return html.Div([
            html.P("Select clusters in DR view, heatmap, or dendrogram to see details.", className="text-muted text-center mt-5")
        ])

# detail-panel-content (simplified)
@app.callback(
    Output('detail-panel-content', 'children'),
    [Input('detail-info-tabs', 'active_tab'),
     Input('dr-visualization-plot', 'clickData'),
     Input('selected-ids-store', 'data')]
)
def update_detail_panel(active_tab, click_data, stored_data):
    """詳細パネルの内容を更新（簡略化版 - Point DetailsとSystem Logのみ）"""
    
    # 選択されたクラスタ情報を取得（ログ用）
    dr_selected_clusters = stored_data.get('dr_selected_clusters', [])
    dr_selected_points = stored_data.get('dr_selected_points', [])
    heatmap_clicked_clusters = stored_data.get('heatmap_clicked_clusters', [])
    dendrogram_clicked_clusters = stored_data.get('dendrogram_clicked_clusters', [])
    last_interaction = stored_data.get('last_interaction_type', None)
    
    if active_tab == 'tab-point-details':
        if click_data:
            point_idx = click_data['points'][0]['pointIndex']
            point_x = click_data['points'][0]['x']
            point_y = click_data['points'][0]['y']
            point_label = df.iloc[point_idx]['label'] if point_idx < len(df) else 'Unknown'
            point_cluster = point_cluster_map.get(point_idx, 'Unknown')
            
            return html.Div([
                html.H6(f"Selected Point Details", className="fw-bold mb-3"),
                dbc.ListGroup([
                    dbc.ListGroupItem(f"Point Index: {point_idx}"),
                    dbc.ListGroupItem(f"X: {point_x:.4f}"),
                    dbc.ListGroupItem(f"Y: {point_y:.4f}"),
                    dbc.ListGroupItem(f"Label: {point_label}"),
                    dbc.ListGroupItem(f"Cluster ID: {point_cluster}"),
                ], flush=True)
            ])
        else:
            return html.Div([
                html.P("Click a point in the DR view to see details.", className="text-muted")
            ])
    
    elif active_tab == 'tab-selection-stats':
        # 簡略化されたSelection Statistics - 基本情報のみ
        return html.Div([
            html.H6("Selection Statistics", className="fw-bold mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{len(dr_selected_points)}", className="text-primary mb-0"),
                            html.P("DR Selected Points", className="text-muted small mb-0")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{len(dr_selected_clusters)}", className="text-success mb-0"),
                            html.P("DR Selected Clusters", className="text-muted small mb-0")
                        ])
                    ])
                ], width=6)
            ], className="mb-3"),
            html.P("Detailed cluster information moved to Cluster Details panel below.", className="text-muted text-center")
        ])
    
    elif active_tab == 'tab-cluster-size':
        # クラスタサイズ分布を表示
        return html.Div([
            html.H6("Cluster Size Distribution", className="fw-bold mb-3"),
            dcc.Graph(
                id='cluster-size-dist-detail',
                figure=fig_cluster_hist,
                style={'height': '300px'}
            )
        ])
    
    elif active_tab == 'tab-system-log':
        log_entries = [
            "2024-11-16 10:30:15 - Application started",
            "2024-11-16 10:30:20 - Dataset loaded: HDBSCAN embedding data",
            f"2024-11-16 10:30:25 - Linkage matrix prepared: {linkage_matrix.shape[0]} merges",
            f"2024-11-16 10:30:30 - Point-to-cluster mapping: {len(point_cluster_map)} points"
        ]
        
        # 選択情報をログに追加
        if dr_selected_clusters or heatmap_clicked_clusters or dendrogram_clicked_clusters:
            log_entries.append(f"2024-11-16 10:30:35 - DR clusters: {len(dr_selected_clusters)}, Heatmap: {len(heatmap_clicked_clusters)}, Dendrogram: {len(dendrogram_clicked_clusters)}")
            log_entries.append(f"2024-11-16 10:30:36 - Selection type: {last_interaction or 'None'}")
            log_entries.append(f"2024-11-16 10:30:37 - Ratio threshold: {DR_SELECTION_CLUSTER_RATIO_THRESHOLD}")
        
        return html.Div([
            html.H6("System Log", className="fw-bold mb-3"),
            html.Div([
                html.P(entry, className="mb-1") for entry in log_entries
            ], style={
                'fontSize': '11px', 
                'fontFamily': 'monospace',
                'backgroundColor': '#f8f9fa',
                'padding': '10px',
                'borderRadius': '4px',
                'maxHeight': '300px',
                'overflowY': 'auto'
            })
        ])
    
    return html.Div()

# detail-info-tabs
@app.callback(
    Output('detail-info-tabs', 'style'),  # ダミーアウトプット
    Input('execute-button', 'n_clicks')
)
def log_execute_button(n_clicks):
    """実行ボタンが押された際のログ出力（モックアップ）"""
    if n_clicks > 0:
        print(f"実行ボタンが押されました (クリック数: {n_clicks})")
    return {}

if __name__ == '__main__':
    app.run(debug=True, port=8050)