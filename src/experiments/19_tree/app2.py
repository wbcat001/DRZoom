import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
import dash_bootstrap_components as dbc
import os
import pickle
np.set_printoptions(suppress=True, precision=4)

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
        # color = 'blue' if colors is None else colors[i]
        color = "blue"
        info = 'N/A' if scores is None or index >= len(scores) else f"{scores[index]:.2f}"
        opacity = 1.0
        if is_selecteds is not None:
            
            color = "orange" if is_selecteds[index] else "skyblue"

        hover_lines = []
        for key, value_list in additional_data:
            value = value_list[index] if index < len(value_list) else "N/A"
            if isinstance(value, float):
                hover_lines.append(f"{key}: {value:.4f}")
            else:
                hover_lines.append(f"{key}: {value}")
        
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
        
        # 選択状態に応じた色の変更（plot_dendrogram_plotlyと同じロジック）
        if is_selecteds is not None:
            color = "orange" if is_selecteds[index] else "skyblue"

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
data_file_path = "src/experiments/18_rapids/result/20251112_044404/embedding.npz"# os.path.join(_base_dir, 'data', 'dr_dummy_data.csv')

label_file_path = "src/experiments/22_word_label/processed_data/w2v_words_100000.txt"# os.path.join(_base_dir, 'data', 'dr_dummy_data.csv')
hdbscan_condensed_tree_file_path = "src/experiments/18_rapids/result/20251112_044404/condensed_tree_object.pkl" # os.path.join(_base_dir, 'data', 'hdbscan_condensed_tree.csv')

with open(label_file_path, 'r', encoding='utf-8') as f:
    true_labels = [line.strip() for line in f.readlines()]
print(f"Loaded true labels from {label_file_path}, length: {len(true_labels)}, sample: {true_labels[:5]}")

similarity_file_path = "src/experiments/19_tree/cluster_similarities.pkl"
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
    "Stability": [stability_dict[new_old_id_map[int(row[2])]] if new_old_id_map[int(row[2])] in stability_dict else 0.0 for row in linkage_matrix]  
}
print(f'extra_info head sample: {extra_info_kwargs["Cluster Size"][:5]}')

df = pd.DataFrame({
    "x": embedding[:,0],
    "y": embedding[:,1],
    "label": true_labels,
    "cluster_label": labels,
    "cluster_id": [point_cluster_map[pid] for pid in point_cluster_map.keys()]
})




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
                        dbc.RadioItems(
                            id='dr-interaction-mode-toggle',
                            options=[
                                {'label': 'Brush Selection', 'value': 'brush'},
                                {'label': 'Zoom/Pan', 'value': 'zoom'}
                            ],
                            value='zoom',
                            inline=True,
                            className="mb-2 text-center"
                        ),
                        
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
                            ], width=6),
                            dbc.Col([
                                dbc.Checklist(
                                    id='dendro-width-option-toggle',
                                    options=[{'label': 'Proportional Width', 'value': 'prop_width'}],
                                    value=[],
                                )
                            ], width=6)
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
                        
                        # クラスタ情報グラフ
                        dcc.Graph(id='cluster-info', className="flex-grow-1")
                    ], className="d-flex flex-column p-3 h-100")
                ], style={'height': '400px'}, className="h-100"),
            ], width=12, className="p-2")
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

# グラフ描画（go.Scatter版でcustomdata問題を解決）
@app.callback(
    Output('dr-visualization-plot', 'figure'),
    [Input('execute-button', 'n_clicks'),
     Input('dr-method-selector', 'value'),
     Input('dr-interaction-mode-toggle', 'value')]
)
def update_dr_plot(n_clicks, method, interaction_mode):
    """DR可視化プロットを更新"""
    fig = px.scatter(
        df,
        x='x',
        y='y',
        hover_data=['label'],
        custom_data=['cluster_label']
    )
    fig.update_traces(
        marker=dict(size=2, opacity=0.5),
        # selector=dict(mode='markers')
    )
    
    # フィードバックに基づくコンパクト化設定
    fig.update_layout(
        # height=500,  <-- DBCで高さをCSSで管理するため削除
        showlegend=False,  # 凡例無効化
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        margin=dict(l=10, r=10, t=10, b=10),  # マージン最小化
        plot_bgcolor='#f8f9fa',  # コンテナの色に合わせて背景色統一
        paper_bgcolor='#f8f9fa'  # 外枠背景色統一
    )
    
    # インタラクションモードに応じた設定
    if interaction_mode == 'brush':
        drag_mode = 'lasso' # "brush"
    else:
        drag_mode = 'zoom'
    
    fig.update_layout(dragmode=drag_mode)
    
    return fig



@app.callback(
    Output('dendrogram-plot', 'figure'),
    Output('cluster-info', 'figure'),
    [
        Input('dendro-width-option-toggle', 'value'),
        Input("dr-visualization-plot", 'selectedData')
    ]
)
def update_dendrogram_plot(width_options, selectedData):
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
    selected_cluster_ids = []
    selected_new_cluster_ids = []
    is_selected = []

    # 選択データがある場合
    if selectedData is not None:
      
        # 選択されたポイントのクラスタidを取得
        selected_cluster_ids = get_clusters_from_points(
            [int(point['pointIndex']) for index, point in enumerate(selectedData['points']) if point["customdata"][0] is not -1],
            point_cluster_map
        )
        print(f"length of selected cluster ids: {len(selected_cluster_ids)}")
        print(f"selected cluster ids: {selected_cluster_ids[:10]}")

        selected_new_cluster_ids = [old_new_id_map[k] for k in selected_cluster_ids if k in old_new_id_map]
        print(f"length of selected new cluster ids: {len(selected_new_cluster_ids)}")
        print(f"selected new cluster ids: {selected_new_cluster_ids[:10]}")

        # デンドログラム上のクラスタID
        # childに含まれている枝を選択状態にする
        is_selected = [True if (row[0] in selected_new_cluster_ids) or (row[1] in selected_new_cluster_ids) else False for row in linkage_matrix]
        
        
        text = f"選択されたデータポイントのクラスターid: {', '.join(map(str, selected_new_cluster_ids)) + '.'.join(map(str, selected_cluster_ids))}"
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
            cluster_sizes=cluster_sizes,
            **extra_info_kwargs
        )
    else:
        print("Using standard plotting")
        fig = plot_dendrogram_plotly(segments, is_selecteds=is_selected, **extra_info_kwargs)
    
    # figureの基づくコンパクト化設定
    fig.update_layout(
        xaxis_title='Data Points',
        yaxis_title='Distance',
        title='Cluster Dendrogram' + (' (Size-Aware)' if prop_width else ''),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),  # マージン最小化
        plot_bgcolor='#f8f9fa',  # コンテナの色に合わせて背景色統一
        paper_bgcolor='#f8f9fa'  # 外枠背景色統一
    )

    # クラスタ(selected cluster ids)間の類似度を表示(similality_matrix_bc などを使用)
    selected_cluster_similarity_matrix1 = np.zeros((len(selected_cluster_ids), len(selected_cluster_ids)))
    selected_cluster_similarity_matrix2 = np.zeros((len(selected_cluster_ids), len(selected_cluster_ids)))
    selected_cluster_similarity_matrix3 = np.zeros((len(selected_cluster_ids), len(selected_cluster_ids)))
    for i, cid1 in enumerate(selected_cluster_ids):
        for j, cid2 in enumerate(selected_cluster_ids):
            if i == j:
                selected_cluster_similarity_matrix1[i, j] = 0.0
                selected_cluster_similarity_matrix2[i, j] = 0.0
                selected_cluster_similarity_matrix3[i, j] = 0.0
            elif i < j:
                key = (cid1, cid2) if (cid1, cid2) in similarity_m else (cid2, cid1)
                selected_cluster_similarity_matrix1[i, j] = similarity_m.get(key, 0)
                selected_cluster_similarity_matrix1[j, i] = selected_cluster_similarity_matrix1[i, j]
                selected_cluster_similarity_matrix2[i, j] = similarity_bc.get(key, 0)
                selected_cluster_similarity_matrix2[j, i] = selected_cluster_similarity_matrix2[i, j]
                selected_cluster_similarity_matrix3[i, j] = similarity_kl.get(key, 0)
                selected_cluster_similarity_matrix3[j, i] = selected_cluster_similarity_matrix3[i, j]
    print(f"Selected cluster similarity matrix shape: {selected_cluster_similarity_matrix1.shape}")
    print(f"shape of selected cluster ids: {len(selected_cluster_ids)}")

    fig_similarity1 = px.imshow(selected_cluster_similarity_matrix1,
                    x=selected_cluster_ids,
                    y=selected_cluster_ids,
                    labels=dict(x="Cluster ID", y="Cluster ID", color="Mahalanobis Distance"),
                    color_continuous_scale='Viridis',
                    ).update_xaxes(type='category').update_yaxes(type='category')
    fig_similarity2 = px.imshow(selected_cluster_similarity_matrix2,
                    x=selected_cluster_ids,
                    y=selected_cluster_ids,
                    labels=dict(x="Cluster ID", y="Cluster ID", color="KL Divergence"),
                    color_continuous_scale='Viridis',
                    ).update_xaxes(type='category').update_yaxes(type='category')
    fig_siilarity3 = px.imshow(selected_cluster_similarity_matrix3,
                    x=selected_cluster_ids,
                    y=selected_cluster_ids,
                    labels=dict(x="Cluster ID", y="Cluster ID", color="Bhattacharyya Coefficient"),
                    color_continuous_scale='Viridis',
                    ).update_xaxes(type='category').update_yaxes(type='category')
    
    return fig, fig_similarity1

# detail-panel-content
@app.callback(
    Output('detail-panel-content', 'children'),
    [Input('detail-info-tabs', 'active_tab'),
     Input('dr-visualization-plot', 'clickData'),
     Input('dr-visualization-plot', 'selectedData')]
)
def update_detail_panel(active_tab, click_data, selected_data):
    """詳細パネルの内容を更新"""
    
    # 選択されたクラスタ情報を取得
    selected_cluster_ids = []
    selected_cluster_info = []
    
    if selected_data is not None and selected_data.get('points'):
        # 選択されたポイントからクラスタIDを取得
        selected_point_indices = [
            int(point['pointIndex']) 
            for point in selected_data['points'] 
            if point.get("customdata") and point["customdata"][0] != -1
        ]
        
        selected_cluster_ids = get_clusters_from_points(
            selected_point_indices, 
            point_cluster_map
        )
        
        # クラスタ情報を収集
        for cluster_id in selected_cluster_ids:
            cluster_points = [pid for pid, cid in point_cluster_map.items() if cid == cluster_id]
            stability_score = stability_dict.get(cluster_id, 0.0)
            
            selected_cluster_info.append({
                'cluster_id': cluster_id,
                'point_count': len(cluster_points),
                'stability': stability_score,
                'sample_points': cluster_points[:5]  # 最初の5点のみ表示
            })
        
        # Stabilityでソート
        selected_cluster_info.sort(key=lambda x: x['stability'], reverse=True)
    
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
        if selected_data and selected_data.get('points'):
            n_selected_points = len(selected_data['points'])
            n_selected_clusters = len(selected_cluster_ids)
            
            # クラスタ一覧テーブル
            cluster_table_data = []
            for info in selected_cluster_info:
                cluster_table_data.append({
                    'Cluster ID': info['cluster_id'],
                    'Points': info['point_count'],
                    'Stability': f"{info['stability']:.4f}",
                    'Sample Points': ', '.join(map(str, info['sample_points']))
                })
            
            return html.Div([
                html.H6("Selection Statistics", className="fw-bold mb-3"),
                
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
                
                # クラスタ一覧
                html.H6("Cluster Details", className="fw-bold mb-2"),
                html.Div([
                    dash_table.DataTable(
                        data=cluster_table_data,
                        columns=[
                            {'name': 'Cluster ID', 'id': 'Cluster ID'},
                            {'name': 'Points', 'id': 'Points'},
                            {'name': 'Stability', 'id': 'Stability'},
                            {'name': 'Sample Points', 'id': 'Sample Points'}
                        ],
                        style_table={'height': '200px', 'overflowY': 'auto'},
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
                ] if cluster_table_data else [
                    html.P("No clusters selected.", className="text-muted")
                ])
            ])
        else:
            return html.Div([
                html.P("Select points in the DR view to see cluster statistics.", className="text-muted")
            ])
    
    elif active_tab == 'tab-system-log':
        log_entries = [
            "2024-11-16 10:30:15 - Application started",
            "2024-11-16 10:30:20 - Dataset loaded: HDBSCAN embedding data",
            f"2024-11-16 10:30:25 - Linkage matrix prepared: {linkage_matrix.shape[0]} merges",
            f"2024-11-16 10:30:30 - Point-to-cluster mapping: {len(point_cluster_map)} points"
        ]
        
        if selected_cluster_ids:
            log_entries.append(f"2024-11-16 10:30:35 - Selected clusters: {selected_cluster_ids[:5]}{'...' if len(selected_cluster_ids) > 5 else ''}")
        
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