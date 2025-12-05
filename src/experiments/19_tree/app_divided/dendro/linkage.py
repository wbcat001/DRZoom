import numpy as np
import pandas as pd


def recurse_leaf_dfs(cluster_tree, current_node):
    """Recursively collect leaf node ids under `current_node`.

    `cluster_tree` is expected to be a 2D array-like where columns
    contain child/child/parent relationships (as in condensed-tree arrays).
    Returns a list of leaf ids reachable from `current_node`.
    """
    child1 = cluster_tree[cluster_tree[:, 2] == current_node][:, 0]
    child2 = cluster_tree[cluster_tree[:, 2] == current_node][:, 1]
    if len(child1) == 0 and len(child2) == 0:
        return [current_node, ]
    else:
        return sum([recurse_leaf_dfs(cluster_tree, child) for child in np.concatenate((child1, child2))], [])


def get_leaves(cluster_tree):
    """Return leaf ids given a cluster_tree array.

    This helper finds the root (max parent id) and returns
    all leaves below it using `recurse_leaf_dfs`.
    """
    root = cluster_tree[:, 2].max()
    return recurse_leaf_dfs(cluster_tree, root)


def _get_leaves(condensed_tree):
    """Robust leaf extractor for different condensed_tree representations.

    Accepts a `pandas.DataFrame`, an ndarray-like `_raw_tree`, or a
    list-of-dicts representation. Returns list of leaf ids.
    """
    try:
        if isinstance(condensed_tree, pd.DataFrame):
            arr = condensed_tree.values
            return get_leaves(arr)
        else:
            arr = np.array(condensed_tree)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return get_leaves(arr)
    except Exception:
        pass
    try:
        if isinstance(condensed_tree, (list, tuple)) and len(condensed_tree) > 0 and isinstance(condensed_tree[0], dict):
            children = [int(r.get('child')) for r in condensed_tree if 'child' in r]
            parents = [int(r.get('parent')) for r in condensed_tree if 'parent' in r]
            leaves = [c for c in children if c not in parents]
            return leaves
    except Exception:
        pass
    return []


def get_linkage_matrix_from_hdbscan(condensed_tree):
    """Convert an HDBSCAN condensed_tree object to a linkage-like matrix.

    The returned `Z` is a numpy array of shape (n_merges, 5) where each row is
    [c1, c2, parent, lam_mapped, count]. `node_id_map` maps original ids found
    in the condensed tree to contiguous integer ids used in `Z`.

    This conversion follows the notebook logic and includes a protective
    mapping step that ensures leaf ids map to 0..N-1. Callers should treat
    `node_id_map` as an opaque mapping used to relate original cluster ids
    to positions in the linkage output.
    """
    linkage_matrix = []
    raw_tree = condensed_tree._raw_tree
    condensed_df = condensed_tree.to_pandas()
    cluster_tree = condensed_df[condensed_df['child_size'] > 1]
    sorted_condensed_tree = cluster_tree.sort_values(by=['lambda_val', 'parent'], ascending=True)

    for i in range(0, len(sorted_condensed_tree), 2):
        if i + 1 < len(sorted_condensed_tree):
            row_a = sorted_condensed_tree.iloc[i]
            row_b = sorted_condensed_tree.iloc[i + 1]
            if row_a['lambda_val'] != row_b['lambda_val']:
                raise ValueError(f"Lambda value mismatch at rows {i} and {i+1}: {row_a['lambda_val']} vs {row_b['lambda_val']}")
            if row_a['parent'] != row_b['parent']:
                raise ValueError(f"Parent ID mismatch at rows {i} and {i+1}: {row_a['parent']} vs {row_b['parent']}")
            child_a = row_a['child']
            child_b = row_b['child']
            lam = row_a['lambda_val']
            total_size = raw_tree[raw_tree['child'] == row_a['parent']]['child_size']
            if len(total_size) == 0:
                total_size = row_a['child_size'] + row_b['child_size']
            else:
                total_size = total_size[0]
            parent_id = row_a['parent']
            linkage_matrix.append([
                int(child_a),
                int(child_b),
                int(parent_id),
                lam,
                total_size,
            ])

    # map leaves to 0..N-1
    node_id_map = {}
    current_id = 0
    try:
        if 'child_size' in condensed_df.columns:
            leaf_vals = condensed_df[condensed_df['child_size'] == 1]['child'].unique().tolist()
        else:
            childs = condensed_df['child'].unique().tolist()
            parents = condensed_df['parent'].unique().tolist() if 'parent' in condensed_df.columns else []
            leaf_vals = [c for c in childs if c not in parents]
    except Exception:
        leaf_vals = _get_leaves(raw_tree)

    for leaf in leaf_vals:
        node_id_map[int(leaf)] = current_id
        current_id += 1

    # ensure mapping covers all ids in linkage_matrix
    try:
        all_ids = set()
        for row in linkage_matrix:
            all_ids.add(int(row[0]))
            all_ids.add(int(row[1]))
            all_ids.add(int(row[2]))
        remaining = sorted([a for a in all_ids if a not in node_id_map])
        for aid in remaining:
            node_id_map[aid] = current_id
            current_id += 1
    except Exception:
        pass

    max_lambda = max(row[3] for row in linkage_matrix) if linkage_matrix else 0.0
    linkage_matrix_mapped = [
        [node_id_map[row[0]], node_id_map[row[1]], node_id_map[row[2]], max_lambda - row[3], row[4]]
        for row in reversed(linkage_matrix)
    ]

    return np.array(linkage_matrix_mapped), node_id_map


def compute_dendrogram_coords(Z, n_points):
    """Compute drawing coordinates for a linkage matrix `Z`.

    Parameters:
    - Z: array-like with rows [c1, c2, dist, count]
    - n_points: number of leaves

    Returns a tuple: `(icoord, dcoord, leaf_order, nodes)` where:
    - `icoord`/`dcoord` are lists used to build segment geometry
    - `leaf_order` is the ordered list of leaf indices used for x placement
    - `nodes` is a list of node dicts (each with `x`,`y`,`size`,`left`,`right`) and
      can be emitted as JSON for a D3 client.

    This function performs only coordinate computation and constructs a
    lightweight node representation suitable for serialization. It does not
    create Plotly traces or perform any rendering.
    """
    n_nodes = 2 * n_points - 1
    nodes = [{'x': None, 'y': 0.0, 'size': 1, 'left': None, 'right': None} for _ in range(n_points)]

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

    return icoord, dcoord, leaf_order, nodes


def get_segments_from_model(model):
    """Build a list of line segments from a precomputed dendrogram model.

    `model` is expected to be a dict-like or tuple containing `icoord` and
    `dcoord` produced by `compute_dendrogram_coords`. This function performs
    only geometry assembly and returns a plain list of segments
    (each segment is a pair of (x,y) tuples) ready for plotting or
    serialization to JSON.
    """
    icoord = model[0]
    dcoord = model[1]
    segments = []
    for icoords, dcoords in zip(icoord, dcoord):
        x1, x2, x3, x4 = icoords
        y1, y2, y3, y4 = dcoords
        segments.append([(x1, y1), (x2, y2)])
        segments.append([(x2, y2), (x3, y3)])
        segments.append([(x4, y4), (x3, y3)])
    return segments


def get_dendrogram_segments2(Z: np.ndarray):
    """Compatibility wrapper: compute coordinates from Z then return segments.

    This thin wrapper keeps the old API while internally separating
    coordinate computation from segment assembly so callers can request the
    model (coords + nodes) separately if they want to serve JSON to D3.
    """
    n_points = Z.shape[0] + 1
    icoord, dcoord, leaf_order, nodes = compute_dendrogram_coords(Z, n_points)
    model = (icoord, dcoord, leaf_order, nodes)
    return get_segments_from_model(model)
