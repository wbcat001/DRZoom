import numpy as np

from itertools import combinations


def print_bounds(matrix):
    """
    行列 [x, y] の列から x_min, x_max, y_min, y_max を計算して出力する関数。

    Args:
        matrix (numpy.ndarray): [x, y] の列を持つ2次元配列。
    """
    x_min, x_max = np.min(matrix[:, 0]), np.max(matrix[:, 0])
    y_min, y_max = np.min(matrix[:, 1]), np.max(matrix[:, 1])
    print(f"x_min: {x_min:.4f}, x_max: {x_max:.4f}, y_min: {y_min:.4f}, y_max: {y_max:.4f}")



# ノードの移動平均距離
def calc_average_displacement(before: np.ndarray, after: np.ndarray) -> float:

    if before.shape != after.shape:
        raise ValueError("The shape of the two arrays must be the same.")
    displacement = np.linalg.norm(after - before, axis=1)
    average_displacement = np.mean(displacement)

    print_bounds(before)
    print_bounds(after)
    print("average_displacement", f"{average_displacement:.4f}")

    return average_displacement


def calc_orthognal_order_comsistency(before: np.ndarray, after: np.ndarray) -> float:
    """
    直行順序保持率を計算する。

    Parameters:
        before (np.ndarray): 再レイアウト前の座標データ (N, 2)。
        after (np.ndarray): 再レイアウト後の座標データ (N, 2)。

    Returns:
        float: 直行順序保持率 (0.0〜1.0)。
    """
    if before.shape != after.shape:
        raise ValueError("The shape of 'before' and 'after' arrays must match.")
    if before.shape[1] != 2:
        raise ValueError("Input arrays must have shape (N, 2) for 2D data.")

    # 全てのペアのインデックスを取得
    indices = list(combinations(range(before.shape[0]), 2))
    consistent_count = 0

    # 各ペアについて順序の一致を確認
    for i, j in indices:
        # 直行順序の比較
        before_x_order = before[i, 0] < before[j, 0]
        before_y_order = before[i, 1] < before[j, 1]
        after_x_order = after[i, 0] < after[j, 0]
        after_y_order = after[i, 1] < after[j, 1]

        # 順序が一致しているか
        if before_x_order == after_x_order and before_y_order == after_y_order:
            consistent_count += 1

    # 全ペア数
    total_pairs = len(indices)

    # 直行順序保持率
    consistency_rate = consistent_count / total_pairs
    print("consistency_rate", f"{consistency_rate:.4f}")

    return consistency_rate
    


def calc_angle_changes(before: np.ndarray, after: np.ndarray, edges: list) -> float:
    """
    リンク（エッジ）の角度変化度合いを計算する。
    
    Parameters:
        before (np.ndarray): 再レイアウト前の座標データ (N, 2)。
        after (np.ndarray): 再レイアウト後の座標データ (N, 2)。
        edges (list of tuple): 各リンクを定義するエッジのリスト [(i, j), ...]。

    Returns:
        float: 平均角度変化度合い。
    """
    if before.shape != after.shape:
        raise ValueError("The shape of 'before' and 'after' arrays must match.")
    if before.shape[1] != 2:
        raise ValueError("Input arrays must have shape (N, 2) for 2D data.")
    
    angle_changes = []

    for i, j in edges:
        # 再レイアウト前のリンクの角度
        dy_before = before[j, 1] - before[i, 1]
        dx_before = before[j, 0] - before[i, 0]
        theta_before = np.arctan2(dy_before, dx_before)

        # 再レイアウト後のリンクの角度
        dy_after = after[j, 1] - after[i, 1]
        dx_after = after[j, 0] - after[i, 0]
        theta_after = np.arctan2(dy_after, dx_after)

        # 角度差を計算 (ラジアン)
        delta_theta = np.abs(theta_after - theta_before)

        # 正規化: 角度差は [0, π] に収める
        delta_theta = np.minimum(delta_theta, 2 * np.pi - delta_theta)

        angle_changes.append(delta_theta)

    # 平均角度変化
    mean_angle_change = np.mean(angle_changes)
    print("mean_angle_change", mean_angle_change)

    return mean_angle_change

# test metrix
if __name__ == "__main__":
    before = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    after = np.array([[0, 1], [1, 2], [2, 4], [3, 10]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    print(f"calc_average_displacement: {calc_average_displacement(before, after)}") # answer: 2.5
    print(f"calc_orthognal_order_comsistency: {calc_orthognal_order_comsistency(before, after)}") # answer: 0.0
    print(f"calculate_angle_changes: {calc_angle_changes(before, after, edges)}") # answer: 0.0
   