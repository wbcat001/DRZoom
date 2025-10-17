import numpy as np
import umap.umap_ as umap
from scipy.linalg import orthogonal_procrustes
from typing import List, Dict, Tuple, Any
from data_manager import DataManager 
import hdbscan

class AnalysisEngine:
    def __init__(self, dm: DataManager):
        self.dm = dm
        self.default_n_neighbors = 15
        self.default_min_dist = 0.1

    def _get_lambda_from_level(self, zoom_level: int) -> float:
        """ズームレベル（1, 2, ...）から対応するHDBSCANのラムダ値を取得"""
        level_index = min(zoom_level - 1, len(self.dm.level_lambdas) - 1)
        return self.dm.level_lambdas[level_index]

    def _extract_representative_points(self, current_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        指定ラムダ値でクラスタリングし、各クラスタから最も安定した点（Core Distance最小）を抽出する。
        """
        
        # 1. 指定lambda値でのクラスタリング結果（ラベル）を取得
        labels, _ = hdbscan.label.hdbscan_tree_to_labels(
            self.dm.hdbscan_tree.condensed_tree_, 
            self.dm.hdbscan_tree._raw_data.shape[0], 
            lambda_val=current_lambda
        )
        
        # 2. HDBSCANによって計算されたCore Distanceを取得
        # これがクラスタ内の「安定性」または「中心性」を示す
        core_distances = self.dm.hdbscan_tree.core_distances_ 
        
        representative_indices = []
        unique_clusters = np.unique(labels[labels != -1])
        
        for cid in unique_clusters:
            member_indices = np.where(labels == cid)[0]
            if len(member_indices) == 0: continue
            
            # クラスタメンバーのCore Distanceを取得し、最小値のインデックスを見つける
            member_core_distances = core_distances[member_indices]
            core_point_index_in_member = np.argmin(member_core_distances)
            
            # 全体インデックスに変換してリストに追加
            representative_indices.append(member_indices[core_point_index_in_member]) 

        return np.array(representative_indices), labels

    def generate_umap_embedding(self, 
                            zoom_level: int, 
                            prev_embedding_id: str = None
                           ) -> Dict[str, Any]:
        # ... (1. 代表点とクラスタリング結果を取得 のコードは省略) ...
        
        # 2. 初期配置の設定 (メンタルマップ保存の厳密化)
        initial_coords = 'spectral'
        
        if prev_embedding_id in self.dm.current_embeddings:
            prev_coords = self.dm.current_embeddings[prev_embedding_id]
            
            # 前のレベルの全点のインデックスを取得
            # (ここでは簡略化のため、前の埋め込みは常に代表点レベルであると仮定)
            # 厳密には、Data Managerに「どのインデックスが埋め込まれたか」を保存しておく必要があります。
            # 例: self.dm.embedded_indices[prev_embedding_id]
            # ここでは、prev_coordsは全体データに対応し、代表点がサブセットとして抽出されると仮定。
            
            # 仮定: 前のレベルが現在の代表点の上位集合である（または初期配置として利用可能）
            # 実際には、target_indicesとprev_embedded_indicesの共通部分を取り、対応する座標を抽出する必要があります。
            
            # 簡略化された初期配置の利用:
            if prev_coords.shape[0] == self.dm.N: # 前の埋め込みが全点の場合
                initial_coords = prev_coords[target_indices]
            elif prev_coords.shape[0] == target_vectors.shape[0]: # 前の埋め込みと現在の埋め込みの代表点の数が同じ場合
                initial_coords = prev_coords
            else:
                # 代表点の対応付けが複雑な場合、UMAPのデフォルト初期化または'spectral'に頼る
                initial_coords = 'spectral'
                
        # 3. UMAP計算
        mapper = umap.UMAP(
            # ... (n_neighbors, min_distなどの設定は省略) ...
            init=initial_coords, 
            random_state=42 
        ).fit(target_vectors)
        
        # 4. Procrustes解析によるアライメント
        aligned_coords = new_coords
        if prev_coords is not None and prev_coords.shape[0] == new_coords.shape[0]:
             # 代表点の数が同じ場合のみアライメント（ここでは非常に簡略化）
             R, _ = orthogonal_procrustes(new_coords, prev_coords)
             aligned_coords = new_coords @ R
        # TODO: 代表点の数が異なる場合（通常こちら）、アライメント処理はより複雑になる。
        # アライメントには、両レベルに共通する「ランドマーク」または「安定したクラスタの中心」のみを使用するべき。
        
        # 5. 結果を保存
        self.dm.current_embeddings[level_id] = aligned_coords
        
        return {
            "level_id": level_id,
            "coordinates": aligned_coords.tolist(), # JSON用にリスト化
            "indices": target_indices.tolist(),
            "cluster_labels": current_labels.tolist(),
            "is_representative": True
        }
    
    def generate_contour_data(self, level_id: str, cluster_id: int) -> Dict[str, Any]:
        """
        指定された埋め込みレベルとクラスタIDに基づき、等高線描画用のデータを生成する。
        """
        if level_id not in self.dm.current_embeddings:
            return {"error": "Invalid level_id"}
        
        coords = self.dm.current_embeddings[level_id]
        
        # 1. 埋め込まれた点のラベルを取得
        # HDBSCANのラベルは、埋め込み計算時に取得済みと仮定（ここでは便宜的に全点のラベルを使用）
        all_labels, _ = self._extract_representative_points(self._get_lambda_from_level(int(level_id.split('_')[1])))
        
        # 2. 対象クラスタの点のみを抽出
        # 実際には、埋め込みに含まれる点（代表点）のラベルを利用する
        target_indices = self.dm.embedded_indices.get(level_id, np.arange(coords.shape[0])) # 埋め込まれたインデックスを取得する仮定
        cluster_labels = all_labels[target_indices]
        
        cluster_coords = coords[cluster_labels == cluster_id]
        
        if cluster_coords.shape[0] < 5:
            return {"contours": []}

        # 3. 2次元埋め込み空間での密度推定 (KDE: Kernel Density Estimation)
        # 非常に計算コストが高くなる可能性があるため、グリッドベースの推定を使用
        # scipy.stats.gaussian_kde や scikit-learn の KernelDensity が利用可能
        
        # 例: 2Dヒストグラム (Heatmap) を使用して密度を近似
        H, xedges, yedges = np.histogram2d(
            cluster_coords[:, 0], 
            cluster_coords[:, 1], 
            bins=50
        )
        
        # 4. 等高線データの抽出 (OpenCVやscipyの等高線抽出機能が必要)
        # ここでは、D3.jsのd3-contourモジュールで処理できる形式にデータを変換することを想定し、
        # シンプルなグリッドデータ（ヒートマップ）を返します。
        
        return {
            "cluster_id": cluster_id,
            "heatmap": H.tolist(),
            "x_range": [xedges[0], xedges[-1]],
            "y_range": [yedges[0], yedges[-1]],
            "description": f"Contour data for cluster {cluster_id} at {level_id}"
        }