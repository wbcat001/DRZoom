import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import pandas as pd
import pickle
from scipy.spatial import procrustes
from typing import Tuple
import random

from core_color import generate_colormap

# df, embeddigngsのうち、dfはなくても良い用にする

class Data:
    def __init__(self, df:pd.DataFrame, embeddings: np.ndarray, window: int = 30, stride:int = 1):
        
        self.df = df
        self.embeddings = embeddings
        self.slided_embeddings = self.calc_slided_embeddings(window=window, stride=stride)
        # self.df = self.calc_slided_df( window=window, stride=stride)
        print(f"data length: {len(self.df)}, {len(self.slided_embeddings)}")
        # self.df["_Index"] = self.df.reset_index().index
        self.indices = self.df["_Index"]
        self.window = window
        self.stride = stride

    def calc_slided_embeddings(self, window: int = 50,stride:int = 1): 
        
        num_vectors = len(self.embeddings)
        result = []
        
        for i in range(num_vectors):
            # ウィンドウ内のベクトルを収集
            window_vectors = []
            for j in range(window):
                index = (i + j) % num_vectors  # 循環インデックス
                window_vectors.append(self.embeddings[index])

            # ウィンドウ内の平均を計算
            window_mean = np.mean(window_vectors, axis=0)
            result.append(window_mean)
        return np.array(result)
    
    def calc_slided_embeddings2(self, window: int = 50,stride:int = 1):
        num_vectors = len(self.embeddings)
        result = []
        for i in range(0, num_vectors, stride):
            # ウィンドウ内のベクトルを収集
            window_vectors = self.embeddings[i:i+window]
            window_mean = np.mean(window_vectors, axis=0)
            result.append(window_mean)

        return np.array(result)

    def calc_slided_df(self, window:int = 50, stride:int = 1):
        
        # 畳み込み処理
        result = []
        for start in range(0, len(self.df), stride):
            end = start + window
            if end <= len(self.df):
                # 各ウィンドウの最初の行を参照
                first_row = self.df.iloc[start]
            result.append(first_row)

        # 結果を新しいデータフレームとして構築
        result_df = pd.DataFrame(result)
        return result_df

    def set_slided_embeddings(self):
        pass


class DataManager:
    def __init__(self, dir_path):
        
        self.dir_path = dir_path
        self.directories = self.get_directories(self.dir_path)
        ## Todo 
        # choose file name
        self.file_names = {"df": "df.csv",
                           "embedding": "vector.pkl"}
        
        # read first directory: alice
        self.data = Data(self.load_df(os.path.join(self.dir_path, self.directories[0], self.file_names.get("df"))), self.load_embeddings(os.path.join(self.dir_path, self.directories[0], self.file_names.get("embedding"))))
 
    def get_directories(self, dir_path):
        return [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    
    def load_df(self, file_path):
        return pd.read_csv(file_path, index_col=0)

    def load_embeddings(self, file_path):
        with open(file_path, "rb") as f:
            embeddings = np.array(pickle.load(f))
         
        return embeddings

    def preprocess(self) -> np.ndarray:
        
        return self.data
    
    # Todo
    # 関数名、関数の場所
    def get_colors(self):


        colormap_event =  {"Setup": "skyblue",
                        "Inciting Incident": "green",
                        "Turning Point": "orange",
                        "Climax": "red",
                        "Resolution": "purple",
                        #    "Development": "yellow",
        }
        colors = generate_colormap(self.data.df, "ERole", default_colormap=colormap_event)

        return colors


#### Dimensionality Reduction

class DimensionalityReducer:
    def __init__(self, method: str = "PCA"):
        self.methods = {
            "PCA": self._pca,
            "t-SNE": self._tsne
        }
        self.method = method
    def get_methods(self):
        return self.methods.keys()

    def reduce(self, data: np.ndarray) -> np.ndarray:
        if self.method:
            dr = self.methods.get(self.method, self._pca)
            return dr(data)
        raise ValueError(f"Unknown method:")
    ## Todo
    # PCAの行列を取得するなどできるようにしたい
    # ハイパーパラメータの調整の機能
    def _pca(self, data: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=2)
        return pca.fit_transform(data)

    def _tsne(self, data: np.ndarray) -> np.ndarray:
        tsne = TSNE(n_components=2, random_state=42)
        return tsne.fit_transform(data)
    



#### Alignment

class AlignmentHandler:
    """
    それぞれ次元削減した結果の data1, data2に関して、data2の結果をdata1にアライメントする
    """
    def __init__(self, method: str = "Procrustes"):
        self.methods = {
            "Procrustes": self._procrustes,
            "No": self._no_alignment
        }
        self.method = method
    
    def _procrustes(self, data1:np.ndarray, data2: np.ndarray):
        print("use procrustes")
        center_data1 = np.mean(data1, axis=0)
        print(f"center of data2 : {center_data1}")
        normalize_data1, aligned_data2, d = procrustes(data1, data2)

        scale_factor = np.linalg.norm(data2 - np.mean(data2, axis=0))
        print(f"scale: {scale_factor:4f}")
        aligned_data2 *= scale_factor
        print(f"center of aligned data2{np.mean(aligned_data2, axis=0)}")

        aligned_data2 += center_data1
        return aligned_data2
    
    def _no_alignment(self, data1: np.ndarray, data2: np.ndarray):
        print("use no alignment")
        
        # _, aligned_data2, d = procrustes(data1, data2)
        # print((len(aligned_data2), len(data2)))
        center_data1 = np.mean(data1, axis=0)
        data2 += center_data1
        return data2


    def align(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        # 仮の実装: メソッドに応じて異なるアライメントを実行可能
        if self.method:
            current_method = self.methods.get(self.method, self._procrustes)
            return current_method(before, after)
        raise ValueError(f"Unknown alignment method: ")
    



class TransitionData:
    
    def __init__(self, data: Data, dimensionality_reducer: DimensionalityReducer):
        self.data = data
        self.reducer = dimensionality_reducer
        x_y = dimensionality_reducer.reduce(data.slided_embeddings)
        data.df["x"] = x_y[:, 0]
        data.df["y"] = x_y[:, 1]
        
        self.from_data = data
        # self.from_to_data = data
        self.to_data = data

        # check
        print(f"first df columns are {self.from_data.df.columns}")
    
    def reset(self):
        self.from_data = self.data
        self.to_data = self.data

    def next(self, to_data):
        self.from_data = self.to_data
        self.to_data = to_data

    def get_position_range(self):
        """
        Get the range of x and y coordinates in the data.
        for x and y range in the plot.
        """
        x_min, x_max, y_min, y_max = 100, -100, 100, -100

        
        for data_x in [self.from_data.df["x"], self.to_data.df["x"]]:
            x_min = min(x_min, min(data_x))
            x_max = max(x_max, max(data_x))
            print(x_min, x_max)
        
        for data_y in [self.from_data.df["y"], self.to_data.df["y"]]:
            y_min = min(y_min, min(data_y))
            y_max = max(y_max, max(data_y))
            print(y_min, y_max)

        return x_min, x_max, y_min, y_max
    


class AnimationManager:

    def __init__(self, 
                 data_manager: DataManager,alignment_handler: AlignmentHandler, dimensionality_reducer: DimensionalityReducer):

        self.alignment_handler = alignment_handler
        self.dimensionality_reducer = dimensionality_reducer
        
        self.data_manager = data_manager
        self.df_full = data_manager.data.df

    ## Todo
    # animationの種類への対応: scroll, squere_area, auto

    def initialize_figure(self):
        pass


    def create_frames(self, x_min, x_max, y_min, y_max, transition_data:TransitionData, steps: int = 1) -> Tuple[list, TransitionData]:
        """
        create frames for animation


        """
        ## Todo 
        # 多段階のアニメーション


        frames = []

        # エリアを下にデータをフィルタする
        new_data, mask = self.filter(transition_data.to_data, x_min, x_max, y_min, y_max)

        transition_data.next(new_data)

        # フレームを作成(from)
        # from_dataの確認
        print(f"from_data: {transition_data.from_data.df.columns}")
        print(f"to_data: {transition_data.to_data.df.columns}")

        print(f'{transition_data.from_data.df[transition_data.from_data.df["_Index"].isin(transition_data.to_data.indices)][["x", "y"]].to_numpy().shape}')

        print(f'same shape? {transition_data.from_data.df[transition_data.from_data.df["_Index"].isin(transition_data.to_data.indices)][["x", "y"]].to_numpy().shape} {self.dimensionality_reducer.reduce(transition_data.to_data.slided_embeddings).shape}')

        
        # フレームを作成(to)
        frame2 = self.alignment_handler.align(transition_data.from_data.df[transition_data.from_data.df["_Index"].isin(transition_data.to_data.indices)][["x", "y"]].to_numpy(), 
        self.dimensionality_reducer.reduce(transition_data.to_data.slided_embeddings))
        
        
        frames.append(transition_data.from_data.df[transition_data.from_data.df["_Index"].isin(transition_data.to_data.indices)][["x", "y"]].to_numpy())
        frames.append(frame2)

        # TransitionData の更新
        transition_data.to_data.df["x"] = frame2[:, 0]
        transition_data.to_data.df["y"] = frame2[:, 1]


        return frames, transition_data
    
    def filter(self, data:Data, x_min, x_max, y_min, y_max):
        
        df = data.df.copy()
        # emb = data.slided_embeddings
        filtered_df = df[(df['x'] >= x_min) & (df['x'] <= x_max) & (df['y'] >= y_min) & (df['y'] <= y_max)]
        filtered_indices = filtered_df.index
  

        print(f"{len(df)} {len(filtered_df)} {len(filtered_indices)}")
        
        filtered_embeddings = self.data_manager.data.embeddings[filtered_indices]
        print(f"len emb {len(filtered_embeddings)}")
        return Data(filtered_df, filtered_embeddings), filtered_indices


