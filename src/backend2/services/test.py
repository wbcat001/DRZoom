from typing import TypedDict, Literal, Union, List

ProcessType = Literal["dimensionality_reduction", "alignment"]
DimensionalityReductionType = Literal["pca", "tsne", "custom_pca"]
AlignmentType = Literal["procrustes", "none"]

class DimensionalityReductionConfig(TypedDict):
    type: Literal["dimensionality_reduction"]
    method: DimensionalityReductionType
    # オプションの追加設定も可
    # n_components: int

class AlignmentConfig(TypedDict):
    type: Literal["alignment"]
    method: AlignmentType
    # optional parameters
    # normalize: bool

PipelineConfig = Union[DimensionalityReductionConfig, AlignmentConfig]

# リストで受け取れるようにする
Pipeline = List[PipelineConfig]


#### カリー化 #####

from typing import Callable
import numpy as np

def align_handler(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # 例: 単純な差分を取る処理（実際にはプロクルステスなど）
    return X - Y

def get_align_function(X: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def align_with_X(Y: np.ndarray) -> np.ndarray:
        return align_handler(X, Y)
    return align_with_X
