import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Literal, Union, List
from dataclasses import dataclass
from abc import ABC
from core import BaseConfig

# 処理の種類の定義
ProcessType = Literal["dimensionality_reduction", "alignment"]  # , "Scaling"
DimensionalityReductionType = Literal["pca", "tsne", "custom_pca"]
AlignmentType = Literal["procrustes", "none"]
# ScalingType = Literal["none", "zscore"]


@dataclass
class BaseProcessConfig(ABC):
    type: ProcessType


@dataclass
class DimensionalityReductionConfig(BaseProcessConfig):
    type = "dimensionality_reduction"
    method: DimensionalityReductionType = "pca"
    n_components: int = 2  # optional


@dataclass
class AlignmentConfig(BaseProcessConfig):
    type: Literal["alignment"] = "alignment"
    method: AlignmentType = "procrustes"


# class ScalingConfig(BaseProcessConfig):
#     type: Literal["scaling"] = "scaling"
#     method: Literal["zscore"] = "zscore"


# パイプライン生成時の設定
ProcessConfig = Union[
    DimensionalityReductionConfig, AlignmentConfig
]  # , ScalingConfig]
PipelineConfig = List[ProcessConfig]


# アプリ全体の設定
class Config(BaseConfig):
    data: str = "default"
    dimensionality_reduction_config: DimensionalityReductionConfig
    alignment_config: AlignmentConfig

    # scaling_config: ScalingConfig = ScalingConfig()
    def __init__(
        self,
        data: str,
        dimensionality_reduction_config: DimensionalityReductionConfig,
        alignment_config: AlignmentConfig,
    ):
        self.data = data
        self.dimensionality_reduction_config = dimensionality_reduction_config
        self.alignment_config = alignment_config
        # self.scaling_config = ScalingConfig()
