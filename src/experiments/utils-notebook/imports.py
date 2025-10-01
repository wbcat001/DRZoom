# notebook_imports.py

# -------------------------
# 基本ライブラリ
# -------------------------
import os
import sys
import time
import math
import random
import warnings

# データ操作
import numpy as np
import pandas as pd

# データ取得
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 次元削減
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE

# クラスタリング
import hdbscan

# 評価指標
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 可視化
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go



