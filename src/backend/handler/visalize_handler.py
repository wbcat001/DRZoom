"""
描画のための座標列を生成する

インデックスを受取、
- データをフィルタ(n)
- 次元削減(n)
- アライメント(一つ前の結果から?)
- (クラスタリング)
を行う

"""

class TransitionData:
    def __init__(self, data, frame:int=2):
        self.data = data
        self.frame = frame

class VisualizeHandler:
    def __init__(self, data, index):
        