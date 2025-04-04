from core import BaseProcessManager
from abc import ABC, abstractmethod
import numpy as np
# Dict形式の
ProcessType =  

class ProcessManager(BaseProcessManager):
    def __init__(self):
        super().__init__()

    def process(self, data):
        # ここを変更することで、パイプラインの構築を変えたり色々
        return data
    


## 設定配列を元に、パイプラインを構築する
## (process_type -> 選択)[] {"dimensionality_reduction": "pca", "alignment": "procrustes"}
class ProcessPipeline:
    def __init__(self): 
    
    def generate(self, process_types: List[ProcessType]):
        # process_typesを元に、Pipelineを生成する
        pass

    def execute(self, data):
        # Implement your pipeline execution logic here
        return data
    

# PipelineのProcessの中でつなぎ合わせるもの
class Processor(ABC):
    def __init__(self):
        pass
        
    def process(self, data):
        pass
        


    



    


    
#### aligner
