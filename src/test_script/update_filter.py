

class TransitionData:
    def __init__(self, position_data, frame:int=1, ):
        self.data_list = [position_data for _ in range(frame)]
        self.frame = frame
        self.indecies = list(range(len(position_data)))

    def update(self, data, indecies):
        self.data_list = data
        self.indecies = indecies

# random array
import numpy as np
sample_position_data = np.random.rand(800,2)
sample_whole_position_data = np.random.rand(1000,2)
transition_data = TransitionData(sample_position_data)

filter_indices = [1,2,3,4,5]
prev_indecies = transition_data.indecies
common_indecies = list(set(filter_indices) & set(prev_indecies))
print(f"common_indecies: {len(common_indecies)}")

# filter
tmp_position_data = sample_whole_position_data[common_indecies]

# index map 元データのidx: tmp_position_dataのidx
index_map = {idx: i for i, idx in enumerate(common_indecies)}

# update idx: 
# update_idx = [index_map[idx] for idx in filter_indices if idx in index_map]

sample_position_data[common_indecies] = tmp_position_data + 1

transition_data.update(sample_position_data, common_indecies)




