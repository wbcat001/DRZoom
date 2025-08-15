# word2vecのベクトルデータを作って保存する, gensim

import gensim
from gensim.models import KeyedVectors

import numpy as np

w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

words = w2v.index_to_key[:1000]  # 最初の1000語を取得

vectors = np.array([w2v[word] for word in words])

np.save('word_vectors.npy', vectors)