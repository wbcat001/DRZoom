import os
from gensim.models import KeyedVectors
from hdbscan.plots import CondensedTree as hdbscan_CondensedTree
import numpy as np
from sklearn.preprocessing import normalize
from datetime import datetime
import plotly.express as px
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

random_state = 42

def load_w2v(n_samples=5000, is_random=True):
    print(os.getcwd())
    print(BASE_DIR)
    file_path = os.path.join(BASE_DIR, "..", "18_rapids", "data", "GoogleNews-vectors-negative300.bin")
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)

    words = model.index_to_key
    print(f"Number of words in the model: {len(words)}")

    if is_random:
        np.random.seed(random_state)
        ramdom_indices = np.random.choice(len(words), size=n_samples, replace=False)

    else:
        ramdom_indices = np.arange(n_samples)
    
    selected_words = [words[i] for i in ramdom_indices]
    selected_vectors = model.vectors[ramdom_indices]

    return selected_vectors, selected_words



# word2vec読み込み
N_100k = 100000
N_500k = 500000

# ----------------------------------------------------
# 10万単語のデータを読み込み・保存
# ----------------------------------------------------
print(f"\n--- Loading and saving data for N={N_100k} ---")
X_100k, words_100k = load_w2v(n_samples=N_100k)
print(f"Loaded {X_100k.shape[0]} word vectors of dimension {X_100k.shape[1]}.")

# 保存ディレクトリの作成
SAVE_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(SAVE_DIR, exist_ok=True)

# データの保存
# 1. ベクトル (X) を .npy 形式で保存
np.save(os.path.join(SAVE_DIR, f"w2v_vectors_{N_100k}.npy"), X_100k)
# 2. 単語 (words) を .txt 形式で保存
with open(os.path.join(SAVE_DIR, f"w2v_words_{N_100k}.txt"), 'w', encoding='utf-8') as f:
    for word in words_100k:
        f.write(word + '\n')
print(f"Data for N={N_100k} saved successfully to '{SAVE_DIR}'.")

# ----------------------------------------------------
# 50万単語のデータを読み込み・保存
# ----------------------------------------------------
print(f"\n--- Loading and saving data for N={N_500k} ---")
X_500k, words_500k = load_w2v(n_samples=N_500k)
print(f"Loaded {X_500k.shape[0]} word vectors of dimension {X_500k.shape[1]}.")

# データの保存
np.save(os.path.join(SAVE_DIR, f"w2v_vectors_{N_500k}.npy"), X_500k)
with open(os.path.join(SAVE_DIR, f"w2v_words_{N_500k}.txt"), 'w', encoding='utf-8') as f:
    for word in words_500k:
        f.write(word + '\n')
print(f"Data for N={N_500k} saved successfully to '{SAVE_DIR}'.")