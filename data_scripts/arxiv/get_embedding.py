import os
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

def get_metadata_df(file_path):
    """
    メタデータをDataFrameに変換する関数
    """
    print(f"Attempting to read metadata from: {file_path}")

    # ファイルが存在することを確認
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"File exists, size: {os.path.getsize(file_path)} bytes")

    try:
        # pd.read_jsonを使用してデータを読み込む
        df = pd.read_json(file_path, lines=True)
    except ValueError as e:
        print(f"Error reading JSON file: {e}")
        raise

    return df

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# File paths
input_file = 'c:/Users/acero/Work_Research/DRZoom/data/arxiv/metadata_0.json'
output_file = 'c:/Users/acero/Work_Research/DRZoom/data/arxiv/abstract_embeddings.pt'

# Read metadata file using the helper function
metadata_df = get_metadata_df(input_file)

# Extract abstracts and compute embeddings
abstract_embeddings = {}
for _, entry in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing abstracts"):
    abstract = entry.get('abstract', '')
    if abstract:
        # Tokenize and encode the abstract
        inputs = tokenizer(abstract, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the [CLS] token representation as the embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        abstract_embeddings[entry['id']] = cls_embedding

# Save embeddings to a file
torch.save(abstract_embeddings, output_file)

print(f"Embeddings saved to {output_file}")