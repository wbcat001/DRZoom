import json
import os
import pandas as pd
import sys
import traceback

def get_metadata_df(file_path):
    """
    メタデータをDataFrameに変換する関数
    """
    print(f"Attempting to read metadata from: {file_path}")

    # ファイルが存在することを確認
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        # 親ディレクトリの内容を表示
        parent_dir = os.path.dirname(file_path)
        print(f"Contents of {parent_dir}:")
        if os.path.exists(parent_dir):
            print(os.listdir(parent_dir))
        else:
            print(f"Parent directory does not exist: {parent_dir}")
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"File exists, size: {os.path.getsize(file_path)} bytes")

    try:
        # pd.read_jsonを使用してデータを読み込む
        df = pd.read_json(file_path, lines=True)
    except ValueError as e:
        print(f"Error reading JSON file: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    return df

if __name__ == "__main__":
    # 現在のワーキングディレクトリを表示
    print(f"Current working directory: {os.getcwd()}")
    
    # 絶対パスを使用していた部分を、相対パスを取得するように変更
    relative_path = os.path.join(os.getcwd(), 'data', 'arxiv', 'metadata_0.json')

    try:
        # メタデータを読み込む
        df = get_metadata_df(relative_path)
        
        if df.empty:
            print("DataFrame is empty. Exiting.")
            sys.exit(1)
        
        # データの基本情報を表示
        print("\n--- Dataset Info ---")
        print(f"Number of papers: {len(df)}")
        print("\n--- Columns ---")
        print(df.columns.tolist())
        
        # 各カラムのデータ型を表示
        print("\n--- Data Types ---")
        print(df.dtypes)
        
        # サンプルデータを表示
        print("\n--- Sample Data ---")
        print(df.head(2).to_string())
        
        # カテゴリの分布を表示
        if 'categories' in df.columns:
            print("\n--- Categories Distribution ---")
            # カテゴリは複数ある場合があるので、スペースで区切られた文字列になっている場合がある
            all_categories = df['categories'].str.split().explode().value_counts().head(10)
            print(all_categories)
        
        # 年ごとの論文数（バージョン1の作成日から年を抽出）
        if 'versions' in df.columns and df['versions'].apply(lambda x: isinstance(x, list) and len(x) > 0).all():
            print("\n--- Papers by Year ---")
            # 最初のバージョンの作成日から年を抽出
            df['year'] = df['versions'].apply(lambda x: x[0]['created'].split()[-1] if isinstance(x, list) and len(x) > 0 else None)
            years_count = df['year'].value_counts().sort_index()
            print(years_count)
        
        # カラムごとの欠損値の数を表示
        print("\n--- Missing Values ---")
        print(df.isna().sum())
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

