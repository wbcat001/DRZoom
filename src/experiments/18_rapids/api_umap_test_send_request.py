import numpy as np
import requests
import json
import time
import sys

# --- 設定 ---
API_URL = "http://localhost:8000/umap"
NUM_ROWS = 1000  # 行数 (データポイント数)
NUM_COLS = 1000   # 列数 (特徴量数)
N_COMPONENTS = 2  # UMAPで削減後の次元数
# ------------

print(f"--- APIテスト開始 ---")
print(f"エンドポイント: {API_URL}")
print(f"データサイズ: {NUM_ROWS}行 x {NUM_COLS}列")

# 1. 大容量データの生成
print("1. ランダムデータの生成中...")
try:
    # 10000 x 1000 のランダムな浮動小数点データを生成
    # メモリを節約するため、float32 (単精度浮動小数点数) を使用
    data_array = np.random.rand(NUM_ROWS, NUM_COLS).astype(np.float32)
    
    # Pythonのリスト形式に変換 (JSONとして送信するため)
    # このToList()操作には時間がかかる場合があります
    data_list = data_array.tolist()
    
    # 送信データオブジェクトの作成
    payload = {
        "data": data_list,
        "n_components": N_COMPONENTS
        # n_neighbors, min_dist などのパラメータは省略するとデフォルト値が使われます
    }
    
    # JSONペイロードの推定サイズを計算 (バイト)
    # float32は通常4バイトですが、JSON文字列に変換されると10文字以上になるため、大きめに推定
    estimated_size_mb = sys.getsizeof(data_list) / (1024 * 1024)
    print(f"   メモリ上のデータサイズ (リスト変換後, 推定): 約 {estimated_size_mb:.2f} MB")

except Exception as e:
    print(f"データの生成中にエラーが発生しました: {e}")
    sys.exit(1)

# 2. APIの呼び出し
print("2. APIを叩いています...")
try:
    start_time = time.time()
    
    # POSTリクエストの送信
    # requestsは自動的にPythonオブジェクトをJSONに変換します
    response = requests.post(
        API_URL, 
        json=payload,
        timeout=300  # 大容量データの処理には時間がかかる可能性があるため、タイムアウトを長めに設定 (5分)
    )
    
    end_time = time.time()
    request_duration = end_time - start_time
    
    # 3. 結果の確認
    response.raise_for_status() # HTTPエラーが発生した場合に例外を発生させる
    
    result = response.json()
    
    # 結果の表示
    print("\n--- 応答結果 ---")
    print(f"HTTPステータス: {response.status_code}")
    print(f"リクエスト送信から応答までの時間: {request_duration:.4f} 秒")
    
    api_time = result.get("execution_time_sec")
    if api_time is not None:
        print(f"FastAPI/GPU 処理時間 ({NUM_ROWS}x{NUM_COLS}): {api_time:.4f} 秒")
    
    embedding = result.get("embedding")
    if embedding and isinstance(embedding, list):
        print(f"埋め込み結果の形状: {len(embedding)} x {len(embedding[0])}")
        print(f"先頭3行の埋め込み: {embedding[:3]}")

except requests.exceptions.HTTPError as e:
    print(f"\n--- エラー発生 ---")
    print(f"HTTPエラー: {e}")
    print(f"応答内容:\n{response.text}")
except requests.exceptions.RequestException as e:
    print(f"\n--- エラー発生 ---")
    print(f"リクエストエラー (接続/タイムアウト): {e}")
except Exception as e:
    print(f"\n--- 予期せぬエラー ---")
    print(f"エラー: {e}")

print("\n--- テスト完了 ---")