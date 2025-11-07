import requests
import json
import random
from typing import List

# --- 設定 ---
# サーバーURLとポートは、FastAPIスクリプトの実行環境に合わせてください。
# 例: uvicorn rapids_gpu_api_extended:app --reload がローカルで実行されている場合
API_URL = "http://127.0.0.1:8000/umap_subset" 

# サーバー側で定義されたデータセットの総点数 (N_TOTAL=100000)
N_TOTAL_POINTS = 500000 
# リクエストで処理したい点の数
N_SUBSET = 5000
# UMAPのパラメータ
N_COMPONENTS = 2
N_NEIGHBORS = 15
MIN_DIST = 0.1

def generate_random_ids(total_range: int, count: int) -> List[int]:
    """
    0からtotal_range-1の範囲から、count個のランダムな一意のIDを生成します。
    """
    if count > total_range:
        count = total_range
    return random.sample(range(total_range), count)

def run_umap_subset_request():
    """
    サーバーにUMAPサブセット計算リクエストを送信します。
    """
    print(f"--- UMAPサブセット計算リクエスト ---")
    print(f"全データ点数: {N_TOTAL_POINTS}点")
    print(f"抽出するサブセット点数: {N_SUBSET}点")

    # 1. リクエスト用のランダムなIDリストを生成
    id_list = generate_random_ids(N_TOTAL_POINTS, N_SUBSET)
    
    # 2. リクエストボディを構築
    payload = {
        "id_list": id_list,
        "n_components": N_COMPONENTS,
        "n_neighbors": N_NEIGHBORS,
        "min_dist": MIN_DIST,
        "metric": "euclidean"
    }

    # 3. POSTリクエストを送信
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status() # HTTPエラーが発生した場合に例外を投げる
        
        # 4. レスポンスを処理
        result = response.json()

        print("\n--- サーバーからの応答 ---")
        print(f"HTTPステータス: {response.status_code}")
        print(f"リクエスト点数: {result.get('requested_ids_count')}点")
        print(f"実際のサブセット形状: {result.get('subset_shape')}")
        print(f"GPU実行時間: {result.get('execution_time_sec'):.4f} 秒")
        
        # 埋め込み結果の確認
        embedding = result.get('embedding')
        if embedding:
            print(f"埋め込み結果の形状: ({len(embedding)}, {len(embedding[0])})")
            print(f"最初の1点の埋め込み: {embedding[0]}")
        
    except requests.exceptions.RequestException as e:
        print(f"\n--- リクエストエラー ---")
        print(f"APIサーバーへの接続に失敗しました。FastAPIサーバーが {API_URL} で起動しているか確認してください。")
        print(f"エラー詳細: {e}")
        if 'response' in locals() and response is not None:
             print(f"サーバー応答: {response.text}")


if __name__ == "__main__":
    run_umap_subset_request()