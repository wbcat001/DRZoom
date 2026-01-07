# api_client.py
"""
API Client for UMAP Zoom Servers
3つのサーバーに対応:
  - CPU版: http://localhost:8002/api/zoom/redraw (推奨)
  - GPU版: http://localhost:8001/api/zoom/redraw
  - 実験用: http://localhost:8000/recalculate_umap (旧形式)
"""

import requests
import numpy as np
import base64
import io
import time
from typing import Optional

# ============================================================================
# サーバー設定 (どちらかを選択)
# ============================================================================

# CPU版 (推奨: 環境依存なし、メンタルマップ保持対応)
API_BASE_URL = "http://127.0.0.1:8001"
API_ENDPOINT = "/api/zoom/redraw"
SERVER_TYPE = "CPU"

# GPU版 (cuML使用)
# API_BASE_URL = "http://127.0.0.1:8001"
# API_ENDPOINT = "/api/zoom/redraw"
# SERVER_TYPE = "GPU"

# 旧形式 (実験用)
# API_BASE_URL = "http://127.0.0.1:8000"
# API_ENDPOINT = "/recalculate_umap"
# SERVER_TYPE = "Legacy"

API_URL = API_BASE_URL + API_ENDPOINT

# --- ヘルパー関数 (サーバー側と共通) ---

def _numpy_to_b64(array: np.ndarray) -> str:
    """NumPy配列をBase64文字列にエンコードする"""
    buff = io.BytesIO()
    np.save(buff, array, allow_pickle=False) 
    return base64.b64encode(buff.getvalue()).decode('utf-8')

def _b64_to_numpy(data_b64: str) -> np.ndarray:
    """Base64文字列からNumPy配列にデコードする"""
    decoded = base64.b64decode(data_b64)
    return np.load(io.BytesIO(decoded))

# --- データの準備 (擬似データ) ---
N_SAMPLES = 1000
N_FEATURES = 50
np.random.seed(42)

# 擬似的な特徴量ベクトル (ダミーデータ)
X_vectors = np.random.randn(N_SAMPLES, N_FEATURES).astype(np.float32)

# 既存の埋め込み座標 (ダミーデータ, グローバル座標をシミュレート)
# 意図的に異なる範囲で生成し、アライメントの効果を確認しやすくする
initial_embedding_coords = np.random.randn(N_SAMPLES, 2).astype(np.float32) * 50 


# --- API呼び出し関数 ---

def call_umap_api(vectors: np.ndarray, initial_coords: Optional[np.ndarray] = None, api_type: str = "init"):
    print(f"\n--- API Call: {api_type} ({SERVER_TYPE}) ---")
    
    # Base64 エンコード
    vectors_b64 = _numpy_to_b64(vectors)
    initial_embedding_b64 = _numpy_to_b64(initial_coords) if initial_coords is not None else None

    # リクエストペイロード (新形式: /api/zoom/redraw)
    if SERVER_TYPE in ["CPU", "GPU"]:
        payload = {
            "vectors_b64": vectors_b64,
            "initial_embedding_b64": initial_embedding_b64,
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
            "n_epochs": 200
        }
        result_key = "coordinates"
    else:
        # 旧形式: /recalculate_umap
        payload = {
            "vectors_b64": vectors_b64,
            "initial_embedding_b64": initial_embedding_b64,
            "n_epochs": 200
        }
        result_key = "embedding_b64"

    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        end_time = time.time()
        
        response.raise_for_status() # HTTPエラーを確認
        
        result = response.json()
        
        # エラーチェック
        if result.get('status') == 'error':
            print(f"❌ API Error: {result.get('message')}")
            return None
        
        # 結果のデコード
        embedding_b64 = result.get(result_key)
        if not embedding_b64:
            print(f"❌ No coordinates in response")
            return None
            
        new_embedding = _b64_to_numpy(embedding_b64)
        
        print(f"✅ Success! New embedding shape: {new_embedding.shape}")
        print(f"   Range: X=[{new_embedding[:, 0].min():.2f}, {new_embedding[:, 0].max():.2f}], "
              f"Y=[{new_embedding[:, 1].min():.2f}, {new_embedding[:, 1].max():.2f}]")
        print(f"   Calculation time: {end_time - start_time:.2f} seconds")
        return new_embedding

    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print(f"   Server URL: {API_URL}")
        print(f"   Make sure the server is running!")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Decode Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- 実行 ---
if __name__ == "__main__":
    print(f"=" * 60)
    print(f"UMAP Zoom API Client")
    print(f"=" * 60)
    print(f"Server Type: {SERVER_TYPE}")
    print(f"API URL: {API_URL}")
    print(f"=" * 60)
    
    # サーバーが起動していることを確認してください。
    
    # 1. 既存座標を初期配置として使用する (メンタルマップ保持あり)
    embedding_init = call_umap_api(
        X_vectors, 
        initial_coords=initial_embedding_coords, 
        api_type="UMAP Init (Global Coords)"
    )

    # 2. デフォルトの初期配置 (メンタルマップ保持なし)
    embedding_default = call_umap_api(
        X_vectors, 
        initial_coords=None, 
        api_type="UMAP Default Init (Spectral)"
    )

    if embedding_init is not None and embedding_default is not None:
        print("\n" + "=" * 60)
        print("座標再計算の結果比較")
        print("=" * 60)
        
        # メンタルマップ保持ありの場合、結果座標は初期座標のスケールと近くなっているはず
        print(f"\n初期座標の範囲:")
        print(f"  X Min/Max: {initial_embedding_coords[:, 0].min():.2f} / {initial_embedding_coords[:, 0].max():.2f}")
        print(f"  Y Min/Max: {initial_embedding_coords[:, 1].min():.2f} / {initial_embedding_coords[:, 1].max():.2f}")
        
        print(f"\nメンタルマップ保持ありの結果:")
        print(f"  X Min/Max: {embedding_init[:, 0].min():.2f} / {embedding_init[:, 0].max():.2f}")
        print(f"  Y Min/Max: {embedding_init[:, 1].min():.2f} / {embedding_init[:, 1].max():.2f}")
        
        # メンタルマップ保持なしの場合、結果座標はデフォルトのUMAPスケール (通常は小さい範囲) になるはず
        print(f"\nメンタルマップ保持なしの結果:")
        print(f"  X Min/Max: {embedding_default[:, 0].min():.2f} / {embedding_default[:, 0].max():.2f}")
        print(f"  Y Min/Max: {embedding_default[:, 1].min():.2f} / {embedding_default[:, 1].max():.2f}")

        # 座標間の距離を比較することで、結果が異なることを確認
        # メンタルマップ保持ありの方が、初期配置の座標と距離が近い(スケールが似ている)はず
        distance_init = np.linalg.norm(initial_embedding_coords - embedding_init)
        distance_default = np.linalg.norm(initial_embedding_coords - embedding_default)

        print(f"\n初期座標とのL2距離:")
        print(f"  メンタルマップ保持あり:  {distance_init:.2f}")
        print(f"  メンタルマップ保持なし: {distance_default:.2f}")
        
        # 期待される結果: distance_init < distance_default
        print(f"\n判定:")
        if distance_init < distance_default:
            print(f"  ✅ メンタルマップ保持が機能しています。")
            print(f"     初期配置のスケール/位置が継承されています。")
        else:
            print(f"  ⚠️ 結果がほぼ同じか、メンタルマップ保持の効果が小さい可能性があります。")