# api_client.py
import requests
import numpy as np
import base64
import io
import time

API_URL = "http://127.0.0.1:8000/recalculate_umap"

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

def call_umap_api(vectors: np.ndarray, initial_coords: np.ndarray | None = None, api_type: str = "init"):
    print(f"\n--- API Call: {api_type} ---")
    
    # Base64 エンコード
    vectors_b64 = _numpy_to_b64(vectors)
    initial_embedding_b64 = _numpy_to_b64(initial_coords) if initial_coords is not None else None

    # リクエストペイロード
    payload = {
        "vectors_b64": vectors_b64,
        "initial_embedding_b64": initial_embedding_b64,
        "n_epochs": 200 # 計算量を抑えるため
    }

    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        end_time = time.time()
        
        response.raise_for_status() # HTTPエラーを確認
        
        result = response.json()
        
        # 結果のデコード
        embedding_b64 = result['embedding_b64']
        new_embedding = _b64_to_numpy(embedding_b64)
        
        print(f"✅ Success! New embedding shape: {new_embedding.shape}")
        print(f"Calculation time: {end_time - start_time:.2f} seconds")
        return new_embedding

    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Decode Error: {e}")
        return None


# --- 実行 ---
if __name__ == "__main__":
    # サーバーが起動していることを確認してください。
    
    # 1. 既存座標を初期配置として使用する (UMAP initあり)
    embedding_init = call_umap_api(
        X_vectors, 
        initial_coords=initial_embedding_coords, 
        api_type="UMAP Init (Global Coords)"
    )

    # 2. デフォルトの初期配置 (UMAP initなし)
    embedding_default = call_umap_api(
        X_vectors, 
        initial_coords=None, 
        api_type="UMAP Default Init (Spectral)"
    )

    if embedding_init is not None and embedding_default is not None:
        print("\n--- 座標再計算の確認 ---")
        
        # UMAP initありの場合、結果座標は初期座標のスケールと近くなっているはず
        print(f"初期座標 X Min/Max: {initial_embedding_coords[:, 0].min():.2f} / {initial_embedding_coords[:, 0].max():.2f}")
        print(f"Initあり X Min/Max: {embedding_init[:, 0].min():.2f} / {embedding_init[:, 0].max():.2f}")
        
        # UMAP initなしの場合、結果座標はデフォルトのUMAPスケール (通常は小さい範囲) になるはず
        print(f"Initなし X Min/Max: {embedding_default[:, 0].min():.2f} / {embedding_default[:, 0].max():.2f}")

        # 座標間の距離を比較することで、結果が異なることを確認
        # Initありの方が、初期配置の座標と距離が近い(スケールが似ている)はず
        distance_init = np.linalg.norm(initial_embedding_coords - embedding_init)
        distance_default = np.linalg.norm(initial_embedding_coords - embedding_default)

        print(f"\n初期座標とのL2距離 (Initあり):   {distance_init:.2f}")
        print(f"初期座標とのL2距離 (Initなし): {distance_default:.2f}")
        
        # 期待される結果: distance_init < distance_default
        if distance_init < distance_default:
            print("✅ UMAP initが初期配置のスケール/位置を継承し、メンタルマップ維持が機能しています。")
        else:
            print("⚠️ UMAP initの効果が薄いか、初期配置のスケールが大きすぎる可能性があります。")