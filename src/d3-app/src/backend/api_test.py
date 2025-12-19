"""
API Endpoint Test Script

バックエンドのAPIエンドポイントに対してリクエストを送り、
レスポンスの構造や配列長などをチェック
"""

import requests
import json
from typing import Any, Dict, List
from pathlib import Path

# API設定
API_BASE_URL = "http://localhost:8000/api"

# テスト用の詳細表示
def print_response_info(endpoint: str, response: requests.Response):
    """レスポンス情報を整形して表示"""
    print(f"\n{'='*70}")
    print(f"Endpoint: {endpoint}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Error: {response.text[:500]}")
        return
    
    try:
        data = response.json()
        print_data_structure(data, indent=0)
    except Exception as e:
        print(f"Error parsing JSON: {e}")


def print_data_structure(data: Any, indent: int = 0, max_depth: int = 3):
    """データ構造を再帰的に表示"""
    prefix = "  " * indent
    
    if indent > max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    if isinstance(data, dict):
        print(f"{prefix}{{ (dict, {len(data)} keys)")
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}  '{key}':")
                print_data_structure(value, indent + 2, max_depth)
            else:
                value_type = type(value).__name__
                if isinstance(value, str) and len(value) > 50:
                    print(f"{prefix}  '{key}': str ({len(value)} chars)")
                elif isinstance(value, (int, float, bool, type(None))):
                    print(f"{prefix}  '{key}': {value_type} = {value}")
                else:
                    print(f"{prefix}  '{key}': {value_type}")
        print(f"{prefix}}}")
    
    elif isinstance(data, list):
        if len(data) == 0:
            print(f"{prefix}[] (empty list)")
        else:
            first_elem = data[0]
            print(f"{prefix}[ (list, {len(data)} elements)")
            print(f"{prefix}  [0] element structure:")
            print_data_structure(first_elem, indent + 2, max_depth)
            if len(data) > 1:
                print(f"{prefix}  ... ({len(data) - 1} more elements)")
            print(f"{prefix}]")
    
    elif isinstance(data, str):
        if len(data) > 100:
            print(f"{prefix}str ({len(data)} chars): {data[:100]}...")
        else:
            print(f"{prefix}str: '{data}'")
    
    else:
        print(f"{prefix}{type(data).__name__}: {data}")


def check_array_lengths(data: Any, path: str = "root") -> Dict[str, int]:
    """配列の長さを再帰的に収集"""
    lengths = {}
    
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}"
            if isinstance(value, list):
                lengths[new_path] = len(value)
            if isinstance(value, (dict, list)):
                lengths.update(check_array_lengths(value, new_path))
    
    elif isinstance(data, list):
        lengths[path] = len(data)
        if len(data) > 0 and isinstance(data[0], (dict, list)):
            for i, item in enumerate(data[:1]):  # Only first item
                lengths.update(check_array_lengths(item, f"{path}[{i}]"))
    
    return lengths


def test_datasets():
    """GET /api/datasets"""
    print("\n" + "="*70)
    print("TEST 1: /api/datasets")
    print("="*70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/datasets")
        print_response_info("/datasets", response)
        
        if response.status_code == 200:
            data = response.json()
            lengths = check_array_lengths(data)
            print("\nArray lengths:")
            for path, length in lengths.items():
                print(f"  {path}: {length}")
    
    except Exception as e:
        print(f"Error: {e}")


def test_initial_data():
    """GET /api/initial_data"""
    print("\n" + "="*70)
    print("TEST 2: /api/initial_data")
    print("="*70)
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/initial_data",
            params={"dataset": "default", "dr_method": "umap"}
        )
        print_response_info("/initial_data", response)
        
        if response.status_code == 200:
            data = response.json()
            lengths = check_array_lengths(data)
            print("\nArray lengths:")
            for path, length in sorted(lengths.items()):
                print(f"  {path}: {length}")
    
    except Exception as e:
        print(f"Error: {e}")


def test_heatmap():
    """GET /api/heatmap"""
    print("\n" + "="*70)
    print("TEST 3: /api/heatmap")
    print("="*70)
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/heatmap",
            params={
                "metric": "kl_divergence",
                "topN": 10
            }
        )
        print_response_info("/heatmap", response)
        
        if response.status_code == 200:
            data = response.json()
            lengths = check_array_lengths(data)
            print("\nArray lengths:")
            for path, length in sorted(lengths.items()):
                print(f"  {path}: {length}")
    
    except Exception as e:
        print(f"Error: {e}")


def test_cluster_detail():
    """GET /api/cluster/<cluster_id>"""
    print("\n" + "="*70)
    print("TEST 4: /api/cluster/<cluster_id>")
    print("="*70)
    
    try:
        # First get initial data to find a valid cluster ID
        init_response = requests.get(
            f"{API_BASE_URL}/initial_data",
            params={"dataset": "default", "dr_method": "umap"}
        )
        
        if init_response.status_code != 200:
            print("Could not get initial data to find cluster ID")
            return
        
        init_data = init_response.json()
        if not init_data.get("data", {}).get("clusterNames"):
            print("No cluster names found")
            return
        
        cluster_ids = list(init_data["data"]["clusterNames"].keys())
        if not cluster_ids:
            print("No cluster IDs available")
            return
        
        cluster_id = cluster_ids[0]
        response = requests.get(f"{API_BASE_URL}/cluster/{cluster_id}")
        print_response_info(f"/cluster/{cluster_id}", response)
        
        if response.status_code == 200:
            data = response.json()
            lengths = check_array_lengths(data)
            print("\nArray lengths:")
            for path, length in sorted(lengths.items()):
                print(f"  {path}: {length}")
    
    except Exception as e:
        print(f"Error: {e}")


def test_point_detail():
    """GET /api/point/<point_id>"""
    print("\n" + "="*70)
    print("TEST 5: /api/point/<point_id>")
    print("="*70)
    
    try:
        # First get initial data to find a valid point ID
        init_response = requests.get(
            f"{API_BASE_URL}/initial_data",
            params={"dataset": "default", "dr_method": "umap"}
        )
        
        if init_response.status_code != 200:
            print("Could not get initial data to find point ID")
            return
        
        init_data = init_response.json()
        if not init_data.get("data", {}).get("points"):
            print("No points found")
            return
        
        point_id = init_data["data"]["points"][0]["i"]
        response = requests.get(f"{API_BASE_URL}/point/{point_id}")
        print_response_info(f"/point/{point_id}", response)
        
        if response.status_code == 200:
            data = response.json()
            lengths = check_array_lengths(data)
            print("\nArray lengths:")
            for path, length in sorted(lengths.items()):
                print(f"  {path}: {length}")
    
    except Exception as e:
        print(f"Error: {e}")


def test_point_to_cluster():
    """POST /api/point_to_cluster"""
    print("\n" + "="*70)
    print("TEST 6: /api/point_to_cluster")
    print("="*70)
    
    try:
        payload = {
            "pointIds": [0, 1, 2, 3, 4],
            "threshold": 0.5
        }
        response = requests.post(
            f"{API_BASE_URL}/point_to_cluster",
            json=payload
        )
        print_response_info("/point_to_cluster", response)
        
        if response.status_code == 200:
            data = response.json()
            lengths = check_array_lengths(data)
            print("\nArray lengths:")
            for path, length in sorted(lengths.items()):
                print(f"  {path}: {length}")
    
    except Exception as e:
        print(f"Error: {e}")


def test_dendrogram_filter():
    """POST /api/dendrogram_filter"""
    print("\n" + "="*70)
    print("TEST 7: /api/dendrogram_filter")
    print("="*70)
    
    try:
        payload = {
            "strahlerRange": [0, 10],
            "stabilityRange": [0.0, 1.0]
        }
        response = requests.post(
            f"{API_BASE_URL}/dendrogram_filter",
            json=payload
        )
        print_response_info("/dendrogram_filter", response)
        
        if response.status_code == 200:
            data = response.json()
            lengths = check_array_lengths(data)
            print("\nArray lengths:")
            for path, length in sorted(lengths.items()):
                print(f"  {path}: {length}")
    
    except Exception as e:
        print(f"Error: {e}")


def print_summary():
    """テスト結果のサマリーを表示"""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("\nエンドポイント一覧:")
    print("  1. GET    /api/datasets")
    print("  2. GET    /api/initial_data?dataset=default&dr_method=umap")
    print("  3. GET    /api/heatmap?metric=kl_divergence&topN=10")
    print("  4. GET    /api/cluster/<cluster_id>")
    print("  5. GET    /api/point/<point_id>")
    print("  6. POST   /api/point_to_cluster")
    print("  7. POST   /api/dendrogram_filter")
    print("\n確認項目:")
    print("  - ステータスコード (200なら成功)")
    print("  - レスポンスデータ構造")
    print("  - 配列の長さ (points, zMatrix, クラスタ数など)")
    print("  - 各フィールドのデータ型")


def main():
    """メイン実行"""
    print("Backend API Test Script")
    print(f"API Base URL: {API_BASE_URL}")
    print("\nバックエンドが http://localhost:8000 で起動していることを確認してください")
    
    try:
        # 簡単な接続テスト
        response = requests.get(f"{API_BASE_URL}/datasets", timeout=2)
        print(f"✓ API接続成功 (Status: {response.status_code})\n")
    except requests.exceptions.ConnectionError:
        print("✗ API接続失敗")
        print("  バックエンドが起動していることを確認してください")
        print("  起動コマンド: cd src/d3-app && python -m uvicorn src.backend.main_d3:app --reload --port 8000")
        return
    except Exception as e:
        print(f"✗ 接続エラー: {e}")
        return
    
    # テスト実行
    test_datasets()
    test_initial_data()
    test_heatmap()
    test_cluster_detail()
    test_point_detail()
    test_point_to_cluster()
    test_dendrogram_filter()
    
    # サマリー表示
    print_summary()


if __name__ == "__main__":
    main()
