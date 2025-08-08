"""
実行時間計測結果をグラフ化して表示するスクリプト
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import argparse


def plot_results(results_file=None, results=None, output_dir='.'):
    """計測結果をグラフ化"""
    if results_file and os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    
    if not results:
        print("結果データがありません。")
        return
    
    # グラフ作成
    plt.figure(figsize=(15, 10))
    
    # 1. 各実装方法の実行時間比較
    plt.subplot(2, 2, 1)
    names = list(results.keys())
    times = [results[name]['total_time'] for name in names]
    plt.bar(names, times)
    plt.title('総実行時間の比較')
    plt.ylabel('時間 (秒)')
    plt.xticks(rotation=45)
    
    # 2. 各実装の平均フレーム時間
    plt.subplot(2, 2, 2)
    avg_times = [results[name]['avg_frame_time'] for name in names]
    plt.bar(names, avg_times)
    plt.title('フレームあたりの平均計算時間')
    plt.ylabel('時間 (秒)')
    plt.xticks(rotation=45)
    
    # 3. 各実装方法の高速化倍率
    plt.subplot(2, 2, 3)
    baseline = results[names[0]]['total_time']  # 最初の実装を基準にする
    speedups = [baseline / results[name]['total_time'] for name in names]
    plt.bar(names, speedups)
    plt.title('高速化倍率')
    plt.ylabel('倍率')
    plt.xticks(rotation=45)
    
    # 4. 各部分の実行時間内訳
    plt.subplot(2, 2, 4)
    components = ['cov_time', 'eig_time', 'other_time']
    data = []
    labels = []
    
    for name in names:
        if 'avg_cov_time' in results[name] and 'avg_eig_time' in results[name]:
            avg_cov = results[name]['avg_cov_time']
            avg_eig = results[name]['avg_eig_time']
            avg_total = results[name]['avg_frame_time']
            avg_other = avg_total - avg_cov - avg_eig
            data.append([avg_cov, avg_eig, avg_other])
            labels.append(name)
    
    data = np.array(data).T
    x = np.arange(len(labels))
    width = 0.8
    
    plt.bar(x, data[0], width, label='共分散行列計算')
    plt.bar(x, data[1], width, bottom=data[0], label='固有値分解')
    plt.bar(x, data[2], width, bottom=data[0]+data[1], label='その他の処理')
    
    plt.title('処理内訳')
    plt.ylabel('時間 (秒)')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    plt.close()
    
    print(f"グラフを {output_dir}/performance_comparison.png に保存しました")


def print_results_table(results_file=None, results=None):
    """計測結果を表形式で表示"""
    if results_file and os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    
    if not results:
        print("結果データがありません。")
        return
    
    names = list(results.keys())
    baseline = results[names[0]]['total_time']
    
    print("\n==== パフォーマンス計測結果 ====")
    print(f"{'実装方法':15s} {'合計時間 (秒)':15s} {'フレーム平均 (秒)':20s} {'高速化倍率':10s}")
    print("-" * 65)
    
    for name in names:
        result = results[name]
        total_time = result['total_time']
        avg_frame = result['avg_frame_time']
        speedup = baseline / total_time if total_time > 0 else 0
        
        print(f"{name:15s} {total_time:15.4f} {avg_frame:20.4f} {speedup:10.2f}倍")
    
    print("\n==== 処理内訳 ====")
    print(f"{'実装方法':15s} {'共分散行列 (秒)':15s} {'固有値分解 (秒)':15s} {'割合 (%)':10s}")
    print("-" * 65)
    
    for name in names:
        result = results[name]
        if 'avg_cov_time' in result and 'avg_eig_time' in result:
            cov_time = result['avg_cov_time']
            eig_time = result['avg_eig_time']
            avg_total = result['avg_frame_time']
            cov_ratio = cov_time / avg_total * 100 if avg_total > 0 else 0
            
            print(f"{name:15s} {cov_time:15.4f} {eig_time:15.4f} {cov_ratio:10.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='実行時間計測結果を表示')
    parser.add_argument('--file', type=str, help='結果JSONファイル')
    parser.add_argument('--output', type=str, default='.', help='出力ディレクトリ')
    args = parser.parse_args()
    
    # 結果表示
    if args.file:
        print_results_table(results_file=args.file)
        plot_results(results_file=args.file, output_dir=args.output)
    else:
        print("使用方法: python view_results.py --file results.json [--output ./output_dir]")
        print("結果ファイルが指定されていません。")
