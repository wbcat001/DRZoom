import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from umap import UMAP
import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 実験結果の保存先
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

# ディレクトリが存在しない場合は作成
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

class DimensionReductionExperiment:
    """
    各種次元削減アルゴリズムを比較するための実験クラス
    """
    def __init__(self, data_path, output_dim=2, random_state=42):
        """
        初期化
        
        Args:
            data_path (str): 高次元データのパス (.pkl または .csv)
            output_dim (int): 出力次元数（通常は2または3）
            random_state (int): 乱数シード
        """
        self.data_path = data_path
        self.output_dim = output_dim
        self.random_state = random_state
        self.data = None
        self.results = {}
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 実験ID（タイムスタンプ）を使用してスナップショットディレクトリを作成
        self.experiment_dir = os.path.join(SNAPSHOTS_DIR, self.timestamp)
        os.makedirs(self.experiment_dir, exist_ok=True)

    def load_data(self):
        """データの読み込み"""
        print(f"Loading data from {self.data_path}")
        if self.data_path.endswith('.pkl'):
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        elif self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path).values
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
        
        print(f"Data loaded with shape: {self.data.shape}")
        
        # データの前処理（必要に応じて）
        self.preprocessed_data = StandardScaler().fit_transform(self.data)
        
        return self

    def run_algorithm(self, name, algorithm):
        """
        指定されたアルゴリズムを実行し、結果と実行時間を記録
        
        Args:
            name (str): アルゴリズム名
            algorithm: 次元削減アルゴリズムのインスタンス
        """
        print(f"Running {name}...")
        start_time = time.time()
        try:
            reduced_data = algorithm.fit_transform(self.preprocessed_data)
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.results[name] = {
                'data': reduced_data,
                'time': execution_time,
                'success': True
            }
            print(f"  Completed in {execution_time:.2f} seconds")
            
        except Exception as e:
            self.results[name] = {
                'data': None,
                'time': None,
                'success': False,
                'error': str(e)
            }
            print(f"  Failed: {str(e)}")
    
    def run_all_algorithms(self):
        """すべての次元削減アルゴリズムを実行"""
        if self.data is None:
            self.load_data()
            
        algorithms = {
            'PCA': PCA(n_components=self.output_dim, random_state=self.random_state),
            'KernelPCA_rbf': KernelPCA(n_components=self.output_dim, kernel='rbf', random_state=self.random_state),
            'KernelPCA_poly': KernelPCA(n_components=self.output_dim, kernel='poly', random_state=self.random_state),
            'TruncatedSVD': TruncatedSVD(n_components=self.output_dim, random_state=self.random_state),
            't-SNE': TSNE(n_components=self.output_dim, random_state=self.random_state),
            'MDS': MDS(n_components=self.output_dim, random_state=self.random_state),
            'Isomap': Isomap(n_components=self.output_dim),
            'LLE': LocallyLinearEmbedding(n_components=self.output_dim, random_state=self.random_state),
            'UMAP': UMAP(n_components=self.output_dim, random_state=self.random_state)
        }
        
        for name, algorithm in algorithms.items():
            self.run_algorithm(name, algorithm)
            
        return self
    
    def evaluate_results(self):
        """実行結果の評価"""
        # 次元削減の質を評価するメトリック（例: ストレス関数、トラストワーシネスなど）
        print("\nEvaluating results...")
        
        if len(self.results) == 0:
            print("No results to evaluate. Run algorithms first.")
            return self
        
        # 高次元空間での距離と低次元空間での距離の関係を評価
        high_dim_distances = pairwise_distances(self.preprocessed_data)
        
        for name, result in self.results.items():
            if not result['success']:
                continue
                
            reduced_data = result['data']
            low_dim_distances = pairwise_distances(reduced_data)
            
            # 相関係数を計算
            corr = np.corrcoef(high_dim_distances.flatten(), low_dim_distances.flatten())[0, 1]
            self.results[name]['correlation'] = corr
            print(f"  {name}: Distance correlation = {corr:.4f}")
            
        return self
    
    def visualize_results(self):
        """結果の可視化"""
        if len(self.results) == 0:
            print("No results to visualize. Run algorithms first.")
            return self
            
        successful_algorithms = [name for name, result in self.results.items() if result['success']]
        
        # 2次元データの場合の散布図
        if self.output_dim == 2:
            n_cols = min(3, len(successful_algorithms))
            n_rows = (len(successful_algorithms) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(5*n_cols, 4*n_rows))
            
            for i, name in enumerate(successful_algorithms, 1):
                result = self.results[name]
                plt.subplot(n_rows, n_cols, i)
                plt.scatter(result['data'][:, 0], result['data'][:, 1], alpha=0.5)
                plt.title(f"{name}\nTime: {result['time']:.2f}s")
                plt.grid(True)
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_dir, f"comparison_scatter_{self.timestamp}.png"))
            plt.close()
            
        # 実行時間のバープロット
        plt.figure(figsize=(10, 6))
        algorithms = []
        times = []
        
        for name, result in self.results.items():
            if result['success']:
                algorithms.append(name)
                times.append(result['time'])
                
        y_pos = np.arange(len(algorithms))
        plt.barh(y_pos, times)
        plt.yticks(y_pos, algorithms)
        plt.xlabel('Execution time (seconds)')
        plt.title('Algorithm Performance Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, f"time_comparison_{self.timestamp}.png"))
        plt.close()
        
        print(f"\nVisualizations saved to {self.experiment_dir}")
        return self
        
    def save_results(self):
        """結果の保存"""
        results_path = os.path.join(self.experiment_dir, f"results_{self.timestamp}.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
            
        # 実験メタデータを保存
        metadata = {
            'timestamp': self.timestamp,
            'data_path': self.data_path,
            'data_shape': self.data.shape,
            'output_dim': self.output_dim,
            'algorithms': list(self.results.keys())
        }
        
        with open(os.path.join(self.experiment_dir, f"metadata_{self.timestamp}.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"Results saved to {results_path}")
        return self
        
    def generate_report(self):
        """実験レポートをマークダウン形式で生成"""
        report_path = os.path.join(self.experiment_dir, f"report_{self.timestamp}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 次元削減アルゴリズム比較実験レポート\n\n")
            f.write(f"## 実験情報\n\n")
            f.write(f"- 実験日時: {self.timestamp}\n")
            f.write(f"- データパス: `{self.data_path}`\n")
            f.write(f"- データ形状: {self.data.shape}\n")
            f.write(f"- 出力次元: {self.output_dim}\n\n")
            
            f.write(f"## 実行結果\n\n")
            f.write("| アルゴリズム | 成功 | 実行時間(秒) | 距離相関 |\n")
            f.write("| --- | --- | --- | --- |\n")
            
            for name, result in self.results.items():
                success = "✓" if result['success'] else "✗"
                time_str = f"{result['time']:.2f}" if result['success'] else "-"
                corr_str = f"{result.get('correlation', '-'):.4f}" if result.get('correlation') is not None else "-"
                f.write(f"| {name} | {success} | {time_str} | {corr_str} |\n")
                
            f.write("\n## 可視化\n\n")
            f.write(f"### 散布図比較\n\n")
            f.write(f"![散布図比較](comparison_scatter_{self.timestamp}.png)\n\n")
            f.write(f"### 実行時間比較\n\n")
            f.write(f"![実行時間比較](time_comparison_{self.timestamp}.png)\n\n")
            
            if any(not result['success'] for result in self.results.values()):
                f.write("\n## エラー\n\n")
                for name, result in self.results.items():
                    if not result['success']:
                        f.write(f"### {name}\n\n")
                        f.write(f"```\n{result['error']}\n```\n\n")
                        
        print(f"Report generated at {report_path}")
        return self

def run_experiment(data_path, output_dim=2):
    """実験を実行する関数"""
    experiment = DimensionReductionExperiment(data_path, output_dim)
    (experiment
        .load_data()
        .run_all_algorithms()
        .evaluate_results()
        .visualize_results()
        .save_results()
        .generate_report()
    )
    return experiment

if __name__ == "__main__":
    # 使用例: Harry Potterデータセットを使用した次元削減比較
    data_path = "../../data/text/harrypotter1/paragraph_embedding.pkl"
    experiment = run_experiment(data_path)