# データセットについて
データセットに関する情報や、扱うデータの形式について

##　 Original data
高次元のデータセットを扱う
例として
- Word2Vec: 300次元、データ数30000000、単語のラベル
- MNIST: 624次元、データ数70000、数字のラベル

データ数に関してはここからサンプリングして扱う場合もある

## Derived Data

### UMAP
次元削減を行うことで2次元の座標になる
(x, y)

### HDBSCAN
密度ベースのクラスタリングを行うことで、データポイントをクラスタに分類できる。また、クラスタ間の階層関係を抽出できる。
クラスタごとにクラスタの安定性を示すStabilityを定義することができる
- condensed tree
クラスタ間の階層関係
(child cluster1, child cluster2, parent cluster, distance, cluster size, stalibity score)

- single linakge tree
- minimum spanning tree

### label
- Word2Vec
HDBSCANを使って抽出したクラスタを要約するラベルを作成している
(cluster idx) -> (summary word)


