/backend2のコードについて

# 背景
研究の中でインタラクティブなデータ可視化を行うようなアプリケーションを開発する際に、実装を進めやすくなるように整理しつつ開発を行いました。
アプリケーション自体の話というよりは、インタラクティブな可視化アプリケーションを作る際のアーキテクチャについての考えたこととその実装を書いています。

- 可視化システムの概要
高次元の時系列データの次元削減による可視化においてズーム動作を行うようなアプリケーションのモック

# (使用技術)
Fast API, D3.js, numpy, pandas, sklearn, scipy

次元削減を含むデータ処理を高速に行うライブラリがあるpythonを使いたくてこの選択になっています。そこまでのこだわりは今回はないです...

# 要件の整理
データをある手法を使って処理することで可視化し、それに対してユーザーがインタラクションを与えることでレイアウトの更新を行うようなアプリケーションを考えます。

### クライアント
1. 初期レイアウトの情報を受け取り描画を行う
2. ユーザーのインタラクションを受け取る(今回はzoom)
3. 新しいレイアウトの情報を受け取り、描画を行う:  この際にユーザーが認知的負荷を減らすためにアニメーションなどを加える
4. 手法などを切り替える設定UI

### バックエンド

1. データの読み込み
2. レイアウト計算方法の設定
3. 初期レイアウトの計算
4. リクエスト(インタラクション)に応じたレイアウト計算
5. レイアウトの補正やアライメント
6. 設定の更新: データ処理手法の切り替え


### バックエンドの詳細な要件

- 処理方法などの設定をユーザーが切り替えたい
- データ処理の手法を追加したい
- 複数のデータ処理を組み合わせたい
- アライメントのために一つ前のレイアウトを保存しておきたい
- 複数ユーザーがアクセスし、それぞれが状態をもつ


主にこれらの要件について、どういう形を取った実装がいいか考えていく

# 実装
## 可視化システムにおける要件に対するモジュールの分割

可視化システムの要件を満たすために、以下のようにモジュールを分割しました。この分割により、柔軟性と拡張性を確保しつつ、各要件に対応する実装を行いました。

### 1. データ処理モジュール
データ処理に関する要件を満たすために、以下のモジュールを設計しました。

- **基底クラス (BaseProcessor)**: データ処理手法を統一的に扱うための抽象クラス。
- **具体的な処理クラス**: PCAやその他の次元削減手法を実装するクラス。
- **パイプラインモジュール**: 複数の処理を組み合わせて実行する仕組み。

これにより、データ処理手法の追加や切り替えが容易になりました。

### 2. セッション管理モジュール
複数ユーザーがアクセスする際に、それぞれの状態を保持するための仕組みを提供します。

- **セッション管理クラス (SessionManager)**: 各セッションごとに状態を管理し、前回のレイアウト情報などを保持。

これにより、ユーザーごとに独立した状態を維持しながら、インタラクションに応じた動的なレイアウト更新が可能になりました。

### 3. 設定管理モジュール
ユーザーが処理手法や設定を動的に変更できるようにするための仕組みを提供します。

- **設定管理クラス (ConfigManager)**: 処理手法やその他の設定を動的に更新・取得する機能を提供。

これにより、ユーザーの要求に応じた柔軟な設定変更が可能になりました。

### 4. レイアウト計算モジュール
インタラクションに応じたレイアウト計算を行うためのモジュールです。

- **レイアウト計算関数**: 前回のレイアウト情報を考慮しながら、新しいレイアウトを計算。
- **補正・アライメント機能**: レイアウトの整合性を保つための補正処理。

これにより、ユーザーが与えるインタラクションに対してスムーズな応答を実現しました。

### 5. クライアントとの通信モジュール
クライアントとバックエンド間のデータ通信を管理するモジュールです。

- **APIエンドポイント**: 初期レイアウトの送信、インタラクションの受信、新しいレイアウトの送信を行う。
- **リアルタイム通信**: 必要に応じてWebSocketなどを利用してリアルタイム性を確保。

これにより、クライアントとバックエンド間の効率的なデータのやり取りを実現しました。

---

これらのモジュール分割により、可視化システムの要件を満たしつつ、拡張性と保守性を高めることができました。

## データ処理部分
### 1. データ処理の柔軟性を確保するための設計
データ処理手法を簡単に追加・切り替えできるように、処理をモジュール化しました。以下は、データ処理の基底クラスと具体的な処理クラスの例です。

```python
# base_processor.py
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    @abstractmethod
    def process(self, data):
        pass
```

```python
# pca_processor.py
import numpy as np
from sklearn.decomposition import PCA
from base_processor import BaseProcessor

class PCAProcessor(BaseProcessor):
    def __init__(self, n_components=2):
        self.pca = PCA(n_components=n_components)

    def process(self, data):
        return self.pca.fit_transform(data)
```

### 2. 処理の組み合わせを可能にするパイプライン
複数のデータ処理を組み合わせるために、パイプラインを実装しました。

```python
# pipeline.py
class DataPipeline:
    def __init__(self, processors):
        self.processors = processors

    def run(self, data):
        for processor in self.processors:
            data = processor.process(data)
        return data
```

使用例:
```python
from pca_processor import PCAProcessor
from some_other_processor import SomeOtherProcessor

pipeline = DataPipeline([
    PCAProcessor(n_components=3),
    SomeOtherProcessor()
])

processed_data = pipeline.run(raw_data)
```

### 3. 状態管理の工夫
複数ユーザーがアクセスする場合に、それぞれの状態を保持するため、セッションごとに状態を管理する仕組みを導入しました。

```python
# session_manager.py
class SessionManager:
    def __init__(self):
        self.sessions = {}

    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {"previous_layout": None}
        return self.sessions[session_id]

    def update_session(self, session_id, key, value):
        if session_id in self.sessions:
            self.sessions[session_id][key] = value
```

バックエンドでの使用例:
```python
from session_manager import SessionManager

session_manager = SessionManager()

def handle_request(session_id, new_data):
    session = session_manager.get_session(session_id)
    previous_layout = session.get("previous_layout")
    # 新しいレイアウトを計算
    new_layout = calculate_layout(new_data, previous_layout)
    session_manager.update_session(session_id, "previous_layout", new_layout)
    return new_layout
```

### 4. 設定の動的更新
ユーザーが処理手法を切り替えられるように、設定を動的に変更できる仕組みを追加しました。

```python
# config_manager.py
class ConfigManager:
    def __init__(self):
        self.config = {}

    def update_config(self, key, value):
        self.config[key] = value

    def get_config(self, key, default=None):
        return self.config.get(key, default)
```

バックエンドでの使用例:
```python
from config_manager import ConfigManager

config_manager = ConfigManager()

def update_processing_method(method_name):
    config_manager.update_config("processing_method", method_name)

def get_processing_method():
    return config_manager.get_config("processing_method", "default_method")
```

これらの工夫により、柔軟性と拡張性を持つデータ処理基盤を構築しました。



