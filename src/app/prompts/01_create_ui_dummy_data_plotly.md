# 01_UIプロトタイプ実装指示とダミーデータ定義

## 1. システム命令とコンテキスト

あなたは、次元削減可視化システム（DimRedVis-Agent）のUIプロトタイプを構築するエージェントです。以下の制約と仕様を厳守し、**バックエンドの計算ロジックを完全に排除**したフロントエンドの静的表示コードを生成してください。

### 1.1 実装環境と制約
* **目的:** UI/UXの設計検証のみ。
* **言語/フレームワーク:** Dash
* **再現性:** 乱数を使用する場合は、必ず `random_state=42` を使用すること。

### 1.2 参照仕様
* **レイアウト/コンポーネント:** `specifications/ui_spec.md` のすべてのセクションを厳守すること。
* **実行トリガー:** `execute-button` が押された際の動作は、コンソールに「実行ボタンが押されました」とログ出力するモックアップとすること。

---

## 2. ダミーデータ定義

以下のデータ構造を生成し、各ビューの表示に使用すること。

### 2.1 DR View用ダミーデータ (DR_DUMMY_DATA)

* **データ構造:** JSON形式のリスト（100点程度）。
* **フィールド:**
    * `x`: 削減後のX座標（ランダムな浮動小数点数）
    * `y`: 削減後のY座標（ランダムな浮動小数点数）
    * `label`: クラスラベル（'A', 'B', 'C'のいずれか）
    * `id`: 元のデータポイントのID（0から99の整数）

```python
import random
random.seed(42) # 再現性確保のためシードを固定

def generate_dr_dummy_data(n=100):
    data = []
    labels = ['A', 'B', 'C']
    for i in range(n):
        data.append({
            'x': random.uniform(-5, 5),
            'y': random.uniform(-5, 5),
            'label': random.choice(labels),
            'id': i
        })
    return data

DR_DUMMY_DATA = generate_dr_dummy_data(100)
```

### 3.2. 参照仕様とファイル操作の制約

* **UI仕様の参照:** `specifications/ui_spec.md` に記載されたレイアウト仕様（A:2, B:4, C:4, D:2）およびすべてのコンポーネントIDを厳守すること。
* **ファイル操作:**
    * **変更禁止:** `app/specifications/ui_spec.md` の内容は**参照のみ**とし、絶対に変更してはならない。
    * **許可:** `src/app/01_protopye_dash` 以下へのディレクトリ、ファイルの生成、およびデバッグに必要な一時的なログ出力ファイルの作成は許可する。