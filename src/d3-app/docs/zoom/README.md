# ズーム機能 - ドキュメント

GPU加速UMAPを使用したズーム機能の実装ドキュメント

## 📖 ドキュメント一覧

### 🚀 はじめに
- **[OVERVIEW.md](OVERVIEW.md)** - プロジェクト概要と現在の状態
- **[QUICK_START.md](QUICK_START.md)** - テスト方法と基本的な使い方

### 🏗️ 技術詳細
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - システムアーキテクチャとデータフロー
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - 実装詳細とコード参照
- **[CODE_CHANGES.md](CODE_CHANGES.md)** - 修正内容の詳細

### 🔧 実装ガイド
- **[FRONTEND_GUIDE.md](FRONTEND_GUIDE.md)** - フロントエンド実装ステップバイステップ
- **[CONFIGURATION.md](CONFIGURATION.md)** - パラメータ設定とチューニング
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - トラブルシューティング

---

## クイックスタート

### テスト実行
```bash
cd src/d3-app/src/backend
uvicorn main_d3:app --port 8000

# 別ターミナルで：
python test_zoom_api.py
```

### APIテスト
```bash
curl -X POST http://localhost:8000/api/zoom/redraw \
  -H "Content-Type: application/json" \
  -d '{"point_ids": [0, 1, 2, 3, 4]}'
```

---

## 実装状況

| 項目 | 状態 |
|------|------|
| バックエンド | ✅ 完成 |
| ドキュメント | ✅ 完成 |
| フロントエンド | ⏳ 実装予定 |

---

## 重要なファイル

- **実装**: `src/d3-app/src/backend/services/d3_data_manager.py`
- **API**: `src/d3-app/src/backend/main_d3.py`
- **テスト**: `src/d3-app/src/backend/test_zoom_api.py`

---

詳細は各ドキュメントを参照してください。
