#!/bin/bash

# JSONファイルのパス
# Adjust the JSON file path to use Unix-style paths for compatibility
JSON_FILE="/mnt/c/Users/acero/Work_Research/DRZoom/data/arxiv/metadata.json"

# ファイルが存在するか確認
if [ ! -f "$JSON_FILE" ]; then
  echo "Error: File $JSON_FILE does not exist."
  exit 1
fi

# データの行数を取得
LINE_COUNT=$(wc -l < "$JSON_FILE")
echo "Number of records in the JSON file: $LINE_COUNT"

# jqを使ってJSONの基本情報を取得
if command -v jq &> /dev/null; then
  echo "--- Sample Data ---"
  jq '.[0:5]' "$JSON_FILE"

  echo "--- Keys in the JSON ---"
  jq 'keys' "$JSON_FILE" | head -n 10
else
  echo "jq is not installed. Install jq to view JSON structure."
fi