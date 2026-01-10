for f in $(git diff --cached --name-only); do
  # 改行コードを削除してから du を実行
  clean_f=$(echo $f | tr -d '\r')
  du -sh "$clean_f"
done