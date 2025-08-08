@echo off
echo 最適化されたPCA実装のテストを開始します
echo -------------------------------------

cd %~dp0
python test_scripts\test_optimized_pca.py

echo.
echo テスト完了！
pause
