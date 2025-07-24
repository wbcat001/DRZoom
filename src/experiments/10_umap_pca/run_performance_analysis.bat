@echo off
echo PCAパフォーマンス最適化の分析を開始します
echo ---------------------------------------

cd %~dp0
python performance_analysis.py

echo.
echo 分析が完了しました！結果を確認してください。
echo pca_optimization_comparison.png が生成されています。
pause
