@echo off
echo UMAP+PCA 分析ダッシュボード を起動します...
echo.

REM 仮想環境がある場合はそれを有効化（オプション）
REM call .venv\Scripts\activate

REM 必要なパッケージがインストールされているか確認
echo 依存パッケージをインストールしています...
pip install -r requirements.txt

echo.
echo アプリケーションを起動します...
echo ブラウザで http://127.0.0.1:8050/ を開いてダッシュボードにアクセスしてください。
echo.

python app.py

REM エラー時に一時停止
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo エラーが発生しました。
  pause
)
