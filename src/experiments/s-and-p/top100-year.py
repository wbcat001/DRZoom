import pandas as pd

# データ読み込みと準備
stocks = pd.read_csv("src/experiments/s-and-p/sp500_stocks.csv", parse_dates=["Date"])

# 月単位に変換
stocks["Month"] = stocks["Date"].dt.to_period("M")

# 各企業・各月ごとの平均株価を計算（もしくは最終日Closeを使いたい場合も可）
monthly_mean = stocks.groupby(["Month", "Symbol"])["Adj Close"].mean().reset_index()

# 各月で上位100社を選出
top100_each_month = (
    monthly_mean
    .sort_values(["Month", "Adj Close"], ascending=[True, False])
    .groupby("Month")
    .head(100)
)

# 一度でもランクインした企業をリストアップ（重複なし）
unique_top100_symbols = top100_each_month["Symbol"].unique()

# 結果表示
print(f"トップ100に一度でも入った企業数: {len(unique_top100_symbols)}")
print(unique_top100_symbols)
