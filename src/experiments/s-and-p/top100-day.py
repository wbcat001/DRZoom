import pandas as pd

# データ読み込み
stocks = pd.read_csv("src/experiments/s-and-p/sp500_stocks.csv", parse_dates=["Date"])

# 日付ごとに上位100企業を抽出
top_100_per_day = (
    stocks.groupby('Date')
    .apply(lambda df: df.nlargest(100, 'Adj Close')['Symbol'])
    .reset_index(level=0, drop=True)
)

# ユニークな企業数を数える
unique_companies = top_100_per_day.unique()
num_companies = len(unique_companies)

print(f"過去に一度でも100位以内に入った企業の数: {num_companies}")
