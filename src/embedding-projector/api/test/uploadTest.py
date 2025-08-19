import requests

# テスト用CSVファイルを作成
import pandas as pd
test_df = pd.DataFrame({'x': [1,2,3], 'y': [4,5,6], 'label': ['A','B','C']})

test_df.to_csv('test.csv', index=False)



# ファイルアップロードテスト
with open('test.csv', 'rb') as f:
    files = {'file': ('test.csv', f, 'text/csv')}
    response = requests.post('http://localhost:8000/files/upload', files=files)
print(response.json())

# ファイルリスト取得テスト
response = requests.get('http://localhost:8000/files/list')
print(response.json())