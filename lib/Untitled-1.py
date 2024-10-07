import pandas as pd
import re

# CSVファイルを読み込む
file_path = r"D:\Galgame_Dataset\data.csv"
df = pd.read_csv(file_path)

# 抽出したい単語のリスト
keywords = ["んくっ", "おほお", "おおお", "はぁっ", "ひう"]  # ここに検索したい複数の単語を指定

# キーワードを正規表現に変換（OR条件で検索）
pattern = '|'.join([re.escape(keyword) for keyword in keywords])

# Text列に複数の単語が含まれている行を抽出
filtered_df = df[df['Text'].str.contains(pattern, na=False, regex=True)]

# 抽出結果をCSVファイルに保存
output_file_path = r"D:\Galgame_Dataset\filtered_data.csv"
filtered_df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"抽出結果が {output_file_path} に保存されました。")
