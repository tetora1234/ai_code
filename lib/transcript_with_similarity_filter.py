import pandas as pd

# フィルタリングしたいCSVファイルのパス
csv_path = r"D:\out.csv"

# CSVを読み込む
df = pd.read_csv(csv_path)

# similarity_score以下の行だけをフィルタリング
threshold = 0.90
filtered_df = df[df['similarity_score'] <= threshold]

# 新しいCSVファイルとして保存
filtered_csv_path = r"D:\filter.csv"
filtered_df.to_csv(filtered_csv_path, index=False)

print(f"フィルタリングされたCSVファイルを保存しました: {filtered_csv_path}")
