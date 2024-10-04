import pandas as pd

# CSVファイルのパス
input_file_path = r"D:\Galgame_Dataset\data.csv"
output_file_path = r"D:\Galgame_Dataset\filter.csv"

# 特定のスピーカーを指定
target_speaker = "シロネ"

# CSVファイルを読み込む
df = pd.read_csv(input_file_path)

# 指定したスピーカーのエントリを抽出
filtered_data = df[df["Speaker"] == target_speaker]

# 結果を新しいCSVファイルに保存
filtered_data.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Filtered data has been saved to {output_file_path}")
