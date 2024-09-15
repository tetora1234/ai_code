import pandas as pd

# CSVファイルを読み込む
file_path = r"C:\Users\user\Desktop\git\ai_code\dataset\whisper\audio\data.csv"
df = pd.read_csv(file_path)

# 内容が空でない行のみを抽出
df_filtered = df.dropna(subset=['内容'])

# 新しいCSVファイルとして保存
output_file_path = r"C:\Users\user\Desktop\git\ai_code\dataset\whisper\audio\data_check.csv"
df_filtered.to_csv(output_file_path, index=False, encoding='utf-8-sig')

print(f"Filtered CSV file saved to {output_file_path}")
