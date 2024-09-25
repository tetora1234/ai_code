import os
import pandas as pd

# CSVファイルのパス
csv_path = r"C:\Users\user\Desktop\git\ai_code\Rgression\check.csv"
output_csv_path = r"C:\Users\user\Desktop\git\ai_code\Rgression\filtered_check.csv"

# CSVファイルを読み込む
df = pd.read_csv(csv_path)

# ファイルが存在し、transcriptが空でない行をフィルタする
filtered_df = df[df['filepath'].apply(os.path.exists) & df['transcript'].notna()]

# filepathとtranscriptの列のみを保持する
filtered_df = filtered_df[['filepath', 'transcript']]

# フィルタしたデータを新しいCSVファイルとして保存
filtered_df.to_csv(output_csv_path, index=False)

print(f"Filtered CSV saved to {output_csv_path}")
