import os
import pandas as pd

# CSVファイルのパス
csv_path = r"D:\transcript.csv"
output_csv_path = r"D:\check.csv"

# CSVファイルを読み込む
df = pd.read_csv(csv_path)

# ファイルパスに"D:\dataset"を先頭に追加
df['full_filepath'] = df['filename'].apply(lambda x: os.path.join(r"D:\dataset", x))

# ファイルが存在し、transcriptが空でない行をフィルタする
filtered_df = df[df['full_filepath'].apply(os.path.exists) & df['transcript'].notna()]

# full_filepathとtranscriptの列のみを保持する
filtered_df = filtered_df[['full_filepath', 'transcript']]

# フィルタしたデータを新しいCSVファイルとして保存
filtered_df.to_csv(output_csv_path, index=False)

print(f"Filtered CSV saved to {output_csv_path}")
