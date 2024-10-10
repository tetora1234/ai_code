import pandas as pd

# CSVファイルのパス
input_file_path = r"C:\Users\user\Downloads\out.csv"
output_file_path = r"C:\Users\user\Downloads\filter_Speaker.csv"

# CSVファイルを読み込む
df = pd.read_csv(input_file_path)

# 特定のスピーカーとFilePathに含まれるタイトルを指定
target_speaker = "ゆめみ"  # 実際のスピーカー名に変更
target_title_keyword = "Aino"  # 正しいキーワードを指定

# 指定したスピーカーとFilePathにタイトルが含まれるエントリを抽出
filtered_data = df[(df["Speaker"] == target_speaker) & (df["FilePath"].str.contains(target_title_keyword))]

# フィルタリング結果を表示
print("\nFiltered Data:")
print(filtered_data)

# 結果を新しいCSVファイルに保存
filtered_data.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"\nFiltered Text data has been saved to {output_file_path}")
