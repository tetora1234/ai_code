import pandas as pd

# CSVファイルの読み込み
csv_file = r"C:\Users\user\Downloads\data.csv"  # 例: 入力ファイルのパスを指定
df = pd.read_csv(csv_file)

# 単語リストを定義
search_words = ['ロリ', '手コキ']  # 検索したい単語リスト

# 'transcript'列に単語が含まれるかどうかチェック
# 各単語がテキストに含まれている行を抽出
df_filtered = df[df['transcript'].apply(lambda x: any(word in x for word in search_words))]

# 結果を新しいCSVに保存
output_file = r"C:\Users\user\Downloads\out.csv"
df_filtered.to_csv(output_file, index=False)

print(f"抽出したデータを {output_file} に保存しました。")
