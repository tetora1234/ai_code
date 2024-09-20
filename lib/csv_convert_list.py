import csv

# 元のCSVファイルパスと出力ファイルパス
input_file = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\dataset\aegi_files.csv"
output_file = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\dataset\slicer_opt.list"

# 出力形式に変換する関数
def convert_csv(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        for row in reader:
            # ファイルパス、分類、テキストを出力形式に合わせる
            audio_path = row['FilePath']
            text = row['Text']
            output_line = f"{audio_path}|slicer_opt|JP|{text}\n"
            outfile.write(output_line)

# 変換実行
convert_csv(input_file, output_file)
print(f"変換が完了しました: {output_file}")
