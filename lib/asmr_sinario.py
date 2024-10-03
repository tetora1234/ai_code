import json
import re
import os

# テキストファイルが入っているフォルダのパス
input_folder = r"D:\asmr_txt\whisp"

# 出力するjsonファイルのパス
output_path = r"D:\asmr_txt\whisp_data.json"

# 「」内のセリフを抽出するパターン
quote_pattern = re.compile(r'「(.*?)」')

# トラックごとに分割するパターン
track_pattern = re.compile(r'(トラック\d+：[\s\S]*?)(?=トラック\d+：|$)')

# （）内のテキストを抽出して削除するパターン
paren_pattern = re.compile(r'（.*?）')

# エンコーディングの候補をリスト化
encodings = ['utf-8', 'shift_jis', 'cp932', 'euc-jp']

# 出力先のディレクトリが存在しない場合は作成
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 全データを格納するリスト
all_data = []

# 指定されたフォルダ内の全てのテキストファイルを処理
for file_name in os.listdir(input_folder):
    if file_name.endswith(".txt"):  # .txtファイルのみを対象にする
        file_path = os.path.join(input_folder, file_name)

        # エンコーディングを試す
        for encoding in encodings:
            try:
                # テキストファイルを指定したエンコーディングで読み込み
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                print(f"ファイル '{file_name}' は {encoding} エンコーディングで正常に読み込まれました。")
                break  # 成功したらループを抜ける
            except UnicodeDecodeError:
                print(f"{encoding} でのデコードに失敗しました。")
                continue

        # トラックごとに分割
        tracks = track_pattern.findall(text)

        # ファイル名から拡張子を除いたタイトルを取得
        title = os.path.splitext(file_name)[0]

        # 各トラックごとに処理
        for track in tracks:
            # 「」内のセリフを全て抽出し、結合
            quotes = quote_pattern.findall(track)
            joined_quotes = ''.join(quotes)

            # （）内のテキストを削除
            cleaned_quotes = paren_pattern.sub('', joined_quotes)
            cleaned_quotes = re.sub(r'うぃすぷ。', '', cleaned_quotes)
            cleaned_quotes = re.sub(r'うぃすぷ', '', cleaned_quotes)
            cleaned_quotes = re.sub(r'おち〇ちん', 'おちんちん', cleaned_quotes)
            cleaned_quotes = re.sub(r'おち○ちん', 'おちんちん', cleaned_quotes)
            cleaned_quotes = re.sub(r'おま〇こ', 'おまんこ', cleaned_quotes)
            cleaned_quotes = re.sub(r'おま○こ', 'おまんこ', cleaned_quotes)
            cleaned_quotes = re.sub(r'▼', ' ', cleaned_quotes)
            cleaned_quotes = re.sub(r'\u200b', ' ', cleaned_quotes)
            cleaned_quotes = re.sub(r'\u3000', ' ', cleaned_quotes)
            cleaned_quotes = re.sub(r'女子〇生', '女子校生', cleaned_quotes)

            # 2000文字以下のトラックはスキップ
            if len(cleaned_quotes) <= 2000:
                continue

            # データを辞書形式にまとめる
            data = {
                "タイトル": title,  # ファイル名をタイトルとして使用
                "内容": cleaned_quotes
            }

            # データをリストに追加
            all_data.append(data)

# すべてのデータをJSON形式で出力
with open(output_path, 'w', encoding='utf-8') as json_file:
    json.dump(all_data, json_file, ensure_ascii=False, indent=4)

print(f"データが {output_path} に保存されました。")
