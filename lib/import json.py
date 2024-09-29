import json
import re
import os

# テキストファイルが入っているフォルダのパス
input_folder = r"E:\data"

# 出力するjsonlファイルのパス
output_path = r"E:\output\data.jsonl"

# 「」内のセリフを抽出するパターン
quote_pattern = re.compile(r'「(.*?)」')

# トラックごとに分割するパターン
track_pattern = re.compile(r'(トラック\d+：[\s\S]*?)(?=トラック\d+：|$)')

# エンコーディングの候補をリスト化
encodings = ['utf-8', 'shift_jis', 'cp932', 'euc-jp']

# 出力先のディレクトリが存在しない場合は作成
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# jsonlファイルに書き込む（追記モードで開く）
with open(output_path, 'w', encoding='utf-8') as jsonl_file:
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
                
                # 2000文字以下のトラックはスキップ
                if len(joined_quotes) <= 2000:
                    continue
                
                # データを辞書形式にまとめる
                data = {
                    "タイトル": title,  # ファイル名をタイトルとして使用
                    "内容": joined_quotes
                }

                # jsonl形式でファイルに書き込み
                json.dump(data, jsonl_file, ensure_ascii=False)
                jsonl_file.write('\n')  # 次のjsonオブジェクトのために改行
