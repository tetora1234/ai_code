import os
import json
import re

# エンコーディング候補のリスト
encodings = ['utf-8', 'shift_jis', 'cp932', 'euc-jp']
# 除外するフォルダ名リスト
excluded_folders = ['$RECYCLE.BIN', 'System Volume Information']

def try_open_file(file_path):
    """複数のエンコーディングでファイルを開き、成功したら内容を返す"""
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as file:
                return file.readlines(), enc  # ファイルの内容と使用したエンコーディングを返す
        except UnicodeDecodeError:
            continue  # エンコーディングエラーが発生したら次の候補へ
    # すべてのエンコーディングで失敗した場合、エラーメッセージを表示
    raise Exception(f"Could not decode the file: {file_path} with any of the given encodings.")

def process_txt_file(file_path):
    try:
        # ファイルを開いて内容を取得
        lines, encoding_used = try_open_file(file_path)
        
        # タイトルの取得
        if len(lines) > 1 and lines[0].strip() == "":
            title = lines[1].strip()  # 1行目が空行なら2行目をタイトルとする
        elif lines[0].strip() == "スタジオりふれぼ":
            title = lines[1].strip()  # 1行目が「スタジオりふれぼ」なら2行目をタイトルとする
        else:
            title = lines[0].strip()  # 1行目をタイトルとする

        if title == '『魔法少女 VS 透明怪人':
            print()

        # "トラック"という行の次の行から内容を取得
        content = ""
        track_found = False
        for line in lines:
            if track_found:
                # 【】で囲まれた部分を削除し、残りをcontentに追加
                line = re.sub(r'【.*?】', '', line)  # 【】内の内容を削除
                line = re.sub(r'《.*?》', '', line)  # 【】内の内容を削除
                content += line
            
            if "トラック" in line:  # トラック行が見つかったら、その次の行から収集
                track_found = True

        content = content.strip().replace("\n", " ")  # 不要な空白を削除し、改行をスペースに置換
        content = content.strip().replace("------------------------------------------------------------------------------------------------------------   ", "")  # 不要な空白を削除し、改行をスペースに置換

        if content == '---------------':
            print()

        # ファイルのデータを辞書形式で返す
        return {
            "タイトル": title,
            "内容": content
        }

    except Exception as e:
        print(e)
        return None  # エラーが発生した場合はNoneを返す

def process_directory(root_dir):
    # すべての.txtファイルのデータを格納するリスト
    all_data = []

    # 再帰的にフォルダを探索し、すべての.txtファイルを処理
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 除外リストにあるフォルダはスキップ
        dirnames[:] = [d for d in dirnames if d not in excluded_folders]

        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(dirpath, filename)
                file_data = process_txt_file(file_path)
                if file_data:
                    all_data.append(file_data)  # 成功したファイルデータをリストに追加

    # まとめたデータをroot_directory直下に保存
    output_file_path = os.path.join(root_dir, 'combined_data.json')
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_data, json_file, ensure_ascii=False, indent=4)

    print(f"All data has been saved to {output_file_path}")

# E:ドライブ以下のフォルダを処理
root_directory = r"D:\asmr_txt\スタジオりふれぼ"
process_directory(root_directory)
