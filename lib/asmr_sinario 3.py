import os
import json
import re
import chardet

# フォルダのパスを指定
base_dir = r'D:\asmr_txt\防鯖潤滑剤'

# データを格納するリスト
data = []

# テキストファイルを自動でエンコーディングを検出して読み込む関数
def read_file_with_encoding(file_path):
    # バイナリモードでファイルを開き、エンコーディングを検出
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    
    # 検出したエンコーディングでファイルを読み込み（エラーを無視）
    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
        return file.read()

# フォルダ内の全てのフォルダとファイルを再帰的に探索
for root, dirs, files in os.walk(base_dir):
    # フォルダ名を取得
    folder_name = os.path.basename(root)
    
    # 各テキストファイルごとに処理
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(root, file_name)
            
            # テキストファイルを自動でエンコーディングを検出して読み込み
            content = read_file_with_encoding(file_path)
            
            # プロローグ部分を削除
            # ここでは「☆プロローグ」とその後の内容を削除します。
            content = re.sub(r'☆プロローグ.*?(\n\d+)', r'\1', content, flags=re.DOTALL)
            content = re.sub(r'プロローグ.*?(\n\d+)', r'\1', content, flags=re.DOTALL)

            # テキストの正規表現置換処理
            content = re.sub(r'うぃすぷ。?', '', content)
            content = re.sub(r'おち[〇○]ちん', 'おちんちん', content)
            content = re.sub(r'おま[〇○]こ', 'おまんこ', content)
            content = re.sub(r'▼', ' ', content)
            content = re.sub(r'\u200b', ' ', content)  # 特定の無視文字をスペースに置換
            content = re.sub(r'\u3000', ' ', content)  # 全角スペースを通常のスペースに置換
            content = re.sub(r'女子[〇○]生', '女子校生', content)
            content = re.sub(r'\n', ' ', content)  # 改行をスペースに置換
            content = re.sub(r'（.*?）', '', content)  # 丸括弧内を削除
            content = re.sub(r'《.*?》', '', content)  # 鍵括弧内を削除
            content = re.sub(r'【.*?】', '', content)  # 角括弧内を削除
            content = re.sub(r'◆.*?◆', '', content)  # ダイヤモンド記号内を削除
            content = re.sub(r' {2,}', ' ', content)  # 複数スペースを1つに置換
            content = re.sub(r'☆', '', content)  # 複数スペースを1つに置換
            content = re.sub(r'プロローグ', '', content)  # 複数スペースを1つに置換
            content = re.sub(r'(\d+)-\d+', r'\1', content)
            content = re.sub(r'－２', '', content)
            content = re.sub(r'－１', '', content)
            content = re.sub(r'－３', '', content)
            content = re.sub(r'－４', '', content)
            content = re.sub(r'－５', '', content)
            content = re.sub(r'－６', '', content)
            content = re.sub(r'－７', '', content)
            content = re.sub(r'－８', '', content)
            content = re.sub(r'－９', '', content)
            content = re.sub(r'―２', '', content)
            content = re.sub(r'―１', '', content)
            content = re.sub(r'―3', '', content)
            content = re.sub(r'―4', '', content)
            content = re.sub(r'―5', '', content)
            content = re.sub(r'―6', '', content)
            content = re.sub(r'―6', '', content)
            content = re.sub(r'―7', '', content)
            content = re.sub(r'―8', '', content)
            content = re.sub(r'―9', '', content)

            # トラックごとに分割（数字で始まる行で分ける）
            # 正規表現で数字の行で分割
            tracks = re.split(r'(?<!\S)(\d+、?)', content)
            print(folder_name)
            if folder_name == "オナニー実況プレイ～あなたのオナニー実況します!～":
                print()

            # 各トラックをJSON用に整形
            for index in range(0, len(tracks), 2):  # 奇数インデックスはトラックの内容
                track_content = tracks[index].strip()  # トラックの前後の空白を削除
                if track_content:  # 内容が空でない場合のみ追加
                    file_data = {
                        "タイトル": f"{folder_name}",  # トラック名をタイトルに使用
                        "内容": track_content  # トラックの内容を「内容」として格納
                    }
                    # データをリストに追加
                    data.append(file_data)

# JSONファイルに書き込む
json_output_path = os.path.join(base_dir, '防鯖潤滑剤.json')
with open(json_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"JSONファイルが作成されました: {json_output_path}")
