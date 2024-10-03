import os
import re

def clean_filename(filename):
    # 拡張子を取得
    name, ext = os.path.splitext(filename)
    
    # 先頭の "track_" とそれに続く数字やアンダースコアを削除
    cleaned_name = re.sub(r'^track_[\d_]*', '', name, flags=re.IGNORECASE)
    
    # 数字 + ピリオド + スペースまたは数字 + ピリオドのパターンを削除
    cleaned_name = re.sub(r'^\d+\.\s*', '', cleaned_name)
    cleaned_name = re.sub(r'\d+\.\s*', '', cleaned_name)
    cleaned_name = re.sub(r' なし', '', cleaned_name)
    cleaned_name = re.sub(r'無し', '', cleaned_name)
    cleaned_name = re.sub(r'効果音なし', '', cleaned_name)
    cleaned_name = re.sub(r'差分', '', cleaned_name)
    cleaned_name = re.sub(r'トラック', '', cleaned_name)
    cleaned_name = re.sub(r'章', '', cleaned_name)
    cleaned_name = re.sub(r'時', '', cleaned_name)
    cleaned_name = re.sub(r'声のみ', '', cleaned_name)
    cleaned_name = re.sub(r'なし', '', cleaned_name)
    cleaned_name = re.sub(r'日目', '', cleaned_name)
    cleaned_name = re.sub(r'反転', '', cleaned_name)
    cleaned_name = re.sub(r'効果音なし', '', cleaned_name)
    cleaned_name = re.sub(r'限目', '', cleaned_name)
    cleaned_name = re.sub(r'まこ', 'まんこ', cleaned_name)
    cleaned_name = re.sub(r'おんこ', 'おまんこ', cleaned_name)
    cleaned_name = re.sub(r'版', '', cleaned_name)
    cleaned_name = re.sub(r'年月日', '', cleaned_name)
    cleaned_name = re.sub(r'効果音あり', '', cleaned_name)
    cleaned_name = re.sub(r'射精音', '', cleaned_name)
    cleaned_name = re.sub(r' ', '', cleaned_name)
    cleaned_name = re.sub(r'第', '', cleaned_name)

    # アルファベットとその他の記号、アンダースコアを削除
    cleaned_name = re.sub(r'[a-zA-Z]', '', cleaned_name)  # アルファベットを削除
    cleaned_name = re.sub(r'[^\w\s]', '', cleaned_name)   # 記号を削除
    cleaned_name = re.sub(r'_', '', cleaned_name)         # アンダースコアを削除
    cleaned_name = re.sub(r'\d+', '', cleaned_name)       # 数字を削除
    
    # 前後の空白を削除
    cleaned_name = cleaned_name.strip()
    
    # 連続する空白を1つに置換
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name)
    
    # ファイル名が空になった場合の処理
    if not cleaned_name:
        cleaned_name = "unnamed_file"
    
    return cleaned_name + ext

def rename_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.mp3', '.wav', '.flac', '.aac')):  # 音声ファイルの拡張子を指定
                old_path = os.path.join(root, filename)
                new_filename = clean_filename(filename)
                new_path = os.path.join(root, new_filename)
                
                if old_path.lower() != new_path.lower():
                    try:
                        os.rename(old_path, new_path)
                        print(f'Renamed: {old_path} -> {new_path}')
                    except Exception as e:
                        print(f'Error renaming {old_path}: {str(e)}')
                else:
                    print(f'No change: {old_path}')

# 使用例
folder_path = r"E:\asmr"
rename_files_in_folder(folder_path)
