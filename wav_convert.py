import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment

OVER_SIZE_LIMIT = 200_000_000
csv.field_size_limit(OVER_SIZE_LIMIT)

def convert_to_wav_16k(input_path, output_path):
    try:
        # FFmpegを使用して16kHzのWAVに変換
        subprocess.run(['ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', '-ar', '16000', output_path, '-y'], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Converted: {input_path} -> {output_path}")
        
        # 元のファイルを削除
        os.remove(input_path)
        print(f"Deleted original file: {input_path}")
        
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False

def process_file(row):
    file_path, content = row
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False

    # 出力ファイル名を生成（同じディレクトリに保存）
    output_path = os.path.splitext(file_path)[0] + '.wav'
    
    return convert_to_wav_16k(file_path, output_path)

def main(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(process_file, row) for row in rows]
        
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    csv_file = r"C:\Users\user\Desktop\git\ai_code\dataset\whisper\audio\data.csv"  # CSVファイルのパスを指定してください
    main(csv_file)