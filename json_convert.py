import json
import csv
import os
from pydub import AudioSegment
import concurrent.futures

def convert_audio(input_path):
    try:
        print(f"変換中: {input_path}")
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000)
        output_path = os.path.splitext(input_path)[0] + ".wav"
        audio.export(output_path, format="wav")
        os.remove(input_path)  # 元のファイルを削除
        print(f"変換完了: {output_path}")
        return output_path
    except Exception as e:
        print(f"変換エラー {input_path}: {e}")
        return None

def process_json_files(directory):
    csv_data = []
    audio_tasks = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == "index.json":
                    file_path = os.path.join(root, file)
                    folder_name = os.path.basename(root)
                    print(f"処理中のフォルダ: {folder_name}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            input_path = os.path.abspath(os.path.join(directory, folder_name, item['FilePath']))
                            
                            # 変換タスクを追加
                            task = executor.submit(convert_audio, input_path)
                            audio_tasks.append((task, item['Text']))

        # すべての変換タスクが完了するのを待つ
        for i, (task, text) in enumerate(audio_tasks, 1):
            result = task.result()
            if result:
                csv_data.append([result.replace('\\', '/'), text])
            print(f"進捗: {i}/{len(audio_tasks)}")
    
    return csv_data

def write_csv(data, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['FilePath', 'Text'])  # ヘッダー行を書き込む
        writer.writerows(data)
    print(f"CSVファイルが作成されました: {output_file}")

# メイン処理
directory = r"C:\Users\user\Desktop\git\ai_code\dataset\whisper\audio"  # 指定されたWindowsの絶対パス
output_file = os.path.join(directory, 'data.csv')

print("処理を開始します...")
csv_data = process_json_files(directory)
write_csv(csv_data, output_file)

print("すべての処理が完了しました。")