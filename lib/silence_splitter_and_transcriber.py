import os
import pandas as pd

# モデル名の変数を追加
MODEL_NAME = "slicer_opt"
OUTPUT_DIR_BASE = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\outputs"

# CSVファイルのパス
CSV_PATH = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\sirone\data\chupa_files.csv"  # ここにCSVファイルのパスを設定

def save_transcription_esd_list(csv_data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in csv_data.iterrows():
            f.write(f"{row['FilePath']}|{MODEL_NAME}|JP|{row['Text']}\n")

def main():
    # CSVファイルを読み込む
    csv_data = pd.read_csv(CSV_PATH)

    # esd.listに書き込み
    esd_list_path = os.path.join(OUTPUT_DIR_BASE, "slicer_opt.list")  # outputs直下に保存
    save_transcription_esd_list(csv_data, esd_list_path)

if __name__ == "__main__":
    main()
