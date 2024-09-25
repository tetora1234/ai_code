import pandas as pd
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from difflib import SequenceMatcher
import os
import librosa
import warnings  # 警告を抑制するために追加
import torch  # GPU対応のために追加
import csv

# 警告を無視
warnings.filterwarnings("ignore")

# デバイスの設定（GPUがあればGPUを使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Whisperモデルを読み込む
model_path = r"C:\Users\user\Desktop\git\ai_code\wisper\models\Visual-novel-whisper"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)  # モデルをGPUに移動

# CSVファイルを読み込む
csv_path = r"C:\Users\user\Downloads\check.csv"
df = pd.read_csv(csv_path)

# 類似度を計算する関数
def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# 新しいCSVファイルとして書き込み
new_csv_path = r"C:\Users\user\Downloads\transcript_with_similarity.csv"

with open(new_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # ヘッダーを書き込む
    writer.writerow(['filename', 'transcript', 'similarity_score'])

    # 音声ファイルに対する推論と比較
    for index, row in df.iterrows():
        # ファイルパスを作成
        audio_file = row['filename']
        
        # 音声ファイルを読み込む（librosaを使用）
        audio, sr = librosa.load(audio_file, sr=16000)  # Whisperは16kHzが推奨されるサンプリングレートです
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)  # 入力をGPUに移動

        # 推論を実行
        predicted_ids = model.generate(inputs)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # CSVのテキストと推論結果を比較して類似度を計算
        csv_text = row['transcript']
        similarity = calculate_similarity(transcription, csv_text)

        # 各行を書き込む
        writer.writerow([audio_file, csv_text, similarity])
        
        print(f"{transcription} #transcription\n{csv_text} #csv_text\nIndex:{index} 類似度:{similarity}\n")

print(f"類似度スコア付きのCSVファイルを保存しました: {new_csv_path}")