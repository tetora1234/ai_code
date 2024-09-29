import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from torch.nn import functional as F
import pandas as pd

# トークナイザーとモデルのロード（保存されたモデルディレクトリを指定）
model_dir = r"C:\Users\user\Desktop\git\ai_code\TextClassification\models\checkpoint-18504"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()  # 推論モードに切り替える

# CSVファイルを逐次処理するために、逐次書き込みモードでファイルを開く
input_csv_path = r"C:\Users\user\Desktop\git\ai_code\TextClassification\dataset\Filtered_Speakers.csv"
output_csv_path = r"C:\Users\user\Desktop\git\ai_code\TextClassification\dataset\out.csv"  # 書き込み権限のある場所に変更

# 入力CSVを読み込みつつ、出力CSVを初期化
df = pd.read_csv(input_csv_path)

# 必要な列だけを用意
df_output = pd.DataFrame(columns=['FilePath', 'transcript', 'predicted_label', 'predicted_probs'])

# 出力ファイルのヘッダーを書き込む
df_output.to_csv(output_csv_path, mode='w', header=True, index=False)

# ラベルのマッピング（数値ラベルと対応するテキストラベルのマップ）
classification_map = {
    0: "usual",
    1: "aegi",
    2: "chupa"
}

# 各行を逐次処理しながら結果を書き込む
for index, row in df.iterrows():
    text = row['Text']  # 各テキストを取得
    FilePath = row['FilePath']

    # テキストをトークナイズ
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # モデル推論
    with torch.no_grad():
        outputs = model(**inputs)

    # softmaxを適用して予測確率を取得し、それをパーセンテージ形式に変換
    probs = F.softmax(outputs.logits, dim=-1).numpy()[0] * 100  # 100倍してパーセンテージに変換

    # 最も高い確率のラベルを予測
    predicted_label_num = np.argmax(probs)
    
    # 数値ラベルをテキストラベルに変換
    predicted_label = classification_map[predicted_label_num]
    
    # パーセンテージとしての予測確率を文字列に変換
    probs_str = ", ".join([f"{classification_map[i]}: {probs[i]:.2f}%" for i in range(len(probs))])
    
    # 必要な列だけのデータを作成
    result_row = pd.DataFrame({
        'transcript': [text],
        'FilePath': [FilePath],
        'predicted_label': [predicted_label],
        'predicted_probs': [probs_str]  # 各クラスの確率を保存
    })
    
    # 結果をCSVファイルに逐次書き込む
    result_row.to_csv(output_csv_path, mode='a', header=False, index=False)

    print(f"行 {index+1}/{len(df)} の処理が完了しました")

print(f"全ての推論結果を {output_csv_path} に書き込みました")
