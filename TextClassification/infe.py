import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from torch.nn import functional as F
import pandas as pd

# トークナイザーとモデルのロード（保存されたモデルディレクトリを指定）
model_dir = r"C:\Users\user\Desktop\git\ai_code\TextClassification\models\checkpoint-11160"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()  # 推論モードに切り替える

# ラベルの逆マッピング
label_inverse_map = {0: "通常", 1: "あえぎ", 2: "チュパ"}

# CSVファイルを逐次処理するために、逐次書き込みモードでファイルを開く
input_csv_path = r"C:\Users\user\Desktop\check.csv"
output_csv_path = r"C:\Users\user\Desktop\data_with_predictions.csv"  # 書き込み権限のある場所に変更

# 入力CSVを読み込みつつ、出力CSVを初期化
df = pd.read_csv(input_csv_path)

# 必要な列だけを用意
df_output = pd.DataFrame(columns=['transcript', 'predicted_label', 'predicted_probs'])

# 出力ファイルのヘッダーを書き込む
df_output.to_csv(output_csv_path, mode='w', header=True, index=False)

# 各行を逐次処理しながら結果を書き込む
for index, row in df.iterrows():
    text = row['transcript']  # 各テキストを取得
    
    # テキストをトークナイズ
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # モデル推論
    with torch.no_grad():
        outputs = model(**inputs)

    # softmaxを適用して予測確率を取得
    probs = F.softmax(outputs.logits, dim=-1).numpy()[0]

    # 最も高い確率のラベルを予測
    predicted_label = np.argmax(probs)
    predicted_label_name = label_inverse_map[predicted_label]
    
    # 各クラスの確率を文字列として保存
    predicted_probs = f"通常:{probs[0]:.4f}, あえぎ:{probs[1]:.4f}, チュパ:{probs[2]:.4f}"
    
    # 必要な列だけのデータを作成
    result_row = pd.DataFrame({
        'transcript': [text],
        'predicted_label': [predicted_label_name],
        'predicted_probs': [predicted_probs]
    })
    
    # 結果をCSVファイルに逐次書き込む
    result_row.to_csv(output_csv_path, mode='a', header=False, index=False)

    print(f"行 {index+1}/{len(df)} の処理が完了しました")

print(f"全ての推論結果を {output_csv_path} に書き込みました")
