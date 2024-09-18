import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from torch.nn import functional as F

# トークナイザーとモデルのロード（保存されたモデルディレクトリを指定）
model_dir = r"C:\Users\user\Desktop\git\ai_code\TextClassification\models\checkpoint-11160"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()  # 推論モードに切り替える

# テキストの指定 (例: "おはようございます")
text = "あっ、あっ、ああァーッ！！こんなっ、気持ちいいッ！"

# ラベルの逆マッピング
label_inverse_map = {0: "通常", 1: "あえぎ", 2: "チュパ"}

# テキストをトークン化して推論
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# モデル推論
with torch.no_grad():
    outputs = model(**inputs)

# softmaxを適用して予測確率を取得
probs = F.softmax(outputs.logits, dim=-1).numpy()[0]

# 最も高い確率のラベルを予測
predicted_label = np.argmax(probs)
predicted_label_name = label_inverse_map[predicted_label]

# 結果を表示
print(f"テキスト: {text}")
print(f"予測されたラベル: {predicted_label_name}")
print(f"各クラスの予測確率: 通常:{probs[0]:.4f}, あえぎ:{probs[1]:.4f}, チュパ:{probs[2]:.4f}")
