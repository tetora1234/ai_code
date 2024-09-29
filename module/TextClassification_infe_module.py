import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from torch.nn import functional as F

# トークナイザーとモデルのロード（保存されたモデルディレクトリを指定）
model_dir = r"C:\Users\user\Desktop\git\ai_code\TextClassification\models\checkpoint-23130"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()  # 推論モードに切り替える

# ラベルの逆マッピング
label_inverse_map = {0: "usual", 1: "aegi", 2: "chupa"}

def classify_text(text):
    """
    テキストを分類し、ラベルを返す関数。
    
    Args:
        text (str): 分類するテキスト。
        
    Returns:
        str: 予測されたラベル名。
        dict: 各クラスの予測確率。
    """
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

    # 結果を返す
    return predicted_label_name, {
        "usual": probs[0],
        "aegi": probs[1],
        "chupa": probs[2]
    }
