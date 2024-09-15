import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
from torch.nn import functional as F

# データセットの読み込み
df = pd.read_csv('dataset.csv')  # CSVファイルのパス

# ラベルをエンコードする
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # "normal" -> 0, "aegi" -> 1, "chupa" -> 2

# データセットを訓練データとテストデータに分割
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2
)

# トークナイザーの初期化（BERTベースのモデル）
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# トークン化
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

# データセットをPyTorchのテンソルに変換
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

# モデルの定義（BERTの分類用ヘッド付きモデル）
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3つのラベル

# トレーニングの設定
training_args = TrainingArguments(
    output_dir='./results',          # 出力ディレクトリ
    num_train_epochs=3,              # エポック数
    per_device_train_batch_size=16,  # バッチサイズ
    per_device_eval_batch_size=16,   # 評価バッチサイズ
    warmup_steps=500,                # ウォームアップステップ
    weight_decay=0.01,               # 重みの減衰率
    logging_dir='./logs',            # ログディレクトリ
    logging_steps=10,
    evaluation_strategy="steps",     # 評価の頻度
)

# トレーナーの初期化
trainer = Trainer(
    model=model,                         # モデル
    args=training_args,                  # トレーニングの引数
    train_dataset=train_dataset,         # トレーニングデータセット
    eval_dataset=test_dataset            # 評価データセット
)

# モデルのトレーニング
trainer.train()

# テストデータで予測
predictions = trainer.predict(test_dataset)

# 予測結果のsoftmax確率を計算
softmax_probs = F.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

# 各テキストに対する予測確率と予測ラベル
for i, text in enumerate(test_texts):
    probs = softmax_probs[i]
    predicted_label = np.argmax(probs)
    predicted_label_name = label_encoder.inverse_transform([predicted_label])[0]
    
    print(f"Text: {text}")
    print(f"Predicted probabilities: normal={probs[0]:.4f}, aegi={probs[1]:.4f}, chupa={probs[2]:.4f}")
    print(f"Predicted label: {predicted_label_name}\n")
