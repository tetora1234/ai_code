
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer

# データセットの読み込み (FilePath, Text, Classification が含まれる)
df = pd.read_csv(r"C:\Users\user\Desktop\git\ai_code\TextClassification\dataset\results.csv")

# 空のテキストやNaNを含む行を削除
df = df.dropna(subset=['transcript', 'classification'])
df = df[df['transcript'].str.strip() != '']

# マルチラベルのマッピング
classification_map = {
    "usual": 0,
    "aegi": 1,
    "chupa": 2
}

# ラベルをマルチラベル形式に変換
df['classification'] = df['classification'].apply(lambda x: x.split(","))
mlb = MultiLabelBinarizer(classes=list(classification_map.keys()))
df['label'] = list(mlb.fit_transform(df['classification']))

# テキストデータとラベルを準備
train_texts = df['transcript'].tolist()
train_labels = df['label'].tolist()

# トークナイザーの初期化
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# トークン化
train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

# データセットをPyTorchのテンソルに変換
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)  # マルチラベル対応
        return item

train_dataset = TextDataset(train_encodings, train_labels)

# モデルの定義（BERTの分類用ヘッド付きモデル、3つのラベルに対応）
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# トレーニングの設定
training_args = TrainingArguments(
    output_dir='C:\\Users\\user\\Desktop\\git\\ai_code\\TextClassification\\models',
    num_train_epochs=10,
    per_device_train_batch_size=256,
    logging_dir='C:\\Users\\user\\Desktop\\git\\ai_code\\TextClassification\\log',
    logging_steps=1,
    save_strategy="epoch"
)

# トレーナーの初期化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# モデルのトレーニング
trainer.train()
