import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# データセットの読み込み (FilePath, Text, Classification が含まれる)
df = pd.read_csv('C:\\Users\\user\\Desktop\\git\\ai_code\\TextClassification\\dataset\\filtered_out.csv')

# 空のテキストやNaNを含む行を削除
df = df.dropna(subset=['Text', 'Classification'])
df = df[df['Text'].str.strip() != '']

# マルチラベルのマッピング
classification_map = {
    "通常会話": 0,
    "エロ会話": 1,
    "あえぎ": 2, 
    "ふぇら": 3,
    "おほごえ": 4,
    "いきごえ": 5
}

# ラベルをマルチラベル形式に変換
df['Classification'] = df['Classification'].apply(lambda x: x.split(","))
mlb = MultiLabelBinarizer(classes=list(classification_map.keys()))
df['label'] = list(mlb.fit_transform(df['Classification']))

# データセットを訓練データとテストデータに分割
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Text'].tolist(),
    df['label'].tolist(),
    test_size=0.2
)

# トークナイザーの初期化
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# トークン化
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
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)  # マルチラベル対応
        return item

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

# モデルの定義（BERTの分類用ヘッド付きモデル、5つのラベルに対応）
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# トレーニングの設定
training_args = TrainingArguments(
    output_dir='C:\\Users\\user\\Desktop\\git\\ai_code\\TextClassification\\models',  # モデルの保存ディレクトリ
    num_train_epochs=30,              # エポック数
    per_device_train_batch_size=16,   # バッチサイズ
    per_device_eval_batch_size=16,    # 評価バッチサイズ
    warmup_steps=500,                 # ウォームアップステップ
    weight_decay=0.01,                # 重みの減衰率
    logging_dir='./TextClassification/logs',  # ログディレクトリ
    logging_steps=10,
    evaluation_strategy="epoch",      # エポックごとに評価
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

# テストデータに対する予測
predictions = trainer.predict(test_dataset)
pred_probs = torch.sigmoid(torch.Tensor(predictions.predictions)).numpy()  # シグモイド関数を適用して確率を取得

# 確率を表示
for i, probs in enumerate(pred_probs):
    print(f"Sample {i+1}: {probs}")

# 評価結果の表示
results = trainer.evaluate()
print(f"Evaluation results: {results}")
