import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from huggingface_hub import login
import torch
import gc

# 定数の定義
HF_TOKEN = "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"
CUDA_DEVICE = "1"
DATA_FILE_PATH = r"C:\Users\user\Desktop\git\ai_code\dataset\llm\output.json"
MODEL_NAME = "AXCXEPT/EZO-Common-T2-2B-gemma-2-it"
OUTPUT_DIR = r"C:\Users\user\Desktop\git\ai_code\models\llm\results"
LOGGING_DIR = r"C:\Users\user\Desktop\git\ai_code\models\llm\logs"
SAVE_DIRECTORY = r"C:\Users\user\Desktop\git\ai_code\models\llm\fine_tuned_model"

# Hugging Faceにログイン
login(token=HF_TOKEN)

# RTX8000を使用するように環境変数を設定
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

# データの読み込み
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# データセットの作成
def create_dataset(data):
    return Dataset.from_dict({
        'タイトル': [item['タイトル'] for item in data],
        '内容': [item['内容'] for item in data]
    })

# テキストの前処理
def preprocess_text(examples):
    return {
        '学習用': [f"タイトル: {examples['タイトル']}\n内容: {examples['内容']}"]
    }

# トークン化関数
def tokenize_function(example):
    # テキストを結合して、タイトルと内容を一つの文字列にする
    text = example['学習用']
    # トークン化、パディング、切り捨てを行う
    tokenized = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    # 入力IDsと注意マスクのみを返す
    return {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0]
    }

# メイン関数
def main():
    # データの読み込みとデータセットの作成
    data = load_data(DATA_FILE_PATH)
    dataset = create_dataset(data)
    
    # テキストの前処理
    dataset = dataset.map(preprocess_text)

    # モデルとトークナイザーの準備
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
    
    # パディングトークンが設定されていない場合、適切なトークンを設定する
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=True)

    # gradient_checkpointing の有効化
    model.config.gradient_checkpointing = True

    # データセットのトークン化
    tokenized_datasets = dataset.map(tokenize_function, remove_columns=dataset.column_names)

    # データコレータの設定
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_dir=LOGGING_DIR,
        logging_steps=10,
        save_strategy="epoch",  # エポックごとにモデルを保存
        fp16=True,
    )

    # トレーナーの設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # トレーニングの実行
    trainer.train()

    # モデルの保存
    trainer.save_model(SAVE_DIRECTORY)

    # トークナイザーの保存
    tokenizer.save_pretrained(SAVE_DIRECTORY)

    # メモリの解放
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
