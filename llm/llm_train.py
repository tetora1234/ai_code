import os
import torch

# RTX8000を使用するように環境変数を設定
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")  # 使用しているGPUの名前を表示

import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import Dataset
from huggingface_hub import login
import gc
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers.trainer_callback import TrainerCallback

# 定数の定義
HF_TOKEN = "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"
DATA_FILE_PATH = r"C:\Users\user\Desktop\git\ai_code\dataset\llm\filtered_output.json"
MODEL_NAME = r"C:\Users\user\Desktop\git\ai_code\models\llm\fine_tuned_model"
OUTPUT_DIR = r"C:\Users\user\Desktop\git\ai_code\models\llm\results"
LOGGING_DIR = r"C:\Users\user\Desktop\git\ai_code\models\llm\logs"
SAVE_DIRECTORY = r"C:\Users\user\Desktop\git\ai_code\models\llm\fine_tuned_model2"

# Hugging Faceにログイン
login(token=HF_TOKEN)

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
    text = example['学習用']
    tokenized = tokenizer(text, padding=True, truncation=True, max_length=7000, return_tensors="pt")
    return {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0]
    }

class MemoryManagementCallback(TrainerCallback):
    def __init__(self, steps_to_cleanup):
        self.steps_to_cleanup = steps_to_cleanup

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.steps_to_cleanup == 0:
            gc.collect()
            torch.cuda.empty_cache()

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

    # PADトークンの追加
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 量子化設定の準備
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 量子化を適用したモデルのロード
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_auth_token=True,
        # quantization_config=quantization_config, #量子化する場合
    )

    # LoRAの設定
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果言語モデルタスク
        r=128,  # ランク
        lora_alpha=128,  # LoRAのスケーリング係数
        lora_dropout=0.1,  # ドロップアウト率
        bias="none",  # バイアスの取り扱い
        target_modules=["q_proj", "v_proj"]  # 適用対象のモジュールを指定
    )

    # LoRAモデルの準備
    model = get_peft_model(model, lora_config)

    # データセットのトークン化
    tokenized_datasets = dataset.map(tokenize_function, remove_columns=dataset.column_names)

    # データコレータの設定
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=25,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        logging_dir=LOGGING_DIR,
        logging_steps=1,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="constant",
    )

    memory_callback = MemoryManagementCallback(steps_to_cleanup=1)

    # トレーナーの設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
        callbacks=[memory_callback],
    )

    # トレーニングの実行
    trainer.train()

    # モデルの保存前にマージ
    merged_model = model.merge_and_unload()

    # モデルの保存
    merged_model.save_pretrained(SAVE_DIRECTORY)
    tokenizer.save_pretrained(SAVE_DIRECTORY)
    print(f"Saving merged model to {SAVE_DIRECTORY}")

if __name__ == "__main__":
    main()
