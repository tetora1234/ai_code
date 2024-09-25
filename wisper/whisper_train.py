import os

# RTX8000を使用するように環境変数を設定
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # RTX8000がシステム上で2番目のGPUである場合

import subprocess
import gc
import torch
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

from datasets import load_dataset, Features, Value, Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# 定数と設定
CSV_PATH = r"C:\Users\user\Downloads\filtered_transcript_with_similarity.csv"
MODEL_CONFIG = r"C:\Users\user\Desktop\git\ai_code\wisper\models\Visual-novel-whisper2\checkpoint-1852"
LANGUAGE = "Japanese"
TASK = "transcribe"
OUTPUT_DIR = r"C:\Users\user\Desktop\git\ai_code\wisper\models\Visual-novel-whisper3"
SAMPLING_RATE = 16000
SAMPLE_FRAC = 0.99  # サンプリングする割合

def load_and_prepare_data(csv_path: str, sample_frac: float) -> Dataset:
    try:
        print(f"Loading dataset from: {csv_path}")
        
        # データセットのスキーマを定義
        features = Features({
            'filename': Value('string'),
            'transcript': Value('string')
        })
        
        dataset = load_dataset('csv', data_files=csv_path, features=features, split='train')
        
        if dataset is None:
            raise ValueError("Dataset is None after loading")
        
        print(f"Dataset loaded successfully. Dataset size before sampling: {len(dataset)}")
        
        # ランダムサンプリング
        dataset = dataset.train_test_split(test_size=sample_frac)['test']
        print(f"Dataset size after sampling: {len(dataset)}")
        
        print("Processing audio files")
        
        def process_audio(example):
            audio_path = example['filename']
            # librosaを使用して音声をロードし、必要に応じてリサンプリング
            audio_array, _ = librosa.load(audio_path, sr=SAMPLING_RATE)
            example['audio'] = {'array': audio_array, 'sampling_rate': SAMPLING_RATE}
            return example
        
        dataset = dataset.map(process_audio, remove_columns=['filename'], num_proc=12)
        
        print("Dataset preparation completed")
        return dataset
    except Exception as e:
        print(f"Error in load_and_prepare_data: {str(e)}")
        raise

# データセットの前処理関数
def prepare_dataset(batch: Dict, processor: WhisperProcessor) -> Optional[Dict]:
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=SAMPLING_RATE).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
    return batch

# データコラトラの定義
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", padding=True)
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding=True)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# カスタムトレーナーの定義
class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # 既存の_save メソッドを呼び出す
        super()._save(output_dir, state_dict)
        
        # プロセッサーを保存
        if self.processor:
            self.processor.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

# メイン処理
if __name__ == "__main__":
    # GPUの指定（RTX8000を使用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")  # 使用しているGPUの名前を表示

    # データの読み込みと前処理
    try:
        dataset = load_and_prepare_data(CSV_PATH, SAMPLE_FRAC)
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        exit(1)
    
    # プロセッサとモデルの設定
    processor = WhisperProcessor.from_pretrained(MODEL_CONFIG, language=LANGUAGE, task=TASK)
    prepared_dataset = dataset.map(
        prepare_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=['audio', 'transcript'],
        num_proc=12
    )
    
    # モデルの設定
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_CONFIG)
    model.to(device)  # モデルを指定したデバイスに移動

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task=TASK)
    model.config.suppress_tokens = []
        
    # データコラトラの設定
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=10,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_train_epochs=10,
        fp16=True if torch.cuda.is_available() else False,  # GPUが利用可能な場合のみTrue
        evaluation_strategy="no",  # エポックごとに評価
        save_strategy="epoch",  # エポックごとに保存
        logging_strategy="steps",  # エポックごとにログを記録
        logging_steps=1,
        report_to=["tensorboard"],
        push_to_hub=False,
        lr_scheduler_type="constant",
    )

    log_dir = training_args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])

    # トレーナーの作成と学習の実行
    trainer = CustomTrainer(
        args=training_args,
        model=model,
        train_dataset=prepared_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        processor=processor,  # プロセッサーをトレーナーに渡す
    )
    
    torch.cuda.empty_cache()  # メモリの解放
    gc.collect()  # ガベージコレクション
    trainer.train()

    # モデルの保存
    trainer.save_model(OUTPUT_DIR)