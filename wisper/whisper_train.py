import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

import subprocess

import librosa
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from pathlib import Path

from datasets import load_dataset, Features, Value, Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Constants and settings
CSV_PATH = r"C:\Users\user\Desktop\git\ai_code\wisper\filtered_data.csv"
MODEL_CONFIG = r"C:\Users\user\Desktop\git\ai_code\wisper\models\whisper-large-v3-nsfw"
LANGUAGE = "Japanese"
TASK = "transcribe"
OUTPUT_DIR = r"C:\Users\user\Desktop\git\ai_code\wisper\models\whisper-large-v3-nsfw_2"
SAMPLING_RATE = 16000
SAMPLE_FRAC = 0.1
WAV_DIR = r"C:\Users\user\Desktop\git\ai_code\wisper\dataset"

def convert_to_wav(input_path: str, output_path: str) -> str:
    """Convert audio file to WAV format."""
    try:
        audio, sr = librosa.load(input_path, sr=SAMPLING_RATE)
        sf.write(output_path, audio, sr)
        return output_path
    except Exception as e:
        print(f"Error converting file {input_path}: {str(e)}")
        return None

def load_and_prepare_data(csv_path: str, sample_frac: float) -> Dataset:
    try:
        print(f"Loading dataset from: {csv_path}")
        
        features = Features({
            'FilePath': Value('string'),
            'Text': Value('string')
        })
        
        dataset = load_dataset('csv', data_files=csv_path, features=features, split='train')
        
        if dataset is None:
            raise ValueError("Dataset is None after loading")
        
        print(f"Dataset loaded successfully. Dataset size before sampling: {len(dataset)}")
        
        dataset = dataset.train_test_split(test_size=sample_frac)['test']
        print(f"Dataset size after sampling: {len(dataset)}")
        
        print("Converting audio files to WAV format and loading audio")
        
        def process_and_load_audio(example):
            input_path = example['FilePath']
            output_path = os.path.join(WAV_DIR, Path(input_path).stem + ".wav")
            
            if not os.path.exists(output_path):
                wav_path = convert_to_wav(input_path, output_path)
            else:
                wav_path = output_path
            
            if wav_path:
                audio_array, _ = librosa.load(wav_path, sr=SAMPLING_RATE)
                example['audio'] = {'array': audio_array, 'sampling_rate': SAMPLING_RATE}
            else:
                example['audio'] = None
            
            return example
        
        os.makedirs(WAV_DIR, exist_ok=True)
        dataset = dataset.map(process_and_load_audio, remove_columns=['FilePath'], num_proc=12)
        
        # Filter out examples where audio is None
        dataset = dataset.filter(lambda example: example['audio'] is not None)
        
        print("Dataset preparation completed")
        print(f"Final dataset size: {len(dataset)}")
        print(f"Columns in the dataset: {dataset.column_names}")
        
        return dataset
    except Exception as e:
        print(f"Error in load_and_prepare_data: {str(e)}")
        raise

def prepare_dataset(batch: Dict, processor: WhisperProcessor) -> Optional[Dict]:
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=SAMPLING_RATE).input_features[0]
    batch["labels"] = processor.tokenizer(batch["Text"]).input_ids
    return batch

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

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        
        if self.processor:
            self.processor.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # Load and preprocess data
    try:
        dataset = load_and_prepare_data(CSV_PATH, SAMPLE_FRAC)
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        exit(1)
    
    # Set up processor and model
    processor = WhisperProcessor.from_pretrained(MODEL_CONFIG, language=LANGUAGE, task=TASK)
    prepared_dataset = dataset.map(
        prepare_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=dataset.column_names,  # Remove all existing columns
        num_proc=12
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_CONFIG)
    model.to(device)

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task=TASK)
    model.config.suppress_tokens = []
        
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=20,
        fp16=True if torch.cuda.is_available() else False,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        report_to=["tensorboard"],
        push_to_hub=False,
        lr_scheduler_type="cosine",
    )

    log_dir = training_args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])

    trainer = CustomTrainer(
        args=training_args,
        model=model,
        train_dataset=prepared_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        processor=processor,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)