import torch
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import gc

# モデル名の変数を追加
MODEL_NAME = "slicer_opt"
MODEL_PATH = rf"C:\Users\user\Desktop\git\ai_code\models\whisper\Visual-novel-whisper\checkpoint-940"
INPUT_DIR = r"C:\Users\user\Downloads\GPT-SoVITS-beta0706\input"
OUTPUT_DIR_BASE = r"C:\Users\user\Downloads\GPT-SoVITS-beta0706\output\asr_opt"
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, "Data", MODEL_NAME, "raw")

# OUTPUT_DIRが存在しない場合は作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(file_path, target_sr=16000):
    audio, orig_sr = librosa.load(file_path, sr=None)
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return {"array": audio, "sampling_rate": target_sr}

def split_on_silence(audio, max_chunk_duration=30, initial_min_silence_len=1000, initial_silence_thresh=-60, keep_silence=200):
    y = audio["array"]
    sr = audio["sampling_rate"]
    
    min_silence_len = initial_min_silence_len
    silence_thresh = initial_silence_thresh
    
    while True:
        non_silent_intervals = librosa.effects.split(
            y,
            top_db=-silence_thresh,
            frame_length=int(sr * min_silence_len / 1000),
            hop_length=int(sr * keep_silence / 1000)
        )
        
        chunks = []
        for interval in non_silent_intervals:
            start, end = interval
            audio_chunk = y[start:end]
            duration = len(audio_chunk) / sr
            
            if duration > max_chunk_duration:
                sub_chunks = split_on_silence(
                    {"array": audio_chunk, "sampling_rate": sr},
                    max_chunk_duration=max_chunk_duration,
                    initial_min_silence_len=min_silence_len-10,
                    initial_silence_thresh=silence_thresh+1,
                    keep_silence=keep_silence
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append({"array": audio_chunk, "sampling_rate": sr})
        
        if all(len(chunk["array"]) / sr <= max_chunk_duration for chunk in chunks):
            return chunks
        
        min_silence_len = max(50, min_silence_len - 10)
        silence_thresh = min(-20, silence_thresh + 1)

def save_audio_chunks(chunks, base_name, output_dir):
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(output_dir, f"{base_name}-{i}.wav")
        sf.write(chunk_path, chunk["array"], chunk["sampling_rate"])
        chunk_paths.append(chunk_path)
    
    return chunk_paths

def transcribe(audio_file_path, processor, model):
    audio_input, _ = librosa.load(audio_file_path, sr=16000)
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="Japanese",
            task="transcribe",
            num_beams=1,
            no_repeat_ngram_size=4,
        )
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    del input_features, predicted_ids
    torch.cuda.empty_cache()
    gc.collect()
    
    return transcription[0]

def process_chunk(chunk_path, processor, model, chunk_index):
    result = transcribe(chunk_path, processor, model)
    print(f'{chunk_index}番目: {result}')
    return result

def post_process_text(text):
    text = text.replace("�", "")
    return text

def save_transcription_esd_list(transcriptions, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for file_name, file_transcriptions in transcriptions.items():
            for i, transcription in enumerate(file_transcriptions):
                processed_transcription = post_process_text(transcription)
                f.write(f"{OUTPUT_DIR}\{file_name}-{i}.wav|{MODEL_NAME}|JP|{processed_transcription}\n")

def process_audio_file(audio_file_path, processor, model, output_dir):
    print(audio_file_path)
    audio = load_audio(audio_file_path)
    audio_chunks = split_on_silence(audio)
    audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    chunk_paths = save_audio_chunks(audio_chunks, audio_file_name, output_dir)

    transcriptions = []

    for i, chunk_path in enumerate(chunk_paths):
        result = process_chunk(chunk_path, processor, model, i)
        transcriptions.append(result)

    return audio_file_name, transcriptions

def main():
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, language="Japanese", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

    all_transcriptions = {}
    
    for file_name in os.listdir(INPUT_DIR):
        if file_name.endswith(('.wav', '.mp3', '.flac')):
            audio_file_path = os.path.join(INPUT_DIR, file_name)
            file_name, transcriptions = process_audio_file(audio_file_path, processor, model, OUTPUT_DIR)
            all_transcriptions[file_name] = transcriptions

    # 結果をesd.listに保存
    esd_list_path = os.path.join(OUTPUT_DIR_BASE, "Data", MODEL_NAME, "slicer_opt.list")
    os.makedirs(os.path.dirname(esd_list_path), exist_ok=True)
    save_transcription_esd_list(all_transcriptions, esd_list_path)

if __name__ == "__main__":
    main()
