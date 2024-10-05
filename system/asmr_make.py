import os
import random
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
import configparser

# module フォルダのパスを sys.path に追加
from main import TextToSpeechSystem

# 初期化
counter = 0
audio_files = []
text_lengths = []  # 音声に対応するテキストの長さを保存

# モデル名に基づいて設定ファイルのパスを動的に構築し、設定を読み込む関数
def load_paths_by_model(model_name):
    base_dir = r"C:\Users\user\Desktop\git\ai_code\system\model_configs"
    file_name = "config.txt"
    
    file_path = os.path.join(base_dir, model_name, file_name)
    
    config = configparser.ConfigParser()
    config.read(file_path, encoding='utf-8')
    
    paths = {
        'usual': {
            'sovits_path': config.get('usual', 'sovits_path'),
            'gpt_path': config.get('usual', 'gpt_path')
        },
        'aegi': {
            'sovits_path': config.get('aegi', 'sovits_path'),
            'gpt_path': config.get('aegi', 'gpt_path')
        },
        'chupa': {
            'sovits_path': config.get('chupa', 'sovits_path'),
            'gpt_path': config.get('chupa', 'gpt_path')
        },
        'csv': {
            'csv_file_path': config.get('csv', 'csv_file_path')
        }
    }
    
    return paths

# モデル名を指定して設定ファイルのパスを読み込み
model_name = 'sirone'  # 'sirone' など別のモデル名も指定可能
paths = load_paths_by_model(model_name)

# CSVファイルのパスも取得できます
csv_file_path = paths['csv']['csv_file_path']

# ファイルパスの定義
text_file_path = r"C:\Users\user\Desktop\git\ai_code\llm\outputs\バイト中の生意気ギャルをバックヤードで逆転わからせ指導!_241003134751.txt"
csv_file_path = paths['csv']['csv_file_path']

# テキストを句読点で分割する関数
def split_sentences(text):
    sentences = re.split(r'(?<=[。！？♥])', text)
    return [s.strip() for s in sentences if s.strip()]

# ファイルから内容部分だけを取得する関数
def extract_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        content = ""
        for line in lines:
            if line.startswith('内容:'):
                content = line.replace('内容:', '').strip()
                break
        return content

# CSVファイルからtranscriptとfilename列を読み込む関数
def load_csv_transcripts(csv_file):
    df = pd.read_csv(csv_file)
    return df[['FilePath', 'Text', 'single_label']].astype(str).values.tolist()

# 類似度を計算し最も似ているテキストとその音声ファイルパスを取得する関数
def get_most_similar_text(sentence, transcripts, label, min_length_tolerance, max_length_tolerance):
    filtered_transcripts = [t for t in transcripts if t[2] == label]
    if not filtered_transcripts:
        print(f"ラベル {label} に対応するトランスクリプトが見つかりません")
        return None, None

    texts = [t[1] for t in filtered_transcripts]  # transcript部分のみ抽出
    vectorizer = TfidfVectorizer()
    
    while True:
        tfidf_matrix = vectorizer.fit_transform(texts + [sentence])
        query_vector = tfidf_matrix[-1]  # 入力文のベクトル
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix[:-1]).flatten()

        sentence_length = len(sentence)
        min_length = max(0, sentence_length - min_length_tolerance)
        max_length = sentence_length + max_length_tolerance

        valid_indices = [i for i, text in enumerate(texts) 
                         if min_length <= len(text) <= max_length]

        if valid_indices:
            valid_similarities = cosine_similarities[valid_indices]
            most_similar_index = valid_indices[valid_similarities.argmax()]
            
            most_similar_text = filtered_transcripts[most_similar_index][1]
            most_similar_filename = filtered_transcripts[most_similar_index][0]
            
            return most_similar_text, most_similar_filename
        
        min_length_tolerance += 1
        if min_length_tolerance < 0:
            print("適切な一致が見つかりません")
            return None, None

# テキストからカテゴリと本文を取得
def extract_category_and_text(text):
    categories = {
        "#USUAL": "usual",
        "#AEGI": "aegi",
        "#CHUPA": "chupa"
    }
    
    category_text_pairs = []
    current_category = None

    # 各文を処理
    for sentence in split_sentences(text):
        for key in categories:
            if sentence.startswith(key):
                current_category = categories[key]
                sentence = sentence.replace(key, "").strip()  # カテゴリ記号を取り除く
                break
        if current_category and sentence:
            category_text_pairs.append((current_category, sentence))

    return category_text_pairs

# ファイルから内容部分を抽出
generated_text = extract_content(text_file_path)

# 生成されたテキストからカテゴリと本文を取得
category_text_pairs = extract_category_and_text(generated_text)

# CSVファイルからtranscriptとfilenameを読み込み
csv_transcripts = load_csv_transcripts(csv_file_path)

for category, sentence in category_text_pairs:
    # 類似度の高いテキストと音声ファイルパスを取得
    similar_text, similar_audio_path = get_most_similar_text(sentence, csv_transcripts, category, min_length_tolerance=0, max_length_tolerance=5)
    
    if similar_text is None or similar_audio_path is None:
        print(f"文 '{sentence}' に対する適切な類似テキストが見つかりませんでした。")
        continue

    output_audio_path = fr"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\outputs\audio_{counter}.wav"

    # 音声合成システムの初期化と処理
    print(f"{category}\n{sentence}\n{similar_text}")
    tts = TextToSpeechSystem(paths[category]['sovits_path'], paths[category]['gpt_path'], similar_audio_path, similar_text)
    tts.process_text_file(sentence, output_audio_path)

    if os.path.exists(output_audio_path):  # ファイルが存在する場合
        audio_files.append(output_audio_path)
        text_lengths.append(len(sentence))  # テキストの長さを保存
        counter += 1  # カウンタをインクリメント
    else:
        print(f"文 '{sentence}' の音声ファイル作成に失敗しました。ファイルが見つかりません。")

# 音声ファイルの結合（無音挿入バージョン）
def combine_audio_files(audio_file_paths, text_lengths, output_path):
    combined = AudioSegment.empty()
    
    for i, file in enumerate(audio_file_paths):
        audio = AudioSegment.from_wav(file)
        combined += audio
        
        # 次のファイルが存在する場合に無音を挿入
        if i < len(audio_file_paths) - 1:
            if text_lengths[i] > 5:
                # 文字数が5文字以上の場合は0.75秒〜1.2秒の無音をランダムで挿入
                silence_duration_ms = random.uniform(750, 900)
            else:
                silence_duration_ms = random.uniform(50, 150)

            silence = AudioSegment.silent(duration=silence_duration_ms)
            combined += silence

    combined.export(output_path, format='wav')

final_output_path = rf"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\outputs\{os.path.basename(text_file_path)}.wav"
combine_audio_files(audio_files, text_lengths, final_output_path)
