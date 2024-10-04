import os
import sys
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
import configparser

# module フォルダのパスを sys.path に追加
sys.path.append(r"C:\Users\user\Desktop\git\ai_code\module")
sys.path.append(r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS")
from TextClassification_infe_module import classify_text
from main import TextToSpeechSystem

# 初期化
counter = 0
pending_text = ""  # 短すぎるテキストを一時的に保存する変数
MIN_TEXT_LENGTH = 10  # 最小文字数

# モデル名に基づいて設定ファイルのパスを動的に構築し、設定を読み込む関数
def load_paths_by_model(model_name):
    # 基本の設定ファイルパスのパターン
    base_dir = r"C:\Users\user\Desktop\git\ai_code\system\model_configs"
    file_name = "config.txt"
    
    # モデル名に基づいてパスを構築
    file_path = os.path.join(base_dir, model_name, file_name)
    
    # 設定ファイルを読み込む
    config = configparser.ConfigParser()
    config.read(file_path, encoding='utf-8')
    
    # 既存の設定フォーマットに合わせて辞書を返す
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
    # 文末の句読点（。、！？♥）や空白、複数のスペースを基準に分割
    sentences = re.split(r'(?<=[。！？♥\s♪])', text)
    return [s.strip() for s in sentences if s.strip()]  # 空白や空の文を除外

# ファイルから内容部分だけを取得する関数
def extract_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('内容:'):
                return line.replace('内容:', '').strip()

# CSVファイルからtranscriptとfilename列を読み込む関数
def load_csv_transcripts(csv_file):
    df = pd.read_csv(csv_file)
    return df[['FilePath', 'Text', 'single_label']].astype(str).values.tolist()

# 類似度を計算し最も似ているテキストとその音声ファイルパスを取得する関数
def get_most_similar_text(sentence, transcripts, label, min_length_tolerance, max_length_tolerance):
    # ラベルに基づいてtranscriptsをフィルタリング
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

        sentence_length = len(sentence)  # sentenceの長さを取得
        min_length = max(0, sentence_length - min_length_tolerance)  # 最小長さ
        max_length = sentence_length + max_length_tolerance  # 最大長さ

        # 有効な類似テキストとそのインデックスをフィルタリング
        valid_indices = [i for i, text in enumerate(texts) 
                         if min_length <= len(text) <= max_length]

        if valid_indices:
            # 最も類似度が高いインデックスを取得
            valid_similarities = cosine_similarities[valid_indices]
            most_similar_index = valid_indices[valid_similarities.argmax()]
            
            most_similar_text = filtered_transcripts[most_similar_index][1]
            most_similar_filename = filtered_transcripts[most_similar_index][0]
            
            return most_similar_text, most_similar_filename
        
        # valid_indicesが空の場合、min_length_toleranceを1下げてリトライ
        min_length_tolerance += 1
        if min_length_tolerance < 0:
            print("適切な一致が見つかりません")
            return None, None  # これ以上リトライできない場合

# ファイルから内容部分を抽出
generated_text = extract_content(text_file_path)

# 生成されたテキストを文に分割
sentences = split_sentences(generated_text)

# CSVファイルからtranscriptとfilenameを読み込み
csv_transcripts = load_csv_transcripts(csv_file_path)

audio_files = []

# 短すぎるテキストを結合する際に記号で終わっている場合、スペースを追加する関数
def add_space_if_ends_with_punctuation(text):
    # 文末が句読点（。、！？♪など）の場合、スペースを追加
    if re.search(r'[。！？♪]$', text):
        return text + " "  # 文末にスペースを追加
    return text  # それ以外の場合はそのまま返す

for sentence in sentences:
    # 保存されている短いテキストがあれば、それを現在の文に結合
    if pending_text:
        # pending_textが記号で終わっていたらスペースを追加
        pending_text = add_space_if_ends_with_punctuation(pending_text)

    sentence = pending_text + sentence

    if len(sentence) < MIN_TEXT_LENGTH:
        pending_text = sentence  # テキストが短すぎる場合は次に結合するために保存
        continue  # すぐに次のループへ

    # 類似度の高いテキストと音声ファイルパスを取得
    label, probs = classify_text(sentence)
    similar_text, similar_audio_path = get_most_similar_text(sentence, csv_transcripts, label, min_length_tolerance=0, max_length_tolerance=5)
    
    if similar_text is None or similar_audio_path is None:
        print(f"文 '{sentence}' に対する適切な類似テキストが見つかりませんでした。")
        continue

    # 出力ファイルパスを定義（インデックスを使ってファイル名を生成）
    output_audio_path = fr"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\outputs\audio_{counter}.wav"

    # 音声合成システムの初期化と処理
    if label == "usual":
        normal_tts_system = TextToSpeechSystem(paths['usual']['sovits_path'], paths['usual']['gpt_path'], similar_audio_path, similar_text)
        normal_tts_system.process_text_file(sentence, output_audio_path)
    elif label == "aegi":
        aegi_tts_system = TextToSpeechSystem(paths['aegi']['sovits_path'], paths['aegi']['gpt_path'], similar_audio_path, similar_text)
        aegi_tts_system.process_text_file(sentence, output_audio_path)
    elif label == "chupa":
        chupa_tts_system = TextToSpeechSystem(paths['chupa']['sovits_path'], paths['chupa']['gpt_path'], similar_audio_path, similar_text)
        chupa_tts_system.process_text_file(sentence, output_audio_path)

    # 短すぎるテキストの保存をリセット
    pending_text = ""

    # 次のファイル用にカウンタをインクリメント
    counter += 1

    audio_files.append(output_audio_path)

# 音声ファイルの結合（無音挿入バージョン）
def combine_audio_files(audio_file_paths, output_path, silence_duration_ms=1000):
    combined = AudioSegment.empty()  # 空の音声を作成
    silence = AudioSegment.silent(duration=silence_duration_ms)
    
    for i, file in enumerate(audio_file_paths):
        audio = AudioSegment.from_wav(file)  # wavファイルを読み込む
        combined += audio  # 音声を結合

        # 最後の音声ファイルでなければ無音を挿入
        if i < len(audio_file_paths) - 1:
            combined += silence

    # 結合された音声を1つのファイルとして出力
    combined.export(output_path, format="wav")

# すべての音声ファイルを結合して1つのファイルに保存
final_output_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\outputs\combined_audio.wav"
combine_audio_files(audio_files, final_output_path)
