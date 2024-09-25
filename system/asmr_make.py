import sys
import re
import sounddevice as sd
from scipy.io import wavfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# module フォルダのパスを sys.path に追加
sys.path.append(r"C:\Users\user\Desktop\git\ai_code\module")
sys.path.append(r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS")
from TextClassification_infe_module import classify_text
from main import TextToSpeechSystem

# 初期化
counter = 0
pending_text = ""  # 短すぎるテキストを一時的に保存する変数
MIN_TEXT_LENGTH = 3  # 最小文字数

# パスとパラメータの定義
normal_sovits_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\base\hime_e24_s1704.pth"
normal_gpt_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\base\hime-e30.ckpt"

aegi_sovits_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\aegi\hime_aegi_e24_s936.pth"
aegi_gpt_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\aegi\hime_aegi-e50.ckpt"

chupa_sovits_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\chupa\hime_chupa_e24_s240.pth"
chupa_gpt_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\chupa\hime_chupa-e50.ckpt"

# ファイルパスの定義
text_file_path = r"C:\Users\user\Desktop\git\ai_code\llm\outputs\240919064744.txt"
csv_file_path = r'C:\Users\user\Desktop\git\ai_code\check.csv'  # CSVファイルのパス

# テキストを句読点で分割する関数
def split_sentences(text):
    sentences = re.split(r'(?<=[。！？♥])', text)
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
    return df[['filename', 'transcript']].astype(str).values.tolist()

# 類似度を計算し最も似ているテキストとその音声ファイルパスを取得する関数
def get_most_similar_text(sentence, transcripts, min_length_tolerance, max_length_tolerance):
    texts = [t[1] for t in transcripts]  # transcript部分のみ抽出
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts + [sentence])
    
    query_vector = tfidf_matrix[-1]  # 入力文のベクトル
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix[:-1]).flatten()

    sentence_length = len(sentence)  # sentenceの長さを取得
    min_length = max(0, sentence_length - min_length_tolerance)  # 最小長さ
    max_length = sentence_length + max_length_tolerance  # 最大長さ

    # 有効な類似テキストとそのインデックスをフィルタリング
    valid_indices = [i for i, text in enumerate(texts) 
                     if min_length <= len(text) <= max_length]

    if not valid_indices:
        return None, None  # 有効なテキストがない場合

    # 最も類似度が高いインデックスを取得
    valid_similarities = cosine_similarities[valid_indices]
    most_similar_index = valid_indices[valid_similarities.argmax()]
    
    most_similar_text = transcripts[most_similar_index][1]
    most_similar_filename = transcripts[most_similar_index][0]
    most_similar_score = cosine_similarities[most_similar_index]
    
    return most_similar_text, most_similar_filename


# ファイルから内容部分を抽出
generated_text = extract_content(text_file_path)

# 生成されたテキストを文に分割
sentences = split_sentences(generated_text)

# CSVファイルからtranscriptとfilenameを読み込み
csv_transcripts = load_csv_transcripts(csv_file_path)

for sentence in sentences:
    # 保存されている短いテキストがあれば、それを現在の文に結合
    sentence = pending_text + sentence

    if len(sentence) < MIN_TEXT_LENGTH:
        pending_text = sentence  # テキストが短すぎる場合は次に結合するために保存
        continue  # すぐに次のループへ

    # 類似度の高いテキストと音声ファイルパスを取得
    similar_text, similar_audio_path = get_most_similar_text(sentence, csv_transcripts, min_length_tolerance=3, max_length_tolerance=10)
    label, probs = classify_text(similar_text)

    # 出力ファイルパスを定義（インデックスを使ってファイル名を生成）
    output_audio_path = fr"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\outputs\audio_{counter}.wav"

    # 音声合成システムの初期化と処理
    if label == "通常":
        normal_tts_system = TextToSpeechSystem(normal_sovits_path, normal_gpt_path, similar_audio_path, similar_text)
        normal_tts_system.process_text_file(sentence, output_audio_path)
    elif label == "あえぎ":
        aegi_tts_system = TextToSpeechSystem(aegi_sovits_path, aegi_gpt_path, similar_audio_path, similar_text)
        aegi_tts_system.process_text_file(sentence, output_audio_path)
    elif label == "チュパ":
        chupa_tts_system = TextToSpeechSystem(chupa_sovits_path, chupa_gpt_path, similar_audio_path, similar_text)
        chupa_tts_system.process_text_file(sentence, output_audio_path)

    print(label)
    print(similar_text)
    print(sentence)

    # 短すぎるテキストの保存をリセット
    pending_text = ""

    # 次のファイル用にカウンタをインクリメント
    counter += 1

    # 音声ファイルを再生
    sample_rate, data = wavfile.read(output_audio_path)
    sd.play(data, samplerate=sample_rate)
    sd.wait()  # 再生が終了するまで待機
