import sys
import re
import sounddevice as sd
from scipy.io import wavfile

# module フォルダのパスを sys.path に追加
sys.path.append(r"C:\Users\user\Desktop\git\ai_code\module")
sys.path.append(r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS")
from TextClassification_infe_module import classify_text
from main import TextToSpeechSystem

# 初期化
counter = 0
# パスとパラメータの定義
normal_sovits_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\base\hime_e24_s1704.pth"
normal_gpt_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\base\hime-e30.ckpt"
normal_audio = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\dataset\normal_folder\him0017.wav"
normal_reference_text = "もう、貴方、足一本しか持てないの？"

aegi_sovits_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\aegi\hime_aegi_e24_s936.pth"
aegi_gpt_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\aegi\hime_aegi-e50.ckpt"
aegi_audio = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\data\aegi_folder\him0437.wav"
aegi_reference_text = "きゃふぅっ！ひぃっ！あはぁっ……はぁ、はぁ、はぁぁっ！あひぃいんんっ"

chupa_sovits_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\chupa\hime_chupa_e24_s240.pth"
chupa_gpt_path = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\chupa\hime_chupa-e50.ckpt"
chupa_audio = r"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\models\data\chupa_folder\him0157.wav"
chupa_reference_text = "ンッ、んふぅっ……チュゥッ……ちゅぷちゅぷっ、んちゅるっ、ちゅぴっ、ちゅぱぁっ……じゅりゅりゅりゅりゅっ……！"

normal_tts_system = TextToSpeechSystem(normal_sovits_path, normal_gpt_path, normal_audio, normal_reference_text)
aegi_tts_system = TextToSpeechSystem(aegi_sovits_path, aegi_gpt_path, aegi_audio, aegi_reference_text)
chupa_tts_system = TextToSpeechSystem(chupa_sovits_path, chupa_gpt_path, chupa_audio, chupa_reference_text)

# ファイルパスの定義
text_file_path = r"C:\Users\user\Desktop\git\ai_code\llm\outputs\240919064744.txt"

# テキストを句読点で分割する関数
def split_sentences(text):
    # 。や！や？で分割し、それぞれの文の末尾に句読点を残す
    sentences = re.split(r'(?<=[。！？♥])', text)
    return [s.strip() for s in sentences if s.strip()]  # 空白や空の文を除外

# ファイルから内容部分だけを取得する関数
def extract_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        # '内容:' の行を探して、その後のテキストを抽出
        for line in lines:
            if line.startswith('内容:'):
                return line.replace('内容:', '').strip()
            
# ファイルから内容部分を抽出
generated_text = extract_content(text_file_path)

# 生成されたテキストを文に分割
sentences = split_sentences(generated_text)

MIN_TEXT_LENGTH = 3  # 最小文字数

for sentence in sentences:
    if len(sentence) < MIN_TEXT_LENGTH:
        continue  # テキストが短すぎる場合はスキップ

    label, probs = classify_text(sentence)

    # 結果を表示
    print(f"テキスト: {sentence}")
    print(f"予測結果: {label}")
    print(f"各クラスの予測確率: {probs}")

    # インデックスを使ってファイル名を生成
    input_audio_path = fr"C:\Users\user\Desktop\git\ai_code\GPTSoVITS\outputs\audio_{counter}.wav"

    if label == "通常":
        normal_tts_system.process_text_file(sentence, input_audio_path)
    elif label == "あえぎ":
        aegi_tts_system.process_text_file(sentence, input_audio_path)
    elif label == "チュパ":
        chupa_tts_system.process_text_file(sentence, input_audio_path)

    # 次のファイル用にカウンタをインクリメント
    counter += 1

    # 音声ファイルを再生
    sample_rate, data = wavfile.read(input_audio_path)
    sd.play(data, samplerate=sample_rate)
    sd.wait()  # 再生が終了するまで待機
