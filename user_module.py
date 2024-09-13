import os
import sys
cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
os.environ["version"] = 'v2'
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, now_dir)
sys.path.insert(0, os.path.join(now_dir, "GPT_SoVITS"))
import gradio as gr
import numpy as np
import os,librosa,torch
from scipy.io.wavfile import write as wavwrite
from GPT_SoVITS.feature_extractor import cnhubert
cnhubert.cnhubert_base_path=cnhubert_base_path
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from time import time as ttime
from GPT_SoVITS.module.mel_processing import spectrogram_torch
import tempfile
from tools.my_utils import load_audio
import os
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = False

loaded_sovits_model = []
loaded_gpt_model = []
ssl_model = cnhubert.get_model()
if (is_half == True):
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def load_model(sovits_path, gpt_path):
    global ssl_model
    global loaded_sovits_model
    global loaded_gpt_model
    vq_model = None
    t2s_model = None
    dict_s2 = None
    dict_s1 = None
    hps = None

    # Sovitsモデルの検索
    for path, dict_s2_, model in loaded_sovits_model:
        if path == sovits_path:
            vq_model = model
            dict_s2 = dict_s2_
            break

    # GPTモデルの検索
    for path, dict_s1_, model in loaded_gpt_model:
        if path == gpt_path:
            t2s_model = model
            dict_s1 = dict_s1_
            break

    # Sovitsモデルのロード
    if dict_s2 is None:
        dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]

    # GPTモデルのロード
    if dict_s1 is None:
        dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]

    # Dictをオブジェクト化
    class DictToAttrRecursive:
        def __init__(self, input_dict):
            for key, value in input_dict.items():
                setattr(self, key, DictToAttrRecursive(value) if isinstance(value, dict) else value)

    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"

    # Sovitsモデルの初期化
    if not vq_model:
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        vq_model = vq_model.half().to(device) if is_half else vq_model.to(device)
        vq_model.eval()
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        loaded_sovits_model.append((sovits_path, dict_s2, vq_model))

    # GPTモデルの初期化
    if not t2s_model:
        t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.half() if is_half else t2s_model
        t2s_model = t2s_model.to(device)
        t2s_model.eval()
        total = sum([param.nelement() for param in t2s_model.parameters()])
        loaded_gpt_model.append((gpt_path, dict_s1, t2s_model))

    hz = 50
    max_sec = config['data']['max_sec']
    
    return vq_model, ssl_model, t2s_model, hps, config, hz, max_sec

def get_spepc(hps, filename):
    audio=load_audio(filename,int(hps.data.sampling_rate)) 
    audio = audio / np.max(np.abs(audio))
    audio=torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length,hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,center=False)
    return spec

def create_tts_fn(vq_model, ssl_model, t2s_model, hps, config, hz, max_sec):
    def tts_fn(ref_wav_path, prompt_text, prompt_language, target_phone, text_language, target_text=None):
        t0 = ttime()
        prompt_text = prompt_text.strip()
        
        with torch.no_grad():
            # 音声ファイルの読み込み
            wav16k, sr = librosa.load(ref_wav_path, sr=16000, mono=False)
            if wav16k.ndim == 2:
                power = np.sum(np.abs(wav16k) ** 2, axis=1)
                direction = power / np.sum(power)
                wav16k = (wav16k[0] + wav16k[1]) / 2
            else:
                direction = np.array([1, 1])

            # 音声の前処理
            wav16k = torch.from_numpy(np.concatenate([wav16k, np.zeros(int(hps.data.sampling_rate * 0.3))])).float()
            wav16k = wav16k.half().to(device) if is_half else wav16k.to(device)

            # SSLモデルからコンテンツ特徴量を抽出
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]

        phones1, _, _ = clean_text(prompt_text, prompt_language)
        phones1 = cleaned_text_to_sequence(phones1)
        
        audio_opt = []
        zero_wav = np.zeros((2, int(hps.data.sampling_rate * 0.3)), dtype=np.float16 if is_half else np.float32)

        # 目標の音素ごとの処理
        for phones2 in get_phone_from_str_list(target_phone, text_language):
            if not phones2 or (len(phones2) == 1 and phones2[0] == ""):
                continue
            
            phones2 = cleaned_text_to_sequence(phones2)
            bert1 = torch.zeros((1024, len(phones1)), dtype=torch.float16 if is_half else torch.float32).to(device)
            bert2 = torch.zeros((1024, len(phones2))).to(bert1)
            bert = torch.cat([bert1, bert2], dim=1).unsqueeze(0).to(device)

            all_phoneme_ids = torch.LongTensor(phones1 + phones2).unsqueeze(0).to(device)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
            prompt = prompt_semantic.unsqueeze(0).to(device)
            
            idx, cnt = 0, 0
            while idx == 0 and cnt < 2:
                with torch.no_grad():
                    pred_semantic, idx = t2s_model.model.infer_panel(
                        all_phoneme_ids, all_phoneme_len, prompt, bert,
                        top_k=config['inference']['top_k'],
                        early_stop_num=hz * max_sec
                    )
                cnt += 1

            if idx == 0:
                return "Error: Generation failure: bad zero prediction.", None

            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            refer = get_spepc(hps, ref_wav_path).half().to(device) if is_half else get_spepc(hps, ref_wav_path).to(device)

            # 音声デコード
            audio = vq_model.decode(pred_semantic, torch.LongTensor(phones2).unsqueeze(0).to(device), refer).cpu().numpy()[0, 0]
            audio = np.expand_dims(audio, 0) * direction[:, np.newaxis]

            # 出力音声を結合
            audio_opt.append(audio)
            audio_opt.append(zero_wav)

        # 結果の音声ファイルを保存
        audio = (hps.data.sampling_rate, (np.concatenate(audio_opt, axis=1) * 32768).astype(np.int16).T)
        filename = tempfile.mktemp(suffix=".wav", prefix=f"{prompt_text[:8].replace(' ', '_')}_{target_text[:8].replace(' ', '_')}_")
        wavwrite(filename, audio[0], audio[1])

        return "Success", audio, filename

    return tts_fn

def get_str_list_from_phone(text, text_language):
    print(text)
    texts=text.split("\n")
    phone_list = []
    for text in texts:
        phones2, word2ph2, norm_text2 = clean_text(text, text_language)
        phone_list.append(" ".join(phones2))
    return "\n".join(phone_list)

def get_phone_from_str_list(str_list:str, language:str = 'ja'):
    sentences = str_list.split("\n")
    phones = []
    for sentence in sentences:
        phones.append(sentence.split(" "))
    return phones

splits={"，","。","？","！",",",".","?","!","~",":","：","—","…",}
def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if (todo_text[-1] not in splits): todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while (1):
        if (i_split_head >= len_text): break
        if (todo_text[i_split_head] in splits):
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def change_reference_audio(prompt_text, transcripts):
    return transcripts[prompt_text]

# モデルのパスと設定
vq_model, ssl_model, t2s_model, hps, config, hz, max_sec = load_model("models/sora/sora_e510_s68850.pth", "models/sora/sora-e20.ckpt")
tts_fn = create_tts_fn(vq_model, ssl_model, t2s_model, hps, config, hz, max_sec)

def main():
    
    # 参照音声ファイルの設定
    example_reference = "そこまで図々しくないよ。あたしがこれで寝るんだよ。一回使ってみたかったんだよね～"
    inp_ref_audio = r"C:\Users\user\Downloads\sr0311.wav"
    
    # テキストの入力とトークン化
    text = "私はお兄ちゃんのだいだいだーいすきな妹なんだから、言うことなんでも聞いてくれますよね！"
    text_language = "ja"
    
    # テキストをトークンに変換
    cleaned_text = get_str_list_from_phone(text, text_language)
    
    # 音声を生成
    output_message, output_audio, output_file = tts_fn(
        inp_ref_audio, example_reference, "ja", cleaned_text, text_language, text
    )
    
    # 結果の表示
    print("生成されたメッセージ:", output_message)
    print("生成された音声ファイルのパス:", output_file)

if __name__ == "__main__":
    main()

