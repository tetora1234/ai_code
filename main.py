import user_module

# モデルのパスと設定
vq_model, ssl_model, t2s_model, hps, config, hz, max_sec = user_module.load_model("models/sora/sora_e510_s68850.pth", "models/sora/sora-e20.ckpt")
tts_fn =  user_module.create_tts_fn(vq_model, ssl_model, t2s_model, hps, config, hz, max_sec)

def main():
    
    # 参照音声ファイルの設定
    example_reference = "そこまで図々しくないよ。あたしがこれで寝るんだよ。一回使ってみたかったんだよね～"
    inp_ref_audio = r"C:\Users\user\Desktop\git\ai_code\GPT-SoVITS-V2_Easy\sr0311.wav"
    
    # テキストの入力とトークン化
    text = "私はお兄ちゃんのだいだいだーいすきな妹なんだから、言うことなんでも聞いてくれますよね！"
    text_language = "ja"
    
    # テキストをトークンに変換
    cleaned_text =  user_module.get_str_list_from_phone(text, text_language)
    
    # 音声を生成
    output_message, output_audio, output_file = tts_fn(
        inp_ref_audio, example_reference, "ja", cleaned_text, text_language, text
    )
    
    # 結果の表示
    print("生成されたメッセージ:", output_message)
    print("生成された音声ファイルのパス:", output_file)

if __name__ == "__main__":
    main()