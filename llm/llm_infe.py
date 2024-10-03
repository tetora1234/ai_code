from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import os
import torch
from datetime import datetime

# RTX8000を使用するように環境変数を設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

model_directory = r"C:\Users\user\Desktop\git\ai_code\llm\models\merged_model"
#model_directory = "akineAItech/kagemusya-7B-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(
    model_directory,
    device_map="auto"
)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # TextStreamerのインスタンスを作成（標準出力にテキストをリアルタイム表示）
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=10000,
        do_sample=True,
        top_k=3,
        temperature=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    new_tokens = outputs[0][inputs["input_ids"].size(1):]
    
    # skip_special_tokens=Falseにして</s>の確認を行う
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return generated_text

def save_generated_text(generated_text, prompt):
    current_time = datetime.now().strftime("%y%m%d%H%M%S")
    # 「タイトル: 」の部分を探して、その後の文字列を抜き出す
    title_prefix = "タイトル: "
    start_index = prompt.find(title_prefix) + len(title_prefix)
    end_index = prompt.find("\n", start_index)

    # タイトル部分を抜き出す
    title = prompt[start_index:end_index].strip()

    filename = f"{title}_{current_time}.txt"
    output_directory = r"C:\Users\user\Desktop\git\ai_code\llm\outputs"
    os.makedirs(output_directory, exist_ok=True)
    file_path = os.path.join(output_directory, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{prompt}{generated_text}")

def generate_full_text(prompt, initial_prompt):
    while True:
        generated_text = generate_text(prompt)
        
        # 生成されたテキストに�が含まれているか確認
        if "�" in generated_text:
            print("�が含まれています。再試行します。")
            continue  # 再試行する
        else:
            break  # �が含まれていない場合はループを抜ける
    
    # EOSトークンとして"</s>"が含まれているかを確認
    if "</s>" in generated_text:
        save_generated_text(generated_text, prompt)
        print("EOSトークンが見つかりました。初期プロンプトで再スタートします。")
        prompt = initial_prompt  # プロンプトを初期プロンプトにリセット
    else:
        prompt += generated_text  # プロンプトを更新
    
    return prompt

# 使用例
initial_prompt = """タイトル: ゲームしながら自由におまんこを使わせてくれるゲーマーカノジョ\n内容: """

generated_text = initial_prompt

while True:
    generated_text = generate_full_text(generated_text, initial_prompt)
