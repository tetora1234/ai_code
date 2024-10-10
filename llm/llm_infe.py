import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from datetime import datetime
import json
import random

# RTX8000を使用するように環境変数を設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

model_directory = r"C:\Users\user\Desktop\git\ai_code\llm\models\kagemusya-7B-v1.5_asmr_v2\merged_model_checkpoint-539"
json_file_path = r"C:\Users\user\Desktop\git\ai_code\llm\dataset\data.json"

tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(
    model_directory,
    device_map="auto"
)

# filtered_data.jsonからタイトルをランダムに取得
def get_random_title_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # JSONファイルがリスト形式で、各項目にタイトルが含まれていると仮定
    titles = [item["タイトル"] for item in data if "タイトル" in item]
    
    # ランダムにタイトルを選択
    random_title = random.choice(titles)
    return random_title

# 初期プロンプトを更新
def create_initial_prompt():
    random_title = get_random_title_from_json(json_file_path)
    
    # タイトルを使用してプロンプトを作成
    initial_prompt = f"タイトル: {random_title} 内容: "
    return initial_prompt

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # TextStreamerのインスタンスを作成（標準出力にテキストをリアルタイム表示）
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=10000,
        do_sample=True,
        top_k=5,
        temperature=0.8,
        repetition_penalty=1.1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    new_tokens = outputs[0][inputs["input_ids"].size(1):]
    
    # skip_special_tokens=Falseにして</s>の確認を行う
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated_text

def save_generated_text(generated_text, prompt):
    current_time = datetime.now().strftime("%y%m%d%H%M%S")

    prompt = prompt.replace("内容: ", "")
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
        f.write(f"{generated_text}")

while True:
    initial_prompt = create_initial_prompt()
    print(initial_prompt)
    generated_text = generate_text(initial_prompt)
    save_generated_text(generated_text, initial_prompt)
