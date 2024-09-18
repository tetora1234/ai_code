from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import os
import torch
from datetime import datetime

# RTX8000を使用するように環境変数を設定
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

model_directory = r"C:\Users\user\Desktop\git\ai_code\models\llm\fine_tuned_model"
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
        max_new_tokens=250,
        do_sample=True,
        top_k=1,
        repetition_penalty=1.1,
        temperature=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    new_tokens = outputs[0][inputs["input_ids"].size(1):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), outputs

def save_generated_text(generated_text, prompt):
    current_time = datetime.now().strftime("%y%m%d%H%M%S")
    filename = f"{current_time}.txt"
    output_directory = r"C:\Users\user\Desktop\git\ai_code\outpus"
    os.makedirs(output_directory, exist_ok=True)
    file_path = os.path.join(output_directory, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{prompt}{generated_text}")

def generate_full_text(prompt, initial_prompt):
    generated_text, outputs = generate_text(prompt)
    
    # EOSトークンが含まれているかどうかをチェック
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id in outputs[0].tolist():
        print("EOSトークンが見つかりました。初期プロンプトで再スタートします。")
        prompt = initial_prompt  # プロンプトを初期プロンプトにリセット
    else:
        prompt += generated_text  # プロンプトを更新
    
    save_generated_text(generated_text, prompt)
    return prompt

# 使用例
initial_prompt = """タイトル: 唾液たっぷりドスケベフェラ純愛オホ声えっち\n内容: """
generated_text = initial_prompt

while True:
    generated_text = generate_full_text(generated_text, initial_prompt)
