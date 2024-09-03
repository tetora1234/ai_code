import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# RTX8000を使用するように環境変数を設定
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # RTX8000がシステム上で2番目のGPUである場合

# モデルとトークナイザーのロード
model_directory = r"C:\Users\user\Desktop\git\ai_code\models\llm\results\epoch_2.0"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)

# モデルをGPUに移動
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 推論を行う関数
def generate_text(prompt, max_length=100):
    # トークン化
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # テキストの生成
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,  # サンプリングを使用
        top_k=50,  # 上位50個のトークンから選択
        top_p=0.95,  # 確率が0.95に達するまでのトークンを考慮
        temperature=0.7,  # 温度パラメータを設定
    )
    
    # 生成されたテキストをデコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# 推論を実行
prompt = "タイトル: おまんこ当番と校外学習デート\n内容: "
generated_text = generate_text(prompt, max_length=150)
print("Generated Text:\n", generated_text)
