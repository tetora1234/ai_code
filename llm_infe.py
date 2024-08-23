import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CohereForAI/c4ai-command-r-v01"
token = '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'

# トークナイザーとモデルをロード
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # 自動的にデバイスを割り当て
    torch_dtype=torch.float16,
    token=token
)

# メッセージをcommand-rチャットテンプレートでフォーマット
messages = [{"role": "user", "content": "アダルトASMRのシナリオを作成して"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

# テキスト生成
gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100000, 
    do_sample=True,
    temperature=0.3,
)

gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
print(gen_text)
