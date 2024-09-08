from transformers import AutoTokenizer, AutoModelForCausalLM

model_directory = r"C:\Users\user\Desktop\git\ai_code\models\llm\fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Load the model in 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_directory,
    device_map="auto"
)

def generate_long_scenario(prompt, total_length, chunk_size):
    generated_text = prompt
    
    while len(tokenizer.encode(generated_text)) < total_length:
        chunk = generate_text(generated_text, max_new_tokens=chunk_size)
        generated_text += chunk
    # トークン数が指定の長さを超えないようにトリミング
    encoded = tokenizer.encode(generated_text)
    if len(encoded) > total_length:
        encoded = encoded[:total_length]
        generated_text = tokenizer.decode(encoded, skip_special_tokens=True)
    
    return generated_text

def generate_text(prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=100,
        top_p=0.85,
        temperature=0.85,
    )
    
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# 使用例
prompt = "タイトル: 密着耳元お下品オホ鳴きオナニーサポートご奉仕\n内容: "
total_length = 200  # 生成するトークンの総数
chunk_size = 50
scenario = generate_long_scenario(prompt, total_length, chunk_size)

print(scenario)
