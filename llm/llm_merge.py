import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 定数の定義
CHECKPOINT_DIR = r"C:\Users\user\Desktop\git\ai_code\llm\models\kagemusya-7B-v1.5_asmr_v1\checkpoint-11365"
SAVE_DIRECTORY = r"C:\Users\user\Desktop\git\ai_code\llm\models\kagemusya-7B-v1.5_asmr_v1\merged_model"
MODEL_NAME = "akineAItech/kagemusya-7B-v1.5"

# GPU設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 量子化設定の準備
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 量子化を適用したモデルのロード
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    use_auth_token=True,
    #device_map="auto",
    quantization_config=quantization_config, #量子化する場合
)

# モデルとトークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
#model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=True, device_map="auto")

# LoRAアダプターの読み込み
lora_model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)

# モデルのマージ
merged_model = lora_model.merge_and_unload()

# モデルの保存
merged_model.save_pretrained(SAVE_DIRECTORY)
tokenizer.save_pretrained(SAVE_DIRECTORY)
print(f"Saving merged model to {SAVE_DIRECTORY}")
