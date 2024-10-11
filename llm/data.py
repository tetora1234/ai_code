import json

# 元のファイルパスと新しいファイルパスを指定
input_file_path = r"C:\Users\user\Desktop\git\ai_code\llm\dataset\data.json"
output_file_path = r"C:\Users\user\Desktop\git\ai_code\llm\dataset\filtered_data.json"

# JSONファイルを読み込む
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 文字数が2000を超えるデータのみをフィルタリング
filtered_data = [item for item in data if 2000 < len(item.get('内容', '')) <= 7000]

# 新しいJSONファイルに保存
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=4)

print(f"フィルタリングされたデータは {output_file_path} に保存されました。")
