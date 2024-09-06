import json

# JSONファイルのパス
json_file_path = r'C:\Users\user\Desktop\git\ai_code\dataset\llm\output.json'
# 保存先の新しいJSONファイルのパス
filtered_json_file_path = r'C:\Users\user\Desktop\git\ai_code\dataset\llm\filtered_output.json'

# JSONファイルを読み込む
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 長さ2048以下のアイテムをフィルタリングする
filtered_data = []
for item in data:
    content = item.get('内容', '')
    content_length = len(content)
    if content_length >= 3000 and content_length <= 4000:
        filtered_data.append(item)
    else:
        print(f"削除対象: タイトル: {item.get('タイトル', 'なし')}, 内容の長さ: {content_length}")

# フィルタリングされたデータを新しいJSONファイルに保存する
with open(filtered_json_file_path, 'w', encoding='utf-8') as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=4)

print(f"フィルタリングされたデータが {filtered_json_file_path} に保存されました")
