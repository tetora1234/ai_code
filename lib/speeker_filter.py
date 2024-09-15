import json

# JSONファイルのパス
input_file_path = r"C:\Users\user\Downloads\ωstar_Bishoujo Mangekyou Ibun - Yuki Onna\index.json"
output_file_path = r"C:\Users\user\Downloads\ωstar_Bishoujo Mangekyou Ibun - Yuki Onna\out.json"

# 特定のスピーカーを指定
target_speaker = "姫"

# JSONファイルを読み込む
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 指定したスピーカーのエントリを抽出
filtered_data = [entry for entry in data if entry["Speaker"] == target_speaker]

# 結果を新しいJSONファイルに保存
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(filtered_data, file, indent=2, ensure_ascii=False)

print(f"Filtered data has been saved to {output_file_path}")
