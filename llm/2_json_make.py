import os
import json

# フォルダパスを指定
folder_path = r"C:\Users\user\Desktop\txt"
output_file = "output.json"

# 出力用のリスト
data = []

# フォルダ内のファイルを読み込む
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                # ファイル内容をパース
                i = 0
                while i < len(lines):
                    title_line = lines[i].strip()
                    if title_line.startswith("ファイル名: "):
                        title = title_line.replace("ファイル名: ", "")
                        i += 1
                        if i < len(lines):
                            content = lines[i].strip()
                            data.append({"タイトル": title, "内容": content})
                    i += 1
        except Exception as e:
            print(f"エラーが発生しました。ファイル名: {filename}, エラー内容: {e}")

# JSONファイルとして保存
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=2)

print(f"データが{output_file}に保存されました。")
