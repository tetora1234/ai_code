import os
import shutil
import pandas as pd

# CSVファイルのパス
csv_file = r"C:\Users\user\Desktop\git\ai_code\system\models\yumemi\filter_Speaker.csv"
# フォルダのパス
normal_folder = r"C:\Users\user\Desktop\git\ai_code\system\models\yumemi\normal_folder"
chupa_folder = r"C:\Users\user\Desktop\git\ai_code\system\models\yumemi\chupa_folder"
aegi_folder = r"C:\Users\user\Desktop\git\ai_code\system\models\yumemi\aegi_folder"

# フォルダが存在しない場合は作成
os.makedirs(normal_folder, exist_ok=True)
os.makedirs(chupa_folder, exist_ok=True)
os.makedirs(aegi_folder, exist_ok=True)

# CSVファイルを読み込む
df = pd.read_csv(csv_file)

# 各分類ごとのデータフレームを作成
normal_df = pd.DataFrame(columns=df.columns)
chupa_df = pd.DataFrame(columns=df.columns)
aegi_df = pd.DataFrame(columns=df.columns)

# 各ファイルを分類に応じてフォルダにコピーし、新しいパスを更新
for index, row in df.iterrows():
    file_path = row['FilePath']
    classification = row['single_label']

    if os.path.exists(file_path):  # ファイルが存在するか確認
        new_path = None
        if classification == 'usual':
            new_path = os.path.join(normal_folder, os.path.basename(file_path))
            shutil.copy(file_path, new_path)  # ファイルをコピー
            normal_df = pd.concat([normal_df, row.to_frame().T])  # '通常'の行を追加
        elif classification == 'chupa':
            new_path = os.path.join(chupa_folder, os.path.basename(file_path))
            shutil.copy(file_path, new_path)  # ファイルをコピー
            chupa_df = pd.concat([chupa_df, row.to_frame().T])  # 'チュパ'の行を追加
        elif classification == 'aegi':
            new_path = os.path.join(aegi_folder, os.path.basename(file_path))
            shutil.copy(file_path, new_path)  # ファイルをコピー
            aegi_df = pd.concat([aegi_df, row.to_frame().T])  # 'あえぎ'の行を追加
        
        # 新しいパスをDataFrameに更新
        df.at[index, 'FilePath'] = new_path
    else:
        print(f"ファイルが見つかりません: {file_path}")

# 更新したCSVファイルを保存（元のファイルに上書き）
df.to_csv(csv_file, index=False)

# 各分類ごとにCSVファイルを保存
normal_df.to_csv(os.path.join(normal_folder, "normal_files.csv"), index=False)
chupa_df.to_csv(os.path.join(chupa_folder, "chupa_files.csv"), index=False)
aegi_df.to_csv(os.path.join(aegi_folder, "aegi_files.csv"), index=False)

print("ファイルのコピーが完了し、各カテゴリのCSVファイルが保存されました。")
