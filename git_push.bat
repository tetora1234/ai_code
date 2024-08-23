@echo off
REM 変更をステージングする
git add .

REM コミットメッセージを入力
set /p commitMsg="Enter commit message: "

REM コミットを実行
git commit -m "%commitMsg%"

REM プッシュを実行
git push origin master
