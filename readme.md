# aim

RealSense L515で取得したbagファイルをmediaPipeで二次元姿勢推定し、
取得ピクセル位置から三次元座標の復元を行う、冗長なアプリケーションです。
mediaPipe側で三次元姿勢推定は可能ですが、realsenseを使用したいためこのような構成にしています。

# installing

1. install python 3.7. (should install latest atable version)
2. install pipenv with using pip. The command line code is :`pip install pipenv`
3. use pipenv command. :`pipenv sync`

## run notification

You must change python interpleter if you use other version python interpleter in VSCode.
After check it, open terminal and command `pipenv run start`.
in details pipenv run start means `pipenv run python apps -m`

※ need to make dir apps/src

※ change config if you need ( config file is in apps/config )

## 開発時参考文献

[ユーザー定義クラスについて](https://python.keicode.com/lang/python-user-defined-exception.php)
