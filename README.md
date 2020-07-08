# One class Mask-RCNN/単一クラス用Mask-RCNN

このリポジトリはInstance segmentationを単一クラスに対して簡単に適用するための https://github.com/matterport/Mask_RCNN のラッパーです。

環境構築後、元画像-二値化マスク画像をそれぞれ`dataset`の`image`および`mask`ディレクトリに配置するだけで準備は完了です。<br>

注意：<br>
二値化マスクは検出したい物体が分離されている必要があります。<br>
これはopencvのブロブ検出関数によって二値化マスク画像から各物体を検出するためです。

<br>

1. aa
   `python -m src prepare -d ./dataset`
   転移学習用の訓練済み重みdownloadを含むのでproxy下のときは注意

2. aa
   `python -m src train -d ./dataset`

3. aa
   `python -m src validation -d ./dataset`

<br>

実行結果はすべて`dataset/logs`に出力されます。

<br>

### 環境作成

※windows10でのみ動作確認<br>
※tensorflowだけは事前にインストールしといた方が安全

```
conda create -n mrcnn python=3.7

conda activate mrcnn

cd ./Mask_RCNN

pip install -r requirements.txt

python setup.py install

pip install click

```


<br>

### その他
pretrained weightsのhdfはOS依存のようなので開けなかったら消して再ダウンロード
